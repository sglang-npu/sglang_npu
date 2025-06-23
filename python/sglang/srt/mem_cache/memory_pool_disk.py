import abc
import logging
import threading

import numpy as np
import torch

from sglang.srt.mem_cache.memory_pool import KVCache, MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import synchronized

logger = logging.getLogger(__name__)


class DiskKVCache(abc.ABC):
    def __init__(
        self,
        device_pool: KVCache,
        disk_path: str,
        disk_to_device_ratio: float,
        disk_size: int,
        page_size: int,
        gpu_id: int,
    ):
        self.device_pool = device_pool
        self.dtype = device_pool.dtype
        self.page_size = page_size
        self.gpu_id = gpu_id
        self.size_per_token = self.get_size_per_token()
        if disk_size > 0:
            self.size = int(disk_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * disk_to_device_ratio)
        # Align the disk pool size to the page size
        self.size = self.size - (self.size % self.page_size)
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer

        # TODO
        # The disk should be larger than the host memory.
        # Verify there is enough available disk.
        # preserve at least xxxGB for other usage.
        # self.mem_state

        requested_bytes = self.size * self.size_per_token
        self.disk_path = disk_path + f".gpu_id-{self.gpu_id}.bin"
        logger.info(
            f"Allocating {requested_bytes / 1e9:.2f} GB ({self.size=}) disk at {self.disk_path} for hierarchical KV cache."
        )

        self.kv_buffer = self.init_kv_buffer()

        # A lock for synchronized operations on memory allocation and state transitions.
        self.lock = threading.RLock()
        self.debug = logger.isEnabledFor(logging.DEBUG)
        self.debug = True
        self.clear()

    @abc.abstractmethod
    def get_size_per_token(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def init_kv_buffer(self):
        raise NotImplementedError()

    def clear(self):
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    def check_size(self, size):
        return size > 0 and size % self.page_size == 0

    @synchronized()
    def alloc(self, need_size: int) -> torch.Tensor:
        assert self.check_size(need_size)

        if need_size > self.available_size():
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    @synchronized()
    def free(self, indices: torch.Tensor) -> int:
        assert self.check_size(len(indices))
        self.free_slots = torch.cat([self.free_slots, indices])
        return len(indices)


class MHATokenToKVPoolDisk(DiskKVCache):
    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        disk_path: str,
        disk_to_device_ratio: float,
        disk_size: int,
        page_size: int,
        gpu_id: int,
    ):
        super().__init__(
            device_pool, disk_path, disk_to_device_ratio, disk_size, page_size, gpu_id
        )

    def get_size_per_token(self):
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.device_pool.layer_num

        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2

    def init_kv_buffer(self):
        shape = (2, self.layer_num, self.size, self.head_num, self.head_dim)
        kv_buffer = torch.from_file(
            self.disk_path,
            shared=True,
            size=np.prod(shape),
            dtype=self.dtype,
        ).view(shape)
        return kv_buffer

    def get_flat_data(self, indices):
        return self.kv_buffer[:, :, indices]

    def assign_flat_data(self, indices, flat_data):
        self.kv_buffer[:, :, indices] = flat_data

    def write_page_all_layers(self, disk_indices, device_indices, device_pool):
        device_indices_cpu = device_indices[:: self.page_size].cpu()
        for i in range(len(device_indices_cpu)):
            h_index = disk_indices[i * self.page_size]
            d_index = device_indices_cpu[i]
            for j in range(self.layer_num):
                self.kv_buffer[0, j, h_index : h_index + self.page_size].copy_(
                    device_pool.k_buffer[j][d_index : d_index + self.page_size],
                    non_blocking=True,
                )
                self.kv_buffer[1, j, h_index : h_index + self.page_size].copy_(
                    device_pool.v_buffer[j][d_index : d_index + self.page_size],
                    non_blocking=True,
                )

    def load_page_per_layer(self, disk_indices, device_indices, device_pool, layer_id):
        device_indices_cpu = device_indices[:: self.page_size].cpu()
        for i in range(len(device_indices_cpu)):
            h_index = disk_indices[i * self.page_size]
            d_index = device_indices_cpu[i]
            device_pool.k_buffer[layer_id - self.start_layer][
                d_index : d_index + self.page_size
            ].copy_(
                self.kv_buffer[
                    0,
                    layer_id - self.start_layer,
                    h_index : h_index + self.page_size,
                ],
                non_blocking=True,
            )
            device_pool.v_buffer[layer_id - self.start_layer][
                d_index : d_index + self.page_size
            ].copy_(
                self.kv_buffer[
                    1,
                    layer_id - self.start_layer,
                    h_index : h_index + self.page_size,
                ],
                non_blocking=True,
            )


class MLATokenToKVPoolDisk(DiskKVCache):
    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        disk_path: str,
        disk_to_device_ratio: float,
        disk_size: int,
        page_size: int,
        gpu_id: int,
    ):
        super().__init__(
            device_pool, disk_path, disk_to_device_ratio, disk_size, page_size, gpu_id
        )

    def get_size_per_token(self):
        self.kv_lora_rank = self.device_pool.kv_lora_rank
        self.qk_rope_head_dim = self.device_pool.qk_rope_head_dim
        self.layer_num = self.device_pool.layer_num

        return (
            (self.kv_lora_rank + self.qk_rope_head_dim)
            * 1
            * self.dtype.itemsize
            * self.layer_num
        )

    def init_kv_buffer(self):
        shape = (
            self.layer_num,
            self.size,
            1,
            self.kv_lora_rank + self.qk_rope_head_dim,
        )
        kv_buffer = torch.from_file(
            self.disk_path,
            shared=True,
            size=np.prod(shape),
            dtype=self.dtype,
        ).view(shape)
        return kv_buffer

    def get_flat_data(self, indices):
        return self.kv_buffer[:, indices]

    def assign_flat_data(self, indices, flat_data):
        self.kv_buffer[:, indices] = flat_data

    def write_page_all_layers(self, disk_indices, device_indices, device_pool):
        device_indices_cpu = device_indices[:: self.page_size].cpu()
        for i in range(len(device_indices_cpu)):
            h_index = disk_indices[i * self.page_size]
            d_index = device_indices_cpu[i]
            for j in range(self.layer_num):
                self.kv_buffer[j, h_index : h_index + self.page_size].copy_(
                    device_pool.kv_buffer[j][d_index : d_index + self.page_size],
                    non_blocking=True,
                )

    def load_page_per_layer(self, disk_indices, device_indices, device_pool, layer_id):
        device_indices_cpu = device_indices[:: self.page_size].cpu()
        for i in range(len(device_indices_cpu)):
            h_index = disk_indices[i * self.page_size]
            d_index = device_indices_cpu[i]
            device_pool.kv_buffer[layer_id - self.start_layer][
                d_index : d_index + self.page_size
            ].copy_(
                self.kv_buffer[
                    layer_id - self.start_layer, h_index : h_index + self.page_size
                ],
                non_blocking=True,
            )
