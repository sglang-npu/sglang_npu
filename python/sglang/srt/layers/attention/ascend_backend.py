from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch_npu
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class ForwardMetadata:

    # calculated map for kv positions [bs * maxseqlen]
    block_tables: Optional[torch.Tensor] = None

    # seq len inputs
    extend_seq_lens_cpu_int: Optional[torch.Tensor] = None
    seq_lens_cpu_int: Optional[torch.Tensor] = None


def _generate_attn_mask(max_seq_len, dtype):
    # Construct lower triangle matrix.
    mask_flag = torch.tril(
        torch.ones((max_seq_len, max_seq_len), dtype=torch.bool)
    ).view(max_seq_len, max_seq_len)
    # Create upper triangle matrix used to mark mask positions.
    mask_flag = ~mask_flag
    # Currently for fp16 dtype, the mask value should be set to -inf.
    # TODO: Eliminate this part in the future.
    if dtype == torch.float16:
        mask_value = torch.finfo(torch.float32).min
    else:
        mask_value = 1
    attn_mask = torch.masked_fill(
        torch.zeros(size=(max_seq_len, max_seq_len)), mask_flag, mask_value
    ).to(dtype)
    return attn_mask


class AttentionMaskBuilder:

    def __init__(
        self,
        max_seq_len: int,
        dtype: torch.dtype,
    ):
        attn_mask = _generate_attn_mask(max_seq_len, dtype)

        self._seq_len_cached = attn_mask.shape[0]
        self.attn_mask_cache = attn_mask

    def get_attn_mask(self, max_seq_len: int, dtype: torch.dtype, device: torch.device):
        self._update_attn_cache(max_seq_len, dtype, device)
        return self.attn_mask_cache[:max_seq_len, :max_seq_len].contiguous()

    def _update_attn_cache(self, seqlen: int, dtype: torch.dtype, device: torch.device):
        if seqlen > self._seq_len_cached:
            self._seq_len_cached = seqlen
            self.attn_mask_cache = _generate_attn_mask(seqlen, dtype)
        if self.attn_mask_cache.device != device:
            self.attn_mask_cache = self.attn_mask_cache.to(device)


class AscendAttnBackend(AttentionBackend):

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = ForwardMetadata()
        self.device = model_runner.device
        self.attn_mask_builder = AttentionMaskBuilder(8192, model_runner.dtype)
        self.page_size = model_runner.page_size
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        if self.use_mla:
            self.kv_lora_rank = model_runner.model_config.kv_lora_rank
            self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
            self.native_attn = TorchNativeAttnBackend(model_runner)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        self.forward_metadata.block_tables = (
            forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : forward_batch.seq_lens.max()
            ][:, :: self.page_size]
            // self.page_size
        )
        if forward_batch.extend_seq_lens is not None:
            self.forward_metadata.extend_seq_lens_cpu_int = (
                forward_batch.extend_seq_lens.cpu().int()
            )
        self.forward_metadata.seq_lens_cpu_int = forward_batch.seq_lens_cpu.int()

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        has_prefixcache = sum(forward_batch.extend_prefix_lens_cpu) > 0
        if not self.use_mla:
            query = q.view(-1, layer.tp_q_head_num * layer.qk_head_dim)
            output = torch.empty(
                (query.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                dtype=query.dtype,
                device=query.device,
            )
            mask = self.attn_mask_builder.get_attn_mask(128, query.dtype, query.device)
            torch_npu._npu_flash_attention_qlens(
                query=query,
                key_cache=k_cache,
                value_cache=v_cache,
                mask=mask,
                block_table=self.forward_metadata.block_tables,
                seq_len=self.forward_metadata.extend_seq_lens_cpu_int,
                context_lens=self.forward_metadata.seq_lens_cpu_int,
                scale_value=layer.scaling,
                num_heads=layer.tp_q_head_num,
                num_kv_heads=layer.tp_k_head_num,
                out=output,
            )
            return output
        elif has_prefixcache:
            raise NotImplementedError(
                f"prefixcache attention is not implemented, seqlen len: {forward_batch.seq_lens_cpu}, \
                    extend_prefix_lens: {forward_batch.extend_prefix_lens_cpu}"
            )
        else:
            attn_output = torch.empty(
                (q.shape[0], layer.tp_q_head_num, layer.v_head_dim),
                device=q.device,
                dtype=q.dtype,
            )
            max_s = max(self.forward_metadata.seq_lens_cpu_int)
            mask = self.attn_mask_builder.get_attn_mask(max_s, q.dtype, q.device)
            torch_npu._npu_flash_attention(
                query=q,
                key=k,
                value=v,
                mask=mask,
                seq_len=self.forward_metadata.seq_lens_cpu_int,
                scale_value=layer.scaling,
                num_heads=layer.tp_q_head_num,
                num_kv_heads=layer.tp_k_head_num,
                out=attn_output,
            )
            attn_output = attn_output.reshape(-1, layer.tp_q_head_num, layer.v_head_dim)
            return attn_output

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )
        if not self.use_mla:
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

            query = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
            num_tokens = query.shape[0]
            output = torch.empty(
                (num_tokens, layer.tp_q_head_num, layer.v_head_dim),
                dtype=query.dtype,
                device=query.device,
            )

            torch_npu._npu_paged_attention(
                query=query,
                key_cache=k_cache,
                value_cache=v_cache,
                num_heads=layer.tp_q_head_num,
                num_kv_heads=layer.tp_k_head_num,
                scale_value=layer.scaling,
                block_table=self.forward_metadata.block_tables,
                context_lens=self.forward_metadata.seq_lens_cpu_int,
                out=output,
            )
            return output.view(num_tokens, layer.tp_q_head_num * layer.v_head_dim)
        else:
            query = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            num_tokens = query.shape[0]
            kv_c_and_k_pe_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                layer.layer_id
            )
            kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
                -1,
                self.page_size,
                layer.tp_k_head_num,
                self.kv_lora_rank + self.qk_rope_head_dim,
            )

            attn_output = torch.empty(
                [num_tokens, layer.tp_q_head_num, self.kv_lora_rank],
                dtype=q.dtype,
                device=q.device,
            )
            torch_npu._npu_paged_attention_mla(
                query=query,
                key_cache=kv_c_and_k_pe_cache,
                num_kv_heads=layer.tp_k_head_num,
                num_heads=layer.tp_q_head_num,
                scale_value=layer.scaling,
                block_table=self.forward_metadata.block_tables,
                context_lens=self.forward_metadata.seq_lens_cpu_int,
                mla_vheadsize=self.kv_lora_rank,
                out=attn_output,
            )
            return attn_output.view(num_tokens, layer.tp_q_head_num * self.kv_lora_rank)
