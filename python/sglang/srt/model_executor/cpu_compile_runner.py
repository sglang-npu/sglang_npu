"""Modified from cuda_graph_runner.py"""

from __future__ import annotations

import bisect
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import tqdm

from sglang.srt.custom_op import CustomOp
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
# from sglang.srt.layers.moe.fused_moe_native import fused_moe_forward_native
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


# def _to_torch(model: torch.nn.Module, reverse: bool, num_tokens: int):
#     for sub in model._modules.values():
#         if isinstance(sub, CustomOp):
#             if reverse:
#                 sub._forward_method = sub.forward_cuda
#                 setattr(sub, "is_torch_compile", False)
#             else:
#                 # NOTE: Temporarily workaround MoE
#                 if "FusedMoE" in sub.__class__.__name__:
#                     if num_tokens == 1:
#                         # The performance of torch.compile on this layer is not always good when bs > 1,
#                         # so we decide to only use torch.compile when bs =1
#                         sub._forward_method = fused_moe_forward_native
#                 else:
#                     sub._forward_method = sub.forward_native
#                 setattr(sub, "is_torch_compile", True)
#         if isinstance(sub, torch.nn.Module):
#             _to_torch(sub, reverse, num_tokens)


@contextmanager
def patch_model(
    model: torch.nn.Module,
    enable_compile: bool,
    num_tokens: int,
    tp_group: GroupCoordinator,
):
    """Patch the model to make it compatible with with torch.compile"""
    backup_ca_comm = None

    try:
        if enable_compile:
            # _to_torch(model, reverse=False, num_tokens=num_tokens)  # not sure why this is needed
            backup_ca_comm = tp_group.ca_comm
            # Use custom-allreduce here.
            # We found the custom allreduce is much faster than the built-in allreduce in torch,
            # even with ENABLE_INTRA_NODE_COMM=1.
            # tp_group.ca_comm = None
            yield torch.compile(
                torch.no_grad()(model.forward),
                fullgraph=True,
                dynamic=True, # TODO explore a best way to set dynamic
            )
        else:
            yield model.forward
    finally:
        if enable_compile:
            # _to_torch(model, reverse=True, num_tokens=num_tokens)
            tp_group.ca_comm = backup_ca_comm


def set_torch_compile_config():
    import torch._dynamo.config
    import torch._inductor.config

    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

    # FIXME: tmp workaround
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 1024


def get_batch_sizes_to_compile(model_runner: ModelRunner):
    # NOTE: may want to simplify this
    server_args = model_runner.server_args
    capture_bs = server_args.cuda_graph_bs

    if capture_bs is None:
        if server_args.speculative_algorithm is None:
            if server_args.disable_cuda_graph_padding:
                capture_bs = list(range(1, 33)) + [64, 96, 128, 160]
            else:
                capture_bs = [1, 2, 4, 7] + [i * 8 for i in range(1, 21)]
        else:
            capture_bs = list(range(1, 33))

    if max(capture_bs) > model_runner.req_to_token_pool.size:
        # In some case (e.g., with a small GPU or --max-running-requests), the #max-running-requests
        # is very small. We add more values here to make sure we capture the maximum bs.
        capture_bs = list(
            sorted(
                set(
                    capture_bs
                    + [model_runner.req_to_token_pool.size - 1]
                    + [model_runner.req_to_token_pool.size]
                )
            )
        )

    capture_bs = [
        bs
        for bs in capture_bs
        # if bs <= model_runner.req_to_token_pool.size
        # and bs <= server_args.cuda_graph_max_bs
    ]
    compile_bs = (
        [bs for bs in capture_bs if bs <= server_args.torch_compile_max_bs]
        if server_args.enable_torch_compile
        else []
    )
    # TODO improve speculative bs
    return compile_bs


class CpuCompileRunner:
    """A CpuCompileRunner runs the forward pass of a model with torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        # Parse args
        self.model_runner = model_runner
        # self.graphs = {}
        # self.output_buffers = {}
        self.compiled_forward = None
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.is_encoder_decoder = model_runner.model_config.is_encoder_decoder
        self.enable_dp_attention = model_runner.server_args.enable_dp_attention
        self.tp_size = model_runner.server_args.tp_size
        self.dp_size = model_runner.server_args.dp_size

        # Batch sizes to capture
        self.compile_bs = get_batch_sizes_to_compile(model_runner)
        self.capture_forward_mode = ForwardMode.DECODE
        self.capture_hidden_mode = CaptureHiddenMode.NULL
        self.num_tokens_per_bs = 1
        if model_runner.spec_algorithm.is_eagle():
            if self.model_runner.is_draft_worker:
                raise RuntimeError("This should not happen")
            else:
                self.capture_forward_mode = ForwardMode.TARGET_VERIFY
                self.num_tokens_per_bs = (
                    self.model_runner.server_args.speculative_num_draft_tokens
                )

        # Attention backend
        self.max_bs = max(self.compile_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs
        # self.model_runner.attn_backend.init_cuda_graph_state(self.max_num_token)
        # self.seq_len_fill_value = (
        #     self.model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
        # )
        self.seq_len_fill_value = 0
        # FIXME(lsyin): leave it here for now, I don't know whether it is necessary
        self.encoder_len_fill_value = 0
        self.seq_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
        )

        if self.enable_torch_compile:
            set_torch_compile_config()

        # Graph inputs
        # NOTE: we don't actually need this
        with torch.device("cpu"):
            # use dtype=torch.int32 to align with the benchmark
            self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int64)
            self.seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int64
            )
            self.out_cache_loc = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.mrope_positions = torch.zeros((3, self.max_bs), dtype=torch.int64)
            self.num_token_non_padded = torch.zeros((1,), dtype=torch.int64)

            # Speculative_inference
            if model_runner.spec_algorithm.is_eagle():
                self.hidden_states = torch.zeros(
                    (self.max_num_token, self.model_runner.model_config.hidden_size),
                    dtype=self.model_runner.dtype,
                )

            if self.is_encoder_decoder:
                # NOTE: encoder_lens can influence the full_text_row_masked_out_mask tensor when doing mixed batch
                self.encoder_lens = torch.full(
                    (self.max_bs,), self.encoder_len_fill_value, dtype=torch.int64
                )
            else:
                self.encoder_lens = None

            if self.enable_dp_attention:
                self.gathered_buffer = torch.zeros(
                    (
                        self.max_bs * self.dp_size,
                        self.model_runner.model_config.hidden_size,
                    ),
                    dtype=self.model_runner.dtype,
                )

        # Capture
        try:
            with self.model_capture_mode():
                self.capture()
        except RuntimeError as e:
            import traceback
            raise Exception(
                f"CPU compile failed: {e}\n"
                f"{traceback.format_exc()}\n"
                "Possible solutions:\n"
                # "1. disable cuda graph by --disable-cuda-graph\n"
                "2. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
                "3. disable torch compile by not using --enable-torch-compile\n"
                "4. set --cuda-graph-max-bs to a smaller value (e.g., 32)\n"
                "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
            )

    @contextmanager
    def model_capture_mode(self):
        if hasattr(self.model_runner.model, "capture_mode"):
            self.model_runner.model.capture_mode = True

        yield

        if hasattr(self.model_runner.model, "capture_mode"):
            self.model_runner.model.capture_mode = False

    def can_run(self, forward_batch: ForwardBatch):
        # TODO update the check for dynamic compile
        return True

        if self.enable_dp_attention:
            min_num_tokens, max_num_tokens = min(
                forward_batch.global_num_tokens_cpu
            ), max(forward_batch.global_num_tokens_cpu)
            is_bs_supported = forward_batch.can_run_dp_cuda_graph and (
                (min_num_tokens == max_num_tokens and max_num_tokens in self.compiled_forward)
                if self.disable_padding
                else max_num_tokens <= self.max_bs
            )
        else:
            is_bs_supported = (
                forward_batch.batch_size in self.compiled_forward
                if self.disable_padding
                else forward_batch.batch_size <= self.max_bs
            )

        # TODO: check this
        # NOTE: cuda graph cannot handle mixed batch (encoder_len = 0)
        # If mixed batch cannot be supported, then encoder_lens can be removed in cuda graph
        # because the full_text_row_masked_out_mask tensor will always be ones
        is_encoder_lens_supported = (
            torch.all(forward_batch.encoder_lens > 0)
            if self.is_encoder_decoder
            else True
        )
        return is_bs_supported and is_encoder_lens_supported

    def capture(self):
        # # TODO: see if this can be removed
        # # Reverse the order to enable better memory sharing across cuda graphs.
        # capture_range = (
        #     tqdm.tqdm(list(reversed(self.compile_bs)))
        #     if get_tensor_model_parallel_rank() == 0
        #     else reversed(self.compile_bs)
        # )
        with patch_model(
            self.model_runner.model,
            True,
            num_tokens=1 * self.num_tokens_per_bs,
            tp_group=self.model_runner.tp_group,
        ) as forward:
            # for bs in capture_range:
            #     self.capture_one_batch_size(bs, forward)
            self.compiled_forward = forward
                # self.graphs[bs] = graph
                # self.output_buffers[bs] = output_buffers

    def capture_one_batch_size(self, bs: int, forward: Callable):
        num_tokens = bs * self.num_tokens_per_bs

        # Graph inputs
        input_ids = self.input_ids[:num_tokens]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        out_cache_loc = self.out_cache_loc[:num_tokens]
        positions = self.positions[:num_tokens]
        if self.is_encoder_decoder:
            encoder_lens = self.encoder_lens[:bs]
        else:
            encoder_lens = None
        mrope_positions = self.mrope_positions[:, :bs]
        self.num_token_non_padded[...] = num_tokens

        if self.enable_dp_attention:
            global_num_tokens = [bs] * self.tp_size
            gathered_buffer = self.gathered_buffer[: bs * self.tp_size]
        else:
            global_num_tokens = None
            gathered_buffer = None

        spec_info = self.get_spec_info(num_tokens)
        if self.capture_hidden_mode != CaptureHiddenMode.FULL:
            self.capture_hidden_mode = (
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        forward_batch = ForwardBatch(
            forward_mode=self.capture_forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum(),
            encoder_lens=encoder_lens,
            return_logprob=False,
            positions=positions,
            global_num_tokens_cpu=global_num_tokens,
            gathered_buffer=gathered_buffer,
            mrope_positions=mrope_positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=self.capture_hidden_mode,
            num_token_non_padded=self.num_token_non_padded,
            global_forward_mode=self.capture_forward_mode,
            lora_paths=None,
        )

        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)

        # trigger torch.compile()
        with torch.no_grad():
            for _ in range(2):
                self.model_runner.tp_group.barrier()
                forward(input_ids, positions, forward_batch)

    def recapture_if_needed(self, forward_batch: ForwardBatch):
        # If the capture_hidden_mode changes, we need to recapture the graph
        hidden_mode_from_spec_info = getattr(
            forward_batch.spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL
        )
        if (
            forward_batch.capture_hidden_mode == CaptureHiddenMode.FULL
            and self.capture_hidden_mode != CaptureHiddenMode.FULL
        ):
            self.capture_hidden_mode = CaptureHiddenMode.FULL
            self.capture()
        elif (
            forward_batch.capture_hidden_mode != CaptureHiddenMode.FULL
            and self.capture_hidden_mode != hidden_mode_from_spec_info
        ):
            self.capture_hidden_mode = hidden_mode_from_spec_info
            self.capture()

    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        # self.recapture_if_needed(forward_batch)

        # raw_bs = forward_batch.batch_size
        # raw_num_token = raw_bs * self.num_tokens_per_bs

        # # Pad
        # if self.enable_dp_attention:
        #     index = bisect.bisect_left(
        #         self.compile_bs, max(forward_batch.global_num_tokens_cpu)
        #     )
        # else:
        #     index = bisect.bisect_left(self.compile_bs, raw_bs)
        # bs = self.compile_bs[index]
        # if bs != raw_bs:
        #     # raise
        #     self.seq_lens.fill_(1)
        #     self.out_cache_loc.zero_()

        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)

        # Replay
        logits_output = self.compiled_forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
        )
        return logits_output

    def get_spec_info(self, num_tokens: int):
        spec_info = None
        if self.model_runner.spec_algorithm.is_eagle():
            from sglang.srt.speculative.eagle_utils import EagleVerifyInput

            if self.model_runner.is_draft_worker:
                raise RuntimeError("This should not happen.")
            else:
                spec_info = EagleVerifyInput(
                    draft_token=None,
                    custom_mask=torch.zeros(
                        (num_tokens * self.model_runner.model_config.context_len),
                        dtype=torch.bool,
                        device="cuda",
                    ),
                    positions=None,
                    retrive_index=None,
                    retrive_next_token=None,
                    retrive_next_sibling=None,
                    retrive_cum_len=None,
                    draft_token_num=self.model_runner.server_args.speculative_num_draft_tokens,
                    spec_steps=self.model_runner.server_args.speculative_num_steps,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                )

        return spec_info