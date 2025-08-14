from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import torch_npu
from torch.nn.functional import scaled_dot_product_attention, softmax

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.distributed import (
    tensor_model_parallel_all_gather,
    get_context_model_parallel_world_size,
    get_context_model_parallel_rank,
    context_model_parallel_all_gather,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
)

import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

def fa_update(all_lse, all_out):
    """
    all_lse: (sp, b*s*hc)
    all_out: (sp, b*s*hc, hDim)
    out: (b*s*hc, hd, 1)
    """

    sp = all_out.shape[0]
    hd = all_out.shape[-1]

    all_lse = all_lse.transpose(0, 1)

    all_out = all_out.permute(1, 2, 0).reshape(-1, sp * hd)

    all_max_lse = torch.max(all_lse, dim=1)[0]
    all_max_lse = all_max_lse.unsqueeze(1)
    all_lse -= all_max_lse

    #(b * s * hc, sp)
    lse_exp = torch.exp(all_lse)
    #(b * s * hc 1)
    sum_lse_exp = torch.sum(lse_exp, dim=-1, keepdim=True)

    #(b * s * hc, sp)
    sum_lse_exp = sum_lse_exp.repeat(1, sp)
    lse_exp = lse_exp / sum_lse_exp

    # oi = lse_exp*oi (b * s * hc, hd, sp) * (b * s * hc, hd, sp)
    lse_exp = lse_exp.unsqueeze(1)
    lse_exp = lse_exp.repeat(1, hd, 1)
    all_out = all_out.reshape(-1, hd, sp)
    all_out = all_out * lse_exp

    # o = sum(oi) (b * s * hc, hd, 1)
    out = torch.sum(all_out, dim=-1, keepdim=True)

    return out

def paged_attention_mla(
    query, #[s, head, 576]
    key_cache, #[-1, 128, 1, 576]
    num_kv_heads, #1
    num_heads, #head
    scale_value,
    block_table, #[[], []]
    context_lens, #[s1, s2]
    mla_vheadsize,
):
    out = []
    lse = []
    last = 0
    for i in range(len(context_lens)):
        kv = []
        seq_len = context_lens[i]
        for page in block_table[i]:
            idx = min(seq_len, 128)
            kv.append(key_cache[page][:idx])
            if seq_len <= 128:
                break
            seq_len -= 128
        kv = torch.cat(kv, dim=0)
        kv = kv.repeat(1, num_heads, 1)
        kv = kv.transpose(0, 1) #[head, S, 576]
        k = kv.transpose(1, 2) #[head, 576, S]
        v = kv[:, :, 0:mla_vheadsize] #[head, S, 512]

        q = query[last:last + context_lens[i]]
        last += context_lens[i]
        q = q.transpose(0, 1) #[head, S, 576]

        qk = torch.bmm(q, k) * scale_value

        l = torch.logsumexp(qk, dim=-1, keepdim=True)
        lse.append(l.transpose(0, 1))

        sm = softmax(qk, dim=-1) #[head S, S]

        o = torch.bmm(sm, v) #[head, S, 512]
        out.append(o.transpose(0, 1))

    out = torch.cat(out, dim=0)
    lse = torch.cat(lse, dim=0)

    return out, lse

def calc_attention(
    query, #[S, head, 576]
    key, #[S, 1, 576]
    value, #[S, 1, 512]
    scale,
):
    q_head = query.shape[1]
    key = key.repeat(1, q_head, 1)
    key = key.permute(1, 2, 0)
    value = value.repeat(1, q_head, 1)
    value = value.transpose(0, 1)
    query = query.transpose(0, 1)
    qk = torch.bmm(query, key) 
    qk = qk * scale
    l = torch.logsumexp(qk, dim=-1, keepdim=True).transpose(0, 1)
    sm = softmax(qk, dim=-1)
    o = torch.bmm(sm, value).transpose(0, 1)
    return o, l

def ring_mla(
    q_nope, #bs head dim
    q_rope, #bs head dim
    k_nope, #bs head dim
    k_rope, #bs head dim
    value,  #bs head dim
    mask,   #not use in mock
    seq_lens,
    head_num, #tp_q_head_num
    kv_head_num, #kv_head_num
    pre_out,
    prev_lse,
    qk_scale,
    mask_type, # mask_type_triu or no_mask
    calc_type, # 'calc_type_first_ring' if prev_out is None else 'calc_type_default',
):
    query = torch.cat([q_nope, q_rope], dim=-1)
    key = torch.cat([k_nope, k_rope], dim=-1)

    out = []
    lse = []
    last = 0
    for i in range(len(seq_lens)):
        seq_len = seq_lens[i]
        q = query[last:last + seq_len]
        k = key[last:last + seq_len]
        v = value[last:last + seq_len]
        last += seq_len

        if mask_type == 'no_mask':
            o, l = calc_attention(q, k, v, qk_scale)
        else:
            out_ = []
            lse_ = []
            for idx in range(q.shape[0]):
                q_ = q[idx:idx + 1]
                k_ = k[:idx + 1]
                v_ = v[:idx + 1]
                o_, l_ = calc_attention(q_, k_, v_, qk_scale)
                out_.append(o_)
                lse_.append(l_)
            o = torch.cat(out_, dim=0)
            l = torch.cat(lse_, dim=0)
        
        lse.append(l) # s, head, 1
        out.append(o) # s, head, dim
    
    out = torch.cat(out, dim=0) # s, head, dim
    lse = torch.cat(lse, dim=0) # s, head, 1

    if calc_type == 'calc_type_first_ring':
        return out, lse
    else:
        max_lse = torch.maximum(lse, prev_lse)
        lse = lse - max_lse
        prev_lse = prev_lse - max_lse
        lse_exp = torch.exp(lse)
        prev_exp = torch.exp(prev_lse)
        sum_exp = lse_exp + prev_exp

        new_lse = torch.log(sum_exp) + max_lse

        lse_exp = lse_exp / sum_exp
        prev_exp = prev_exp / sum_exp

        out = out * lse_exp + pre_out * prev_exp
        
        return out, new_lse


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
        self.forward_metadata = None
        self.device = model_runner.device
        self.attn_mask_builder = AttentionMaskBuilder(8192, model_runner.dtype)
        self.page_size = model_runner.page_size
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        if self.use_mla:
            self.kv_lora_rank = model_runner.model_config.kv_lora_rank
            self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
            self.native_attn = TorchNativeAttnBackend(model_runner)
        self.graph_metadata = {}
        self.max_context_len = model_runner.model_config.context_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.graph_mode = False

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        self.forward_metadata = ForwardMetadata()
        if global_server_args_dict["enable_sp"]:
            seq_lens_max = forward_batch.sp_seq_lens.max()
            self.forward_metadata.seq_lens_cpu_int = forward_batch.sp_seq_lens.cpu().int()
        else:
            seq_lens_max = forward_batch.seq_lens.max()
            self.forward_metadata.seq_lens_cpu_int = forward_batch.seq_lens_cpu.int()

        self.forward_metadata.block_tables = (
            forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : seq_lens_max
            ][:, :: self.page_size]
            // self.page_size
        )
        if forward_batch.extend_seq_lens is not None:
            self.forward_metadata.extend_seq_lens_cpu_int = (
                forward_batch.extend_seq_lens.cpu().int()
            )

        self.graph_mode = False

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.graph_metadata = {
            "block_tables": torch.empty(
                (max_bs, self.max_context_len // self.page_size),
                dtype=torch.int32,
                device=self.device,
            ),
        }

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        metadata = ForwardMetadata()

        metadata.block_tables = self.graph_metadata["block_tables"][:bs, :]
        metadata.seq_lens_cpu_list = seq_lens.cpu().int().tolist()

        self.graph_metadata[bs] = metadata
        self.forward_metadata = metadata

        self.graph_mode = True

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        metadata = self.graph_metadata[bs]
        max_len = seq_lens_cpu[:bs].max().item()
        max_seq_pages = (max_len + self.page_size - 1) // self.page_size

        metadata.block_tables[:bs, :max_seq_pages].copy_(
            self.req_to_token[req_pool_indices[:bs], :max_len][:, :: self.page_size]
            // self.page_size
        )
        metadata.block_tables[:bs, max_seq_pages:].fill_(0)
        metadata.block_tables[bs:, :].fill_(0)

        self.forward_metadata = metadata

        self.graph_mode = True

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

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

            cp_size = get_context_model_parallel_world_size()
            cp_rank = get_context_model_parallel_rank()
            
            if cp_size > 1:
                num_tokens = q.shape[0]
                #q [bs/cp, q_head, qk_dim(nope+rope)]
                #k [bs/cp, 1, qk_dim(nope+rope)]
                #v [bs/cp, 1, v_dim(nope)]

                k_ = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
                v_ = v.view(-1, layer.tp_k_head_num, layer.v_head_dim)

                k_ = context_model_parallel_all_gather(k_, dim=0)
                v_ = context_model_parallel_all_gather(v_, dim=0)

                def ring_split(mtx, cp_size, num_head, head_dim, seq_lens):
                    front_part = []
                    back_part = []

                    mtx = mtx.reshape(cp_size, -1, num_head, head_dim)

                    last = 0
                    for seq_len in seq_lens:
                        assert seq_len % 2 == 0                 
                        front_part.append(mtx[:, last : last + seq_len // 2, :, :])
                        back_part.append(mtx[:, last + seq_len // 2 : last + seq_len, :, :])
                        last += seq_len

                    front_tensor = torch.cat(front_part, dim=1).unsqueeze(1)
                    back_tensor = torch.cat(back_part, dim=1).unsqueeze(1)

                    ret = torch.cat([front_tensor, back_tensor], dim=1)
                    return ret

                q_ = ring_split(q, 1, layer.tp_q_head_num, layer.qk_head_dim, self.forward_metadata.seq_lens_cpu_int.seq_lens).squeeze(0)
                k_ = ring_split(k_, cp_size, layer.tp_k_head_num, layer.qk_head_dim, self.forward_metadata.seq_lens_cpu_int.seq_lens)
                v_ = ring_split(v_, cp_size, layer.tp_k_head_num, layer.v_head_dim, self.forward_metadata.seq_lens_cpu_int.seq_lens)
                
                q_idxs = [cp_rank, cp_size * 2 - cp_rank - 1]

                out = []

                for q_i in range(len(q_idxs)):
                    q_idx = q_idxs[q_i]
                    q_nope, q_rope = torch.split(q_[q_i, :, :, :], self.kv_lora_rank, dim=-1)
                    prev_out = None
                    prev_lse = None

                    go_output = torch.empty(
                        [num_tokens // 2, layer.tp_q_head_num, self.kv_lora_rank],
                        dtype=q.dtype,
                        device=q.device,
                    )
                    lse_output = torch.empty(
                        [num_tokens // 2, layer.tp_q_head_num, 1],
                        dtype=q.dtype,
                        device=q.device
                    )

                    for ring_idx in range(cp_size):
                        kv_idxs = [ring_idx, cp_size * 2 - ring_idx -1]
                        for kv_i in range(len(kv_idxs)):
                            kv_idx = kv_idxs[kv_i]
                            
                            if q_idx < kv_idx:
                                continue
                            
                            k_nope, k_rope = torch.split(k_[ring_idx, kv_i, :, :, :], self.kv_lora_rank, dim=-1)
                            value = v_[ring_idx][kv_i]
                            max_s = max(self.forward_metadata.seq_lens_cpu_int)
                            mask = self.attn_mask_builder.get_attn_mask(max_s, q.dtype, q.device)
                            
                            torch_npu.atb.ring_mla(
                                q_nope,
                                q_rope,
                                k_nope,
                                k_rope,
                                value,
                                mask,
                                self.forward_metadata.seq_lens_cpu_int // 2,
                                head_num=layer.tp_q_head_num,
                                kv_head_num=layer.tp_k_head_num,
                                pre_out=prev_out,
                                prev_lse=prev_lse,
                                qk_scale=layer.scaling,
                                mask_type='mask_type_triu' if q_idx == kv_idx else 'no_mask',
                                calc_type='calc_type_first_ring' if prev_out is None else 'calc_type_default',
                                output=go_output,
                                softmax_lse=lse_output,
                            )

                            prev_out = go_output.clone()
                            prev_lse = lse_output.clone()
                    out.append(go_output)

                def ring_concat(out, seq_lens):
                    ret = []
                    last = 0
                    for seq_len in seq_lens:
                        assert seq_len % 2 == 0
                        for tensor in out:
                            ret.append(tensor[last:last + seq_len // 2, :, :])
                        last += seq_len // 2
                    return torch.cat(ret, dim=0)

                attn_output = ring_concat(out, self.forward_metadata.seq_lens_cpu_int)
                attn_output = attn_output.reshape(num_tokens, layer.tp_q_head_num * layer.v_head_dim)
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
            if self.graph_mode:
                k_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                    layer.layer_id
                ).view(-1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim)
                v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
                    layer.layer_id
                ).view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim)
                query = q.view(-1, 1, layer.tp_q_head_num * layer.qk_head_dim)
                if self.forward_metadata.seq_lens_cpu_int is None:
                    actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_list
                else:
                    actual_seq_len_kv = (
                        self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                    )
                num_tokens = query.shape[0]
                workspace = (
                    torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                        query,
                        k_cache,
                        v_cache,
                        block_table=self.forward_metadata.block_tables,
                        block_size=self.page_size,
                        num_heads=layer.tp_q_head_num,
                        num_key_value_heads=layer.tp_k_head_num,
                        input_layout="BSH",
                        scale=layer.scaling,
                        actual_seq_lengths_kv=actual_seq_len_kv,
                    )
                )
                output = torch.empty(
                    (num_tokens, 1, layer.tp_q_head_num * layer.v_head_dim),
                    dtype=q.dtype,
                    device=q.device,
                )
                softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)
                torch_npu.npu_fused_infer_attention_score.out(
                    query,
                    k_cache,
                    v_cache,
                    block_table=self.forward_metadata.block_tables,
                    block_size=self.page_size,
                    num_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="BSH",
                    scale=layer.scaling,
                    actual_seq_lengths_kv=actual_seq_len_kv,
                    workspace=workspace,
                    out=[output, softmax_lse],
                )
            else:
                k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
                    layer.layer_id
                )
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
            if self.graph_mode:
                kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                    layer.layer_id
                ).view(-1, self.page_size, layer.head_dim)
                k_rope = kv_cache[:, :, layer.v_head_dim :]
                c_kv = kv_cache[:, :, : layer.v_head_dim]
                k_rope_cache = k_rope.view(
                    -1,
                    layer.tp_k_head_num,
                    self.page_size,
                    layer.head_dim - layer.v_head_dim,
                )
                c_kv_cache = c_kv.view(
                    -1, layer.tp_v_head_num, self.page_size, layer.v_head_dim
                )
                q = q.contiguous().view(-1, layer.tp_q_head_num, 1, layer.head_dim)
                q_nope = q[:, :, :, : layer.v_head_dim]
                q_rope = q[:, :, :, layer.v_head_dim :]

                if self.forward_metadata.seq_lens_cpu_int is None:
                    actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_list
                else:
                    actual_seq_len_kv = (
                        self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                    )

                output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q_nope,
                    c_kv_cache,
                    c_kv_cache,
                    query_rope=q_rope,
                    key_rope=k_rope_cache,
                    num_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="BNSD",
                    scale=layer.scaling,
                    antiquant_mode=0,
                    antiquant_scale=None,
                    block_table=self.forward_metadata.block_tables,
                    block_size=self.page_size,
                    actual_seq_lengths_kv=actual_seq_len_kv,
                )
                return output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
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
