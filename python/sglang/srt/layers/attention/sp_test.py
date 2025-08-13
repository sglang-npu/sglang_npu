import torch
import torch_npu

num_tokens = 20
tp_q_head_num = 8
tp_k_head_num = 1
kv_lora_rank = 512
head_dim = 576
attn_tp_size = 2
attn_sp_size = attn_tp_size
scaling = 0.041
num_blocks = 32
block_size = 128

query_0 = torch.randn(
    (num_tokens, tp_q_head_num, 576),
    dtype=torch.float16,
).npu()

query_1 = torch.randn(
    (num_tokens, tp_q_head_num, 576),
    dtype=torch.float16,
).npu()

kv_cache = torch.randn (
    (num_blocks, block_size, tp_k_head_num, head_dim),
    dtype=torch.float16,
).npu()

block_tables = torch.tensor([[0, 1]], dtype=torch.int32).npu()
seq_lens_cpu_int = torch.tensor([256], dtype=torch.int32)



def sp_calc(query, kv, block_tables, seq_lens_cpu_int):
    go_output = torch.empty(
        [num_tokens, tp_q_head_num * attn_tp_size, kv_lora_rank],
        dtype=query.dtype,
        device=query.device,
    )
    lse_output = torch.empty(
        [num_tokens, tp_q_head_num * attn_tp_size, 1],
        dtype=query.dtype,
        device=query.device,
    )
    torch_npu.atb.npu_multi_head_latent_attention_with_lse(
        query[:, :, :kv_lora_rank],
        query[:, :, kv_lora_rank:],
        kv[:, :, :, :kv_lora_rank],
        kv[:, :, :, kv_lora_rank:],
        block_tables,
        seq_lens_cpu_int,
        tp_q_head_num * attn_tp_size,
        scaling,
        tp_k_head_num,
        calc_type="calc_type_ring",
        output=go_output,
        lse=lse_output
    )

    return go_output, lse_output

def sp_update(go_lse_output, attn_tp_rank):
    # [S*sp, head, head_dim] [S*sp, head, 1]
    go_output, lse_output = go_lse_output.split([kv_lora_rank, 1], dim=-1)
    # [S*sp, head/tp, head_dim] -> [sp, S, head/tp, head_dim]
    go_output = go_output.reshape(attn_sp_size, -1, tp_q_head_num * attn_tp_size, kv_lora_rank)
    go_output = torch.chunk(go_output, chunks=attn_tp_size, dim=2)[attn_tp_rank]
    go_output = go_output.reshape(attn_sp_size, -1, kv_lora_rank)
    #[S*sp, head/tp, 1] -> [sp, S, head/tp, 1]
    lse_output = lse_output.reshape(attn_sp_size, -1, tp_q_head_num * attn_tp_size, 1)
    lse_output = torch.chunk(lse_output, chunks=attn_tp_size, dim=2)[attn_tp_rank]
    lse_output = lse_output.reshape(attn_sp_size, -1)

    original_dtype=go_output.dtype
    if original_dtype != torch.float32:
        go_output = go_output.to(torch.float32)
        lse_output = lse_output.to(torch.float32)

    attn_output = torch.zeros(
        (num_tokens * tp_q_head_num, kv_lora_rank, 1),
        dtype=torch.float32,
        device=query_0.device
    )

    torch_npu.atb._npu_fa_update(lse_output, go_output, 0, attn_sp_size, attn_output)

    if original_dtype != torch.float32:
        attn_output = attn_output.to(original_dtype)

    return attn_output.reshape(num_tokens, tp_q_head_num * kv_lora_rank)


def sp(query_0, query_1, kv):
    query = torch.cat([query_0, query_1], dim=1)
 
    go0, lse0 = sp_calc(query, kv, torch.tensor([[0]], dtype=torch.int32).npu(), [128])
    go1, lse1 = sp_calc(query, kv, torch.tensor([[1]], dtype=torch.int32).npu(), [128])
    
    # [S, head, head_dim + 1]
    go_lse_output0 = torch.cat([go0, lse0], dim=-1)
    go_lse_output1 = torch.cat([go1, lse1], dim=-1)

   # all gather
    go_lse_output = torch.cat([go_lse_output0, go_lse_output1], dim = 0)

    return sp_update(go_lse_output, 0), sp_update(go_lse_output, 1)


def golden(query, kv):
    attn_output = torch.empty(
        [num_tokens, tp_q_head_num, kv_lora_rank],
        dtype=query.dtype,
        device=query.device,
    )
    torch_npu._npu_paged_attention_mla(
        query=query,
        key_cache=kv,
        num_kv_heads=tp_k_head_num,
        num_heads=tp_q_head_num,
        scale_value=scaling,
        block_table=block_tables,
        context_lens=seq_lens_cpu_int,
        mla_vheadsize=kv_lora_rank,
        out=attn_output,
    )

    return attn_output.view(num_tokens, tp_q_head_num * kv_lora_rank)


def main():
    #tp0
    tp0_golden = golden(query_0, kv_cache)
    #tp1
    tp1_golden = golden(query_1, kv_cache)

    tp0_sp, tp1_sp = sp(query_0, query_1, kv_cache)

    print(torch.allclose(tp0_golden, tp0_sp, rtol=1e-6, atol=1e-6))
    print(torch.allclose(tp1_golden, tp1_sp, rtol=1e-6, atol=1e-6))


    print("tp0_golden:",tp0_golden.shape, tp0_golden.sum())
    print("tp1_golden:",tp1_golden.shape, tp1_golden.sum())
    print("tp0_sp:", tp0_sp.shape, tp0_sp.sum())
    print("tp1_sp:", tp1_sp.shape, tp1_sp.sum())


def main2():
    std = golden(query_0, kv_cache)

    go0, lse0 = sp_calc(query_0, kv, torch.tensor([[0]], dtype=torch.int32).npu(), [128])
    go1, lse1 = sp_calc(query_0, kv, torch.tensor([[1]], dtype=torch.int32).npu(), [128])

    go_lse_output0 = torch.cat([go0, lse0], dim=-1)
    go_lse_output1 = torch.cat([go1, lse1], dim=-1)

    go_lse_output = torch.cat([go_lse_output0, go_lse_output1], dim = 0)

     original_dtype=go_output.dtype
    if original_dtype != torch.float32:
        go_output = go_output.to(torch.float32)
        lse_output = lse_output.to(torch.float32)

    attn_output = torch.zeros(
        (num_tokens * tp_q_head_num, kv_lora_rank, 1),
        dtype=torch.float32,
        device=query_0.device
    )

    torch_npu.atb._npu_fa_update(lse_output, go_output, 0, attn_sp_size, attn_output)

    if original_dtype != torch.float32:
        attn_output = attn_output.to(original_dtype)
    
    print(torch.allclose(std, attn_output, rtol=1e-6, atol=1e-6))

    print("std:". std.shape, std.sum())
    print("attn:", attn_output.shape, attn_output.sum())

if __name__=='__main__':
    #main()
    main2()
