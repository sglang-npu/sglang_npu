import torch
import torch.distributed as dist
import torch_npu

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import DeepEPMode
from sglang.srt.utils import get_int_env_var


class AscendDeepEPDispatcher:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.auto,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        self.rank = torch.distributed.get_rank(group=group)
        backend = group._get_backend(torch.device("npu"))
        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(self.rank)
        self.num_experts = num_experts
        self.ep_size = get_tensor_model_parallel_world_size()
        self.num_max_dispatch_tokens_per_rank = get_int_env_var(
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 48
        )
        # global bs 支持取0或BS*ep_world_size
        self.global_bs = self.num_max_dispatch_tokens_per_rank * self.ep_size 

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        kwargs_mc2 = {
            "x": hidden_states,
            "expert_ids": topk_idx.to(torch.int32),
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.num_experts,
            "global_bs": self.global_bs,
        }

        stage1_kwargs = {
            "scales": None,
            "quant_mode": 2,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": self.ep_size,
            "ep_rank_id": self.rank,
            "group_tp": self.moe_all_to_all_group_name,
            "tp_world_size": 1,
            "tp_rank_id": 0,
        }
        kwargs_mc2.update(stage1_kwargs)
        output = torch_npu.npu_moe_distribute_dispatch(**kwargs_mc2)
        (
            expand_x,
            dynamic_scale,
            expand_idx,
            expert_token_nums,
            ep_recv_counts,
            tp_recv_counts,
        ) = output[0:6]
        return (
            (expand_x, dynamic_scale),
            expand_idx,
            topk_weights,
            None,
            ep_recv_counts,
            expert_token_nums,
            None,
            None,
            tp_recv_counts,
        )

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        expand_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_batch: ForwardBatch,
        ep_send_counts,
        tp_recv_counts,
    ):
        kwargs_mc2 = {
            "expand_x": hidden_states,
            "expert_ids": topk_idx.to(torch.int32),
            "expand_idx": expand_idx,
            "expert_scales": topk_weights.to(torch.float32),
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.num_experts,
            "global_bs": self.global_bs,
        }

        stage3_kwargs = {
            "ep_send_counts": ep_send_counts,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": self.ep_size,
            "ep_rank_id": torch.distributed.get_rank(),
            "tp_send_counts": tp_recv_counts,
            "group_tp": self.moe_all_to_all_group_name,
            "tp_world_size": 1,
            "tp_rank_id": 0,
        }
        kwargs_mc2.update(stage3_kwargs)

        final_hidden_states = torch_npu.npu_moe_distribute_combine(**kwargs_mc2)
        return final_hidden_states
