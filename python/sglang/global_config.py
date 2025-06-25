"""Global configurations"""

import os

from sglang.environ import Envs


class GlobalConfig:
    """
    Store some global constants.

    See also python/sglang/srt/managers/schedule_batch.py::global_server_args_dict, which stores
    many global runtime arguments as well.
    """

    def __init__(self):
        # Verbosity level
        # 0: do not output anything
        # 2: output final text after every run
        self.verbosity = 0

        # Default backend of the language
        self.default_backend = None

        # Runtime constants: New generation token ratio estimation
        self.default_init_new_token_ratio = Envs.SGLANG_INIT_NEW_TOKEN_RATIO.get(0.7)
        self.default_min_new_token_ratio_factor = (
            Envs.SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR.get(0.14)
        )
        self.default_new_token_ratio_decay_steps = (
            Envs.SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS.get(600)
        )

        # Runtime constants: others
        self.retract_decode_steps = 20
        self.flashinfer_workspace_size = os.environ.get(
            "FLASHINFER_WORKSPACE_SIZE", 384 * 1024 * 1024
        )

        # Output tokenization configs
        self.skip_special_tokens_in_output = True
        self.spaces_between_special_tokens_in_out = True

        # Language frontend interpreter optimization configs
        self.enable_precache_with_tracing = True
        self.enable_parallel_encoding = True


global_config = GlobalConfig()
