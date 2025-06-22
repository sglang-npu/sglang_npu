import os
import warnings
from typing import Any


class EnvField:
    def __init__(self, default: Any):
        self.default = default

    def __set_name__(self, owner, name):
        self.name = name

    def parse(self, value: str) -> Any:
        return value

    def get(self, default: Any = None) -> Any:
        value = os.getenv(self.name)
        if value is None:
            return default or self.default
        try:
            return self.parse(value)
        except ValueError as e:
            warnings.warn(
                f'Invalid value for {self.name}: {e}, using default "{self.default}"'
            )
            return self.default

    def set(self, value: Any):
        # NOTE: we have to make sure the value is string so that it is compatible with the parser
        os.environ[self.name] = str(value)

    def clear(self):
        os.environ.pop(self.name, None)

    def __get__(self, instance, owner):
        if isinstance(instance, Envs):
            return self.get()

        return self

    def __set__(self, instance, value):
        if value is None:
            self.clear()
        else:
            self.set(value)


class EnvFieldBool(EnvField):
    def parse(self, value: str) -> bool:
        value = value.lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        if value in ["false", "0", "no", "n"]:
            return False
        raise ValueError(f'"{value}" is not a valid boolean value')


class EnvFieldInt(EnvField):
    def parse(self, value: str) -> int:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid integer value')


class EnvFieldFloat(EnvField):
    def parse(self, value: str) -> float:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid float value')


class Envs:
    # fmt: off
    SGLANG_ENABLE_TORCH_INFERENCE_MODE = EnvFieldBool(False)
    SGLANG_SET_CPU_AFFINITY = EnvFieldBool(False)
    SGLANG_MOE_PADDING = EnvFieldBool(False)
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN = EnvFieldBool(True)
    SGLANG_USE_MODELSCOPE = EnvFieldBool(False)
    SGLANG_DISABLE_REQUEST_LOGGING = EnvFieldBool(False)
    SGLANG_INT4_WEIGHT = EnvFieldBool(False)
    USE_VLLM_CUTLASS_W8A8_FP8_KERNEL = EnvFieldBool(False)
    SGLANG_SUPPORT_CUTLASS_BLOCK_FP8 = EnvFieldBool(False)
    SGLANG_ENABLE_TORCH_COMPILE = EnvFieldBool(False)
    SGLANG_FORCE_FP8_MARLIN = EnvFieldBool(False)
    SGLANG_CUTLASS_MOE = EnvFieldBool(False)
    SYNC_TOKEN_IDS_ACROSS_TP = EnvFieldBool(False)
    SGLANG_GRAMMAR_TIMEOUT = EnvFieldFloat(300)
    SGLANG_PROFILE_WITH_STACK = EnvFieldBool(True)
    SGL_FORCE_SHUTDOWN = EnvFieldBool(False)
    SGLANG_DEBUG_MEMORY_POOL = EnvFieldBool(False)
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL = EnvFieldBool(False)
    SGLANG_TEST_REQUEST_TIME_STATS = EnvFieldBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_CANARY = EnvFieldBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS = EnvFieldBool(False)
    SGLANG_LOG_EXPERT_LOCATION_METADATA = EnvFieldBool(False)
    SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK = EnvFieldBool(False)

    # ================================================
    # Logging
    # ================================================
    SGLANG_GC_LOG = EnvFieldBool(False)


    # ================================================
    # Attention / Kernel Backends
    # ================================================
    SGLANG_IS_FLASHINFER_AVAILABLE = EnvFieldBool(True)
    SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS = EnvFieldBool(False)
    SGLANG_USE_AITER = EnvFieldBool(False)
    SGL_IS_FIRST_RANK_ON_NODE = EnvFieldBool(True)
    SGLANG_ENABLE_FLASHINFER_GEMM = EnvFieldBool(False)

    # ================================================
    # DeepGemm
    # ================================================

    SGL_JIT_DEEPGEMM_PRECOMPILE = EnvFieldBool(True)
    SGL_JIT_DEEPGEMM_COMPILE_WORKERS = EnvFieldInt(4)
    SGL_IN_DEEPGEMM_PRECOMPILE_STAGE = EnvFieldBool(False)
    SGL_ENABLE_JIT_DEEPGEMM = EnvFieldBool(True)


    # ================================================
    # Runtime Configuration
    # ================================================
    SGLANG_INIT_NEW_TOKEN_RATIO = EnvFieldFloat(None)
    SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR = EnvFieldFloat(None)
    SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS = EnvFieldInt(None)
    SGLANG_DISABLE_OUTLINES_DISK_CACHE = EnvFieldBool(True)
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER = EnvFieldBool(True)
    SGLANG_DISABLE_OPENAPI_DOC = EnvFieldBool(False)

    # ================================================
    # Testing Environment Variables
    # ================================================
    SGLANG_TEST_RETRACT = EnvFieldBool(False)
    SGLANG_RECORD_STEP_TIME = EnvFieldBool(False)
    SGLANG_IS_IN_CI = EnvFieldBool(False)

    # fmt: on


envs = Envs()


def convert_SGL_to_SGLANG():
    for key, value in os.environ.items():
        if key.startswith("SGL_"):
            new_key = key.replace("SGL_", "SGLANG_")
            warnings.warn(
                f"Environment variable {key} is deprecated, please use {new_key}"
            )
            os.environ[new_key] = value


convert_SGL_to_SGLANG()

if __name__ == "__main__":
    # Example usage for envs
    envs.SGLANG_TEST_RETRACT = None
    print(f"{envs.SGLANG_TEST_RETRACT=}")
    envs.SGLANG_TEST_RETRACT ^= True
    print(f"{envs.SGLANG_TEST_RETRACT=}")

    # Example usage for EnvVars
    Envs.SGLANG_TEST_RETRACT.clear()
    print(f"{Envs.SGLANG_TEST_RETRACT.get()=}")
    print(f"{Envs.SGLANG_TEST_RETRACT.get(True)=}")
    Envs.SGLANG_TEST_RETRACT.set(True)
    print(f"{Envs.SGLANG_TEST_RETRACT.get()=}")
