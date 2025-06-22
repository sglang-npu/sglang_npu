import os
import warnings
from typing import Any


class EnvField:
    def __init__(self, default: Any):
        self.default = default

    def parse(self, value: str) -> Any:
        return value

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        value = os.getenv(self.name)
        if value is None:
            return self.default
        try:
            return self.parse(value)
        except ValueError as e:
            warnings.warn(
                f'Invalid value for {self.name}: {e}, using default "{self.default}"'
            )
            return self.default

    def __set__(self, instance, value):
        # NOTE: we have to make sure the value is string so that it is compatible with the parser
        if value is None:
            os.environ.pop(self.name, None)
        else:
            os.environ[self.name] = str(value)


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


class EnvVars:
    # fmt: off
    SGLANG_ENABLE_TORCH_INFERENCE_MODE = EnvFieldBool(False)
    SGLANG_SET_CPU_AFFINITY = EnvFieldBool(False)
    SGLANG_MOE_PADDING = EnvFieldBool(False)
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN = EnvFieldBool(True)
    SGLANG_USE_MODELSCOPE = EnvFieldBool(False)
    SGLANG_DISABLE_REQUEST_LOGGING = EnvFieldBool(False)
    SGLANG_INT4_WEIGHT: EnvFieldBool = EnvFieldBool(False)

    # ================================================
    # Attention / Kernel Backends
    # ================================================
    SGLANG_IS_FLASHINFER_AVAILABLE = EnvFieldBool(True)
    SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS = EnvFieldBool(False)
    SGLANG_USE_AITER = EnvFieldBool(False)

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
    SGLANG_IS_IN_CI = EnvFieldBool(False)

    # fmt: on


envs = EnvVars()


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
    # Example usage
    print(envs.SGLANG_TEST_RETRACT)
    envs.SGLANG_TEST_RETRACT ^= 1
    print(envs.SGLANG_TEST_RETRACT)
    # unset the value, using the default
    envs.SGLANG_TEST_RETRACT = None
    print(envs.SGLANG_TEST_RETRACT)
