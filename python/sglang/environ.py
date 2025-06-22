import os
import warnings
from typing import Any


class EnvField:
    def __init__(self, name: str, default: Any):
        self.name = name
        self.default = default

    def parser(self, name: str, default: Any) -> Any:
        value = os.getenv(name)
        if value is not None:
            return value
        return default

    def __get__(self, instance, owner):
        return self.parser(self.name, self.default)

    def __set__(self, instance, value):
        # NOTE: we have to make sure the value is string so that it is compatible with the parser
        if value is None:
            os.environ.pop(self.name, None)
        else:
            os.environ[self.name] = str(value)


class EnvFieldBool(EnvField):
    def __init__(self, name: str, default: bool):
        super().__init__(name, default)

    def parser(self, name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is not None:
            value = value.lower()
            if value in ["true", "1", "yes", "y"]:
                return True
            if value in ["false", "0", "no", "n"]:
                return False
        return default


class EnvFieldInt(EnvField):
    def __init__(self, name: str, default: int):
        super().__init__(name, default)

    def parser(self, name: str, default: int) -> int:
        value = os.getenv(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                warnings.warn(f"Invalid value for {name}: {value}")
        return default


class EnvVars:
    # fmt: off
    SGLANG_ENABLE_TORCH_INFERENCE_MODE = EnvFieldBool("SGLANG_ENABLE_TORCH_INFERENCE_MODE", False)
    SGLANG_SET_CPU_AFFINITY = EnvFieldBool("SGLANG_SET_CPU_AFFINITY", False)
    SGLANG_MOE_PADDING = EnvFieldBool("SGLANG_MOE_PADDING", False)
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN = EnvFieldBool("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", True)

    # ================================================
    # Environment variables for testing
    # ================================================

    SGLANG_TEST_RETRACT = EnvFieldBool("SGLANG_TEST_RETRACT", False)

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
