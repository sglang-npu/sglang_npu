import os
import warnings
from typing import Any, Callable, Optional


def _get_bool_env_var(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is not None:
        value = value.lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        if value in ["false", "0", "no", "n"]:
            return False
    return default


def _get_int_env_var(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is not None:
        try:
            return int(value)
        except ValueError:
            warnings.warn(f"Invalid value for {name}: {value}")
    return default


def _get_str_env_var(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is not None:
        return value
    return default


class EnvField:
    def __init__(
        self, name: str, default: bool, parser: Optional[Callable[[str], Any]] = None
    ):
        self.name = name
        self.default = default
        self.parser = parser
        if self.parser is None:
            if type(self.default) == bool:
                self.parser = _get_bool_env_var
            elif type(self.default) == int:
                self.parser = _get_int_env_var
            elif type(self.default) == str:
                self.parser = _get_str_env_var
            else:
                raise ValueError(f"Unsupported type: {type(self.default)} for {name}")

    def __get__(self, instance, owner):
        return self.parser(self.name, self.default)

    def __set__(self, instance, value):
        # NOTE: we have to make sure the value is string so that it is compatible with the parser
        if value is None:
            os.environ.pop(self.name, None)
        else:
            os.environ[self.name] = str(value)


class EnvVars:
    SGLANG_MOE_PADDING = EnvField("SGLANG_MOE_PADDING", False)

    # ================================================
    # Environment variables for testing
    # ================================================

    SGLANG_TEST_RETRACT = EnvField("SGLANG_TEST_RETRACT", False)


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
