import os
import warnings


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


def convert_SGL_to_SGLANG():
    for key, value in os.environ.items():
        if key.startswith("SGL_"):
            new_key = key.replace("SGL_", "SGLANG_")
            warnings.warn(
                f"Environment variable {key} is deprecated, please use {new_key}"
            )
            os.environ[new_key] = value


class EnvVars:
    @property
    def SGLANG_MOE_PADDING(self) -> bool:
        return _get_bool_env_var("SGLANG_MOE_PADDING", False)


envs = EnvVars()

convert_SGL_to_SGLANG()
