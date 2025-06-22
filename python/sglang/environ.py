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


class EnvVars:
    @property
    def SGLANG_MOE_PADDING(self) -> bool:
        return _get_bool_env_var("SGLANG_MOE_PADDING", False)


envs = EnvVars()
