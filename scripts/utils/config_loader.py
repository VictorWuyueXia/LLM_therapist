import os
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            return data
        return {}


