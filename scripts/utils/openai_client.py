import os
from typing import Optional

import openai


def configure_openai_from_env(env_var: str = "OPENAI_API_KEY") -> None:
    key: Optional[str] = os.environ.get(env_var)
    if key is None or not key.strip():
        # Let downstream raise if used without key, per user rule (no silent suppression)
        return
    openai.api_key = key


