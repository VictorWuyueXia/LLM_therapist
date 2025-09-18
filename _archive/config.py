# scripts/config.py
import os
from typing import Any, Dict

# Lightweight YAML config loader; default to env/fallbacks if YAML missing.
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

_ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
_CONFIG_PATH = os.path.join(_ROOT_DIR, "config.yaml")

def _load_yaml_config() -> Dict[str, Any]:
    if yaml is None:
        return {}
    if not os.path.exists(_CONFIG_PATH):
        return {}
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return {}
        return data

_CFG = _load_yaml_config()

# Subject and data layout
SUBJECT_ID = str(_CFG.get("subject_id", os.environ.get("SUBJECT_ID", "8901")))

DATA_DIR = _CFG.get(
    "data_dir",
    os.path.join(_ROOT_DIR, "Subject_8899_Mobicom_SenSys"),
)
RESULT_DIR = _CFG.get(
    "result_dir",
    os.path.join(DATA_DIR, "data_collection_results"),
)

QUESTION_LIB_FILENAME = _CFG.get(
    "question_lib_filename",
    os.path.join(DATA_DIR, f"question_lib_v3_{SUBJECT_ID}.json"),
)
REPORT_FILE = _CFG.get(
    "report_file",
    os.path.join(RESULT_DIR, f"Report_{SUBJECT_ID}.csv"),
)
NOTES_FILE = _CFG.get(
    "notes_file",
    os.path.join(RESULT_DIR, f"Notes_{SUBJECT_ID}.csv"),
)
RECORD_CSV = _CFG.get(
    "record_csv",
    os.path.join(DATA_DIR, "record.csv"),
)

# RL hyperparameters
ITEM_N_STATES = int(_CFG.get("item_n_states", 20))
EPSILON = float(_CFG.get("epsilon", 0.9))
ALPHA = float(_CFG.get("alpha", 0.1))
GAMMA = float(_CFG.get("gamma", 0.9))

ITEM_IMPORTANCE = _CFG.get(
    "item_importance",
    [0, 5, 4, 4, 2, 5, 2, 2, 1, 3, 4, 3, 1, 4, 2, 4, 3, 1, 4, 4],
)
NUMBER_QUESTIONS = _CFG.get(
    "number_questions",
    [0, 4, 1, 2, 2, 5, 1, 1, 1, 2, 3, 2, 1, 1, 3, 2, 1, 3, 1, 1],
)

# OpenAI defaults
OPENAI_MODEL = _CFG.get("openai_model", "gpt-4o")

# Max tokens per module/functionality (reflect current code defaults)
_MAX_TOKENS = _CFG.get("max_tokens", {}) or {}

MAX_TOKENS_RESPONSE_ANALYZER = int(_MAX_TOKENS.get("response_analyzer", 400))
MAX_TOKENS_REFLECTION_VALIDATION = int(_MAX_TOKENS.get("reflection_validation", 400))
MAX_TOKENS_CBT = int(_MAX_TOKENS.get("cbt", 300))

_MT_TXT = _MAX_TOKENS.get("text_generators", {}) or {}
MAX_TOKENS_TEXTGEN_SYNONYMOUS = int(_MT_TXT.get("synonymous", 200))
MAX_TOKENS_TEXTGEN_THERAPIST = int(_MT_TXT.get("therapist_chat", 200))
MAX_TOKENS_TEXTGEN_CHANGE = int(_MT_TXT.get("change", 120))
MAX_TOKENS_TEXTGEN_CHANGE_POS = int(_MT_TXT.get("change_positive", 120))
MAX_TOKENS_TEXTGEN_CHANGE_NEG = int(_MT_TXT.get("change_negative", 120))