import re
import json
from src.response_analyzer import classify_dimension_and_score
from src.utils.log_util import get_logger
logger = get_logger("ResponseBridge")

def _normalize_dim_score(dim: str, score: int):
    """
    Normalize dimension label to start with DLA_ when appropriate and
    validate score range.
    """
    # If dimension looks like digits underscored then text (e.g. 3_talk), prepend DLA_
    if re.match(r"^\d+_[A-Za-z_]+$", dim):
        dim = "DLA_" + dim
    elif not dim.startswith("DLA_"):
        # If it doesn't start with DLA_, it's invalid
        return None

    # Ensure score is an integer between 0 and 2 (inclusive)
    if not isinstance(score, int) or score < 0 or score > 2:
        return None

    return dim, score

def _parse_dim_score_from_text(text: str):
    """
    Parse '[dim][sep][score]' from a free-form text line.
    Accept separators: comma, colon, hyphen, or whitespace.
    """
    # First, match at the start of line, fairly lenient about separators and whitespace
    m = re.match(r"^\s*([A-Za-z0-9_]+)\s*[,:\-\s]\s*([0-2])\s*$", text)
    if not m:
        # If that fails, try searching anywhere as a fallback for DLA_ or dimension pattern
        m = re.search(r"((?:DLA_)?\d+_[A-Za-z_]+)\s*[,:\-\s]\s*([0-2])", text)
    if m:
        # If match is found, extract dimension and score
        dim = m.group(1).strip()
        score = int(m.group(2))
        # Normalize and check dimension and score
        norm = _normalize_dim_score(dim, score)
        if norm:
            logger.debug(f"Parsed dimension-score (text): {norm[0]}, {norm[1]}")
            return norm
    # If nothing matched, return None to indicate failure
    return None

def _parse_from_json_like(raw: str):
    """
    If the model returns JSON-like content, try to extract:
    - {'res': '3_talk, 1'}
    - {'dimension': '3_talk', 'score': 1}
    """
    s = str(raw).strip()
    # Only try if string looks like JSON object
    if s.startswith("{") and s.endswith("}"):
        # Parse JSON to dictionary
        data = json.loads(s)
        logger.debug(f"Parsed JSON classification: {data}")
        # Lowercase all keys for robust lookup
        kl = {str(k).lower(): v for k, v in data.items()}
        # Case 1: 'res' key contains a string like '3_talk, 1'
        if "res" in kl:
            val = str(kl["res"]).strip()
            got = _parse_dim_score_from_text(val)
            if got:
                return got
        # Case 2: JSON object has separate 'dimension' and 'score' keys
        if "dimension" in kl and "score" in kl:
            dim = str(kl["dimension"]).strip()
            sc = int(kl["score"])
            norm = _normalize_dim_score(dim, sc)
            if norm:
                logger.debug(f"Parsed dimension-score (json): {norm[0]}, {norm[1]}")
                return norm
        # Fallback: try to extract from string form of the object in case the above failed
        got = _parse_dim_score_from_text(s)
        if got:
            return got
    # If not JSON object or any extraction method failed
    return None

def get_openai_resp(user_input):
    """
    Main entry point to process model response or user input and extract (dimension, score).
    Fallbacks to ('NA', 99) on parse failure.
    """
    # Preprocess: get first 10 lowercased tokens after removing some punctuation for basic pattern catches
    lower = [t.lower() for t in user_input.replace(".", " ").replace(",", " ").replace("?", " ").split()[:10]]

    # Detect easy and common cases up front, for quick handling:
    if "stop" in lower: 
        return "DLA", "Stop"
    if "yes" in lower:  
        return "DLA", "Yes"
    if "no"  in lower:  
        return "DLA", "No"
    if "maybe"  in lower:  
        return "DLA", "Maybe"
    if "question"  in lower:  
        return "DLA", "Question"

    try:
        # Use the response analyzer to try to classify the input
        raw = classify_dimension_and_score(user_input)
        # Take just the first line (in case of multi-line output)
        first = str(raw).strip().splitlines()[0].strip()
        logger.debug(f"OpenAI raw: {raw}")
        logger.debug(f"First line parsed: {first}")
    except Exception as e:
        # Log failure for diagnostics, fallback code
        logger.debug(f"classify_dimension_and_score exception: {e}")
        return "NA", 99

    # Try to match general words like Yes/No/Stop/Question/Maybe, possibly with a number after a comma
    m = re.match(r"^\s*(Yes|No|Stop|Question|Maybe)\s*,?\s*(\d+)?\s*$", first, flags=re.IGNORECASE)
    if m:
        # token is one of the general words
        token = m.group(1).strip().lower()
        # Return token capitalized if it's one of the general words
        if token in ("yes", "no", "maybe", "question", "stop"):
            return "DLA", token.capitalize()

    # Try to parse result in case it's JSON-ish (either first line or whole raw)
    got = _parse_from_json_like(first)
    if not got:
        got = _parse_from_json_like(str(raw))
    if got:
        return got

    # Maybe it's a plain-text dimension,score (e.g. '3_talk, 1' or 'DLA_3_talk,1')
    got = _parse_dim_score_from_text(first)
    if got:
        return got

    # If response is 'Other, N', always fallback to NA,99
    m = re.match(r"^\s*(Other)\s*,\s*(\d+)\s*$", first, flags=re.IGNORECASE)
    if m:
        return "NA", 99

    # If all else fails, fallback
    logger.debug("Failed to parse classification, fallback to NA,99")
    return "NA", 99