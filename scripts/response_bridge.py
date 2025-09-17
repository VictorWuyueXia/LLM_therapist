# scripts/response_bridge.py
import re
from scripts.response_analyzer import classify_dimension_and_score

def get_openai_resp(user_input):
    lower = [t.lower() for t in user_input.replace(".", " ").replace(",", " ").replace("?", " ").split()[:5]]
    if "stop" in lower: return "DLA", "Stop"
    if "yes" in lower:  return "DLA", "Yes"
    if "no"  in lower:  return "DLA", "No"
    try:
        raw = classify_dimension_and_score(user_input)
        first = str(raw).strip().splitlines()[0].strip()
    except Exception:
        return "NA", 99
    m = re.match(r"^\s*(Yes|No|Stop|Question|Maybe)\s*,?\s*(\d+)?\s*$", first, flags=re.IGNORECASE)
    if m:
        token = m.group(1).strip().lower()
        if token == "maybe": return "DLA", "Question"
        if token in ("yes","no","stop","question"):
            return "DLA", token.capitalize()
    m = re.match(r"^\s*([A-Za-z0-9_]+)\s*,\s*([0-2])\s*$", first)
    if m:
        dim, score = m.group(1).strip(), int(m.group(2))
        if re.match(r"^\d+_[A-Za-z_]+$", dim): dim = "DLA_" + dim
        elif not dim.startswith("DLA_"):       return "NA", 99
        return dim, score
    m = re.match(r"^\s*(Other)\s*,\s*(\d+)\s*$", first, flags=re.IGNORECASE)
    if m: return "NA", 99
    return "NA", 99