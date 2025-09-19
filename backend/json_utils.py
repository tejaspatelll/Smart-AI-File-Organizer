import json
import re
from typing import Any


def extract_json_array(text: str) -> Any:
    """Extract and parse a JSON array from a potentially noisy LLM response.

    Prefers strict JSON (whole string is an array). Otherwise, finds the
    outermost '[' ... ']' span and attempts to parse. Strips fenced code
    blocks if present.
    """
    t = (text or "").strip()
    if not t:
        raise ValueError("Empty response")

    # Fast path: exact JSON array
    if t.startswith("[") and t.endswith("]"):
        return json.loads(t)

    # Remove fenced code blocks that sometimes wrap JSON
    t = re.sub(r"```[a-zA-Z]*\n([\s\S]*?)```", r"\1", t)

    start = t.find("[")
    end = t.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in response")

    candidate = t[start : end + 1].strip()
    return json.loads(candidate)


