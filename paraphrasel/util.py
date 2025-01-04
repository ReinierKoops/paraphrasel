import json
import os
from typing import Dict

# Adjust the path if your JSON is elsewhere
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config/language_configs.json")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    LANGUAGE_CONFIGS: Dict[str, Dict] = json.load(f)


def get_language_config(language_code: str) -> Dict:
    """
    Returns a dict containing 'model_name' and 'tokenizer_params' for the given language_code.
    Raises ValueError if language_code is not found.
    """
    if language_code not in LANGUAGE_CONFIGS:
        raise ValueError(f"No config found for language code '{language_code}'.")
    return LANGUAGE_CONFIGS[language_code]
