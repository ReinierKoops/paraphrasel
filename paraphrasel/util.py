import json
import os
from typing import Dict, Optional

# Adjust the path if your JSON is elsewhere
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config/language_configs.json")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    LANGUAGE_CONFIGS: Dict[str, Dict] = json.load(f)


def get_language_config(
    language_code: str, override_config: Optional[dict] = None
) -> Dict:
    """
    Returns a dict containing 'model_name' and 'params' for the given language_code.
    Raises ValueError if language_code is not found.
    """
    language_config = LANGUAGE_CONFIGS if override_config is None else override_config
    if language_code not in language_config:
        raise ValueError(f"No config found for language code '{language_code}'.")
    return language_config[language_code]


def create_language_config(
    language: str, model_name: str, params: Optional[Dict]
) -> Dict:
    return {
        language: {
            "model_name": model_name,
            "params": params if params else {},
        }
    }
