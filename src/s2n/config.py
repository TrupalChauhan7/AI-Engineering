"""Load project settings from config/config.yaml.

WHY: every module reads settings from ONE place. No hardcoded paths or
model names anywhere else. Change an experiment by editing config.yaml.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # -> project/
CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Return the whole config as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
