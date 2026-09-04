"""Smoke test — proves the package imports and config loads.

Run: pytest -q     (a green test here means the skeleton is wired correctly.)
"""

from s2n.config import load_config


def test_config_loads():
    cfg = load_config()
    assert "reliability" in cfg
    assert 0 <= cfg["reliability"]["reliable_max"] < cfg["reliability"]["unreliable_min"]
