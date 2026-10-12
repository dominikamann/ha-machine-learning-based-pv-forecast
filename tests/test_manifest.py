import json
from pathlib import Path

from custom_components.ml_pv_forecast.const import DOMAIN

ROOT = Path(__file__).parents[1]


def test_manifest_matches_domain():
    manifest = json.loads((ROOT / "custom_components/ml_pv_forecast/manifest.json").read_text())
    assert manifest["domain"] == DOMAIN
    assert manifest["config_flow"] is True
    assert any(r.startswith("scikit-learn") for r in manifest["requirements"])
