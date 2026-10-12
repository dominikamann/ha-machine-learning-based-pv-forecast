import json
from pathlib import Path

from custom_components.ml_pv_forecast.const import DOMAIN

ROOT = Path(__file__).parents[1]


def test_manifest_matches_domain():
    manifest = json.loads((ROOT / "custom_components/ml_pv_forecast/manifest.json").read_text())
    assert manifest["domain"] == DOMAIN
    assert manifest["config_flow"] is True
    assert any(r.startswith("scipy") for r in manifest["requirements"])


def test_integration_only_needs_packages_with_ha_os_wheels():
    # scikit-learn/joblib have no musllinux aarch64 wheels: Home Assistant OS cannot install them.
    manifest = json.loads((ROOT / "custom_components/ml_pv_forecast/manifest.json").read_text())
    assert not any(r.startswith(("scikit-learn", "joblib")) for r in manifest["requirements"])
    sources = (ROOT / "custom_components/ml_pv_forecast").rglob("*.py")
    offenders = [p.name for p in sources if any(m in p.read_text() for m in ("sklearn", "joblib"))]
    assert offenders == []
