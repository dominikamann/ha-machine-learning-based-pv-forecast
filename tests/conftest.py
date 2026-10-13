import pytest

from custom_components.ml_pv_forecast.core import model as model_module
from custom_components.ml_pv_forecast.core.variants import STANDARD


@pytest.fixture(autouse=True)
def standard_variant_only(request, monkeypatch):
    """Training four variants is slow; only tests marked ``all_variants`` do it."""
    if request.node.get_closest_marker("all_variants") is None:
        monkeypatch.setattr(model_module, "VARIANTS", (STANDARD,))
