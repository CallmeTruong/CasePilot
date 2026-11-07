from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-live", action="store_true", help="Run paid provider regression tests")
    parser.addoption("--run-models", action="store_true", help="Load pinned local embedding models")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-models"):
        for item in items:
            if "models" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Local models require --run-models"))
    if not config.getoption("--run-live"):
        skip = pytest.mark.skip(reason="Paid provider calls require --run-live")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def retail_environment():
    from casepilot.integration.benchmark import load_manifest, prepare_runtime

    manifest = Path(__file__).resolve().parents[1] / "configs/benchmark-manifest.json"
    prepare_runtime(load_manifest(manifest), manifest)
    from tau2.domains.retail.environment import get_environment

    return get_environment()
