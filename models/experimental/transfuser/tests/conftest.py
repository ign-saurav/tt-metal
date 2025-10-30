import pytest
import os


def pytest_addoption(parser):
    parser.addoption("--data-root", action="store", default=None, help="Path to dataset root")
    parser.addoption("--frame", action="store", default=None, help="Frame id like 0120")
    parser.addoption("--image-variant", action="store", default=None, help="Image variant (raw/imagenet/etc.)")


@pytest.fixture(scope="session")
def cli_args(request):
    data_root = (
        request.config.getoption("--data-root")
        or os.environ.get("TRANSFUSER_DATA_ROOT")
        or "Scenario3_Town01_curved_route0_11_23_20_02_59"
    )
    frame = request.config.getoption("--frame") or os.environ.get("TRANSFUSER_FRAME") or "0120"
    image_variant = request.config.getoption("--image-variant") or os.environ.get("TRANSFUSER_IMAGE_VARIANT") or "raw"
    return {"data_root": data_root, "frame": frame, "image_variant": image_variant}
