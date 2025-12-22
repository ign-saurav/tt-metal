import os
import subprocess
from pathlib import Path
from loguru import logger
import pytest


@pytest.fixture(scope="session")
def is_ci_v2_env():
    yield "TT_GH_CI_INFRA" in os.environ


class CIv2ModelDownloadUtils_:
    @staticmethod
    def download_from_ci_v2_cache(
        model_path,
        timeout_in_s,
        download_dir_suffix="",
        endpoint_prefix="http://large-file-cache.large-file-cache.svc.cluster.local//mldata/model_checkpoints/pytorch/huggingface",
    ):
        assert model_path, f"model_path cannot be empty when downloading - what is wrong with you?: {model_path}"
        assert isinstance(
            timeout_in_s, int
        ), f"{timeout_in_s} is not an integer, which it should be because it's a timeout duration"

        download_dir = Path("/tmp/ttnn_model_cache/") / download_dir_suffix
        download_dir.mkdir(parents=True, exist_ok=True)
        download_dir_str = str(download_dir)

        if model_path and not model_path.endswith("/"):
            model_path = model_path + "/"

        endpoint = f"{endpoint_prefix}/{model_path}"

        try:
            subprocess.run(
                [
                    "wget",
                    "-r",
                    "-nH",
                    "-x",
                    "--cut-dirs=5",
                    "-np",
                    "--progress=dot:giga",
                    "-R",
                    "index.html*",
                    "-P",
                    download_dir_str,
                    endpoint,
                ],
                check=True,
                text=True,
                timeout=timeout_in_s,
            )
        except subprocess.TimeoutExpired as err:
            logger.error(f"Timeout of {timeout_in_s} seconds occurred while downloading from {endpoint}.")
            raise err
        except Exception as err:
            logger.error(
                f"Unknown error occurred while trying to download from {endpoint}. Check above logs from wget call."
            )
            logger.error(err)
            raise err

        return download_dir / Path(model_path)


@pytest.fixture(scope="session")
def model_location_generator(is_ci_v2_env):
    def model_location_generator_(
        model_version,
        model_subdir="",
        download_if_ci_v2=False,
        ci_v2_timeout_in_s=300,
        endpoint_prefix="http://large-file-cache.large-file-cache.svc.cluster.local//mldata/model_checkpoints/pytorch/huggingface",
        download_dir_suffix="model_weights",
    ):
        model_folder = Path("tt_dnn-models") / model_subdir
        internal_weka_path = Path("/mnt/MLPerf") / model_folder / model_version
        has_internal_weka = internal_weka_path.exists()

        download_from_ci_v2 = download_if_ci_v2 and is_ci_v2_env

        if download_from_ci_v2:
            assert (
                not has_internal_weka
            ), "For some reason, we see a file existing at the expected MLPerf location: {internal_weka_path} on CIv2. Please use the opportunity to clean up your model and get rid of MLPerf if you're moving to CIv2"
            assert (
                not model_subdir
            ), f"model_subdir is set to {model_subdir}, but we don't support further levels of directories in the large file cache in CIv2"
            civ2_download_path = CIv2ModelDownloadUtils_.download_from_ci_v2_cache(
                model_version,
                download_dir_suffix=download_dir_suffix,
                timeout_in_s=ci_v2_timeout_in_s,
                endpoint_prefix=endpoint_prefix,
            )
            logger.info(f"For model location, using CIv2 large file cache: {civ2_download_path}")
            return civ2_download_path
        elif has_internal_weka:
            logger.info(f"For model location, using internal MLPerf path: {internal_weka_path}")
            return internal_weka_path
        else:
            logger.info(
                f"For model location, local copy not found, so likely downloading straight from HF: {model_version}"
            )
            return model_version

    return model_location_generator_
