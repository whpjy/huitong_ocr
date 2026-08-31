from __future__ import annotations

from pathlib import Path

import pytest

from api.mobile_config import MobileConfigError, load_mobile_config


def _write_config(path: Path, first: bool, second: bool) -> None:
    path.write_text(
        f"""version: 1
mobile:
  models:
    hunyuan_ocr:
      enabled: {str(first).lower()}
      model_key: multimodal:hunyuan_ocr
      label: HunyuanOCR
    hunyuan_ocr_ppocr:
      enabled: {str(second).lower()}
      model_key: hybrid:hunyuan_ocr
      label: HunyuanOCR + PP-OCRv6
""",
        encoding="utf-8",
    )


def test_mobile_image_quality_defaults_to_disabled(tmp_path: Path) -> None:
    config_path = tmp_path / "mobile.yaml"
    _write_config(config_path, True, False)

    config = load_mobile_config(config_path)

    assert config.image_quality_enabled is False


def test_mobile_image_quality_can_be_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "mobile.yaml"
    _write_config(config_path, True, False)
    content = config_path.read_text(encoding="utf-8").replace(
        "mobile:\n",
        "mobile:\n  image_quality:\n    enabled: true\n",
    )
    config_path.write_text(content, encoding="utf-8")

    config = load_mobile_config(config_path)

    assert config.image_quality_enabled is True


@pytest.mark.parametrize("first,second", [(False, False), (True, True)])
def test_mobile_config_requires_exactly_one_enabled_model(
    tmp_path: Path,
    first: bool,
    second: bool,
) -> None:
    config_path = tmp_path / "mobile.yaml"
    _write_config(config_path, first, second)

    with pytest.raises(MobileConfigError, match="必须且只能启用一个模型"):
        load_mobile_config(config_path)
