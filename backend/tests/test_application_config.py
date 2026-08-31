from __future__ import annotations

from pathlib import Path

from api.application_config import load_application_config


def test_application_data_root_is_relative_to_config(tmp_path: Path) -> None:
    data_root = tmp_path / "application-data"
    data_root.mkdir()
    config_path = tmp_path / "applications.yaml"
    config_path.write_text(
        """version: 1
paths:
  data_root: application-data
applications:
  max_workers: 2
  materials:
    id_card_front: DG12
    id_card_back: DG13
    driver_license: DG14
    vehicle_license: Z002
""",
        encoding="utf-8",
    )

    config = load_application_config(config_path)

    assert config.data_root == data_root.resolve()
    assert config.material_codes["vehicle_license"] == "Z002"
    assert config.max_workers == 2


def test_application_data_root_can_be_overridden(
    tmp_path: Path,
    monkeypatch,
) -> None:
    override = tmp_path / "override"
    monkeypatch.setenv("APPLICATION_DATA_ROOT", str(override))

    config = load_application_config()

    assert config.data_root == override.resolve()
