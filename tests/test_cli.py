from pathlib import Path

from landscapyml.__main__ import _extract_config_path_from_overrides


def test_extract_config_path_from_file(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("test")
    overrides, cfg_path = _extract_config_path_from_overrides([str(cfg), "foo=1"])
    assert overrides == ["foo=1"]
    assert cfg_path == cfg


def test_extract_config_path_from_dir(tmp_path: Path):
    cfg_dir = tmp_path / "conf"
    cfg_dir.mkdir()
    cfg = cfg_dir / "config.yaml"
    cfg.write_text("test")
    overrides, cfg_path = _extract_config_path_from_overrides([str(cfg_dir)])
    assert overrides == []
    assert cfg_path == cfg


def test_extract_config_path_ignores_overrides():
    overrides, cfg_path = _extract_config_path_from_overrides(["foo=1", "+bar=2"])
    assert overrides == ["foo=1", "+bar=2"]
    assert cfg_path is None
