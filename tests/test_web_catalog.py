"""Cross-check all public catalog codes, image files and unavailable states."""

import hashlib
import json
from pathlib import Path

import pytest

from app import drug_info, simple_drug_info

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def catalog():
    drug_info.configure_catalog(ROOT / "app/assets/drug_catalog")


def test_all_public_files_match_manifest():
    manifest = json.loads((ROOT / "docs/web_catalog_manifest.json").read_text())
    listed = {ROOT / item["file"] for item in manifest["files"]}
    actual = {path for path in (ROOT / "app/assets/drug_catalog").rglob("*") if path.is_file()}
    assert listed == actual
    for item in manifest["files"]:
        content = (ROOT / item["file"]).read_bytes()
        assert len(content) == item["bytes"]
        assert hashlib.sha256(content).hexdigest() == item["sha256"]


def test_every_product_renders_with_same_official_code():
    assert len(drug_info.DATA) == 91
    unavailable = photos = 0
    for name, item in drug_info.DATA.items():
        detail = drug_info.render(name)
        simple = simple_drug_info.render_simple(name)
        seq = item["seq"]
        assert f'data-item-seq="{seq}"' in detail
        assert f'data-item-seq="{seq}"' in simple
        assert "itemSeq=" + seq in detail
        assert "itemSeq=" + seq in simple
        if not item["official"]:
            unavailable += 1
            assert "조회되지 않습니다." in simple
        if item.get("image"):
            photos += 1
            assert Path(item["image"]["path"]).is_file()
    assert unavailable == 7
    assert photos == 83


def test_unknown_name_is_not_given_an_invented_use():
    assert "공식 정보를" in simple_drug_info.render_simple("unknown pill")


def test_catalog_cannot_serve_arbitrary_file(tmp_path):
    seq = "123456789"
    data = {"test": {"seq": seq, "image": {"item_seq": seq, "path": "secret.key"}}}
    (tmp_path / "drug_info_91.json").write_text(json.dumps(data))
    (tmp_path / "drug_supplement_7.json").write_text("{}")
    (tmp_path / "drug_simple_91.json").write_text("{}")
    with pytest.raises(ValueError, match="Official photo"):
        drug_info.configure_catalog(tmp_path)
