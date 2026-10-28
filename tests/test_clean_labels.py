"""수동 라벨 수정은 원본 박스를 기준으로 적용한다."""

from copy import deepcopy

import pytest

from scripts.clean_labels import apply_fixes, clean


def record():
    return dict(
        image_id=1,
        file_name="sample.png",
        width=100,
        height=100,
        boxes=[[0, 0, 20, 20], [30, 30, 50, 50]],
        category_ids=[10, 20],
    )


def fix(cid, action, value):
    return dict(file_name="sample.png", category_id=str(cid), action=action, new_value=value)


@pytest.mark.parametrize("reverse", [False, True])
def test_swap_uses_original_labels(reverse):
    r = record()
    before = deepcopy(r)
    fixes = [fix(10, "class", "20"), fix(20, "class", "10")]
    result = apply_fixes(r, fixes[::-1] if reverse else fixes, [])
    assert result["category_ids"] == [20, 10]
    assert r == before


def test_chain_does_not_relabel_previous_target():
    result = apply_fixes(record(), [fix(10, "class", "20"), fix(20, "class", "30")], [])
    assert result["category_ids"] == [20, 30]


def test_added_box_is_not_targeted_by_later_class_fix():
    result = apply_fixes(record(), [fix(10, "add", "60 60 80 80"), fix(10, "class", "30")], [])
    assert result["category_ids"] == [30, 20, 10]


def test_box_fix_still_targets_original_class_after_relabel():
    result = apply_fixes(record(), [fix(10, "class", "20"), fix(10, "box", "1 2 21 22")], [])
    assert result["boxes"] == [[1, 2, 21, 22], [30, 30, 50, 50]]


def test_missing_target_raises():
    with pytest.raises(ValueError, match="수정 대상 없음"):
        apply_fixes(record(), [fix(99, "class", "10")], [])


def test_fix_restores_dropped_image_but_check_does_not():
    dropped = {("sample.png", None)}
    restored, _ = clean([record()], dropped, [fix(10, "class", "30")])
    checked, _ = clean([record()], dropped, [fix(10, "check", "")])
    assert len(restored) == 1
    assert checked == []
