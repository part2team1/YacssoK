"""어노테이션 읽기와 PyTorch Dataset (데이터 B 담당).

대회 어노테이션은 COCO 형식 JSON이다. 두 가지 배치를 모두 읽는다.
  1) JSON 파일 하나에 모든 이미지가 들어 있는 경우
  2) 폴더 아래에 JSON이 여러 개 흩어져 있는 경우 (알약 하나당 JSON 한 개 등)
어느 경우든 이미지 파일 이름 기준으로 합쳐서 Record 목록을 만든다.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.interfaces import ClassInfo, Record, Target
from src.utils import category_to_label, load_json, resolve_path

# ── 1. 어노테이션 → Record 목록 ─────────────────────────────


def _iter_json_files(ann_path: Path) -> list[Path]:
    if ann_path.is_file():
        return [ann_path]
    return sorted(ann_path.rglob("*.json"))


def load_records(ann_path: str | Path) -> tuple[list[Record], dict[int, str]]:
    """COCO 형식 JSON(파일 또는 폴더)을 읽어 이미지별 Record와 {category_id: 이름}을 돌려준다.

    - bbox는 COCO 원본 xywh를 팀 표준 xyxy로 바꿔 저장한다.
    - 같은 이미지가 여러 JSON에 나뉘어 있으면 file_name 기준으로 합친다.
    - 같은 박스가 두 번 들어 있으면 하나만 남긴다.
    """
    ann_path = resolve_path(ann_path)
    images: dict[str, dict] = {}
    boxes: dict[str, list] = defaultdict(list)
    categories: dict[int, str] = {}

    for json_file in _iter_json_files(ann_path):
        data = load_json(json_file)
        local_images = {img["id"]: img for img in data.get("images", [])}
        for cat in data.get("categories", []):
            # 카테고리 이름이 비어 있으면 이미지 정보의 제품명(dl_name)을 쓴다
            name = cat.get("name") or next(
                (im.get("dl_name") for im in local_images.values() if im.get("dl_name")), None
            )
            categories.setdefault(int(cat["id"]), str(name or cat["id"]))
        for img in local_images.values():
            images.setdefault(img["file_name"], img)
        for ann in data.get("annotations", []):
            img = local_images.get(ann["image_id"])
            bbox = ann.get("bbox")
            if (
                img is None or not isinstance(bbox, list) or len(bbox) != 4
            ):  # 대회 규칙: 길이 4인 bbox만 유효
                continue
            x, y, w, h = (float(v) for v in bbox)
            item = (
                round(x, 1),
                round(y, 1),
                round(x + w, 1),
                round(y + h, 1),
                int(ann["category_id"]),
            )
            if item not in boxes[img["file_name"]]:
                boxes[img["file_name"]].append(item)

    records: list[Record] = []
    for file_name, img in sorted(images.items()):
        items = boxes.get(file_name, [])
        records.append(
            Record(
                image_id=int(img["id"]),
                file_name=file_name,
                width=int(img.get("width", 0)),
                height=int(img.get("height", 0)),
                boxes=[list(it[:4]) for it in items],
                category_ids=[it[4] for it in items],
            )
        )
    return records, categories


def build_class_map(records: list[Record], categories: dict[int, str]) -> list[ClassInfo]:
    """학습 데이터에 실제로 나온 category_id만 골라 0부터 번호를 붙인다."""
    used = sorted({cid for r in records for cid in r["category_ids"]})
    return [
        ClassInfo(label_idx=i, category_id=cid, name=categories.get(cid, str(cid)))
        for i, cid in enumerate(used)
    ]


# ── 2. 이미지 읽기 ───────────────────────────────────────────


def read_image_rgb(path: str | Path) -> np.ndarray:
    """OpenCV는 BGR로 읽으므로 반드시 RGB로 바꾼다. 한글 경로도 읽을 수 있게 imdecode 사용."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── 3. PyTorch Dataset ──────────────────────────────────────


class PillDataset(Dataset):
    """(image, target)을 돌려준다. 형식은 interfaces.Target 참고.

    image: float32 [3, H, W], RGB, 0~1
    target["boxes"]: xyxy 절대 픽셀 / target["labels"]: label_idx
    """

    def __init__(
        self,
        records: list[Record],
        image_dir: str | Path,
        class_map: list[ClassInfo],
        transforms=None,
    ):
        self.records = records
        self.image_dir = resolve_path(image_dir)
        self.cat2label = category_to_label(class_map)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Target]:
        rec = self.records[idx]
        img = read_image_rgb(self.image_dir / rec["file_name"])
        h, w = img.shape[:2]
        boxes = np.array(rec["boxes"], dtype=np.float32).reshape(-1, 4)
        labels = np.array([self.cat2label[c] for c in rec["category_ids"]], dtype=np.int64)

        if self.transforms is not None:
            img, boxes, labels = self.transforms(img, boxes, labels)

        image = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        target = Target(
            boxes=torch.from_numpy(boxes).float().reshape(-1, 4),
            labels=torch.from_numpy(labels).long(),
            image_id=rec["image_id"],
            orig_size=(h, w),
        )
        return image, target


def collate_fn(batch):
    """이미지마다 박스 개수가 달라 리스트로 묶는다."""
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


# ── 4. 분할 목록 읽기 ────────────────────────────────────────


def load_split(records: list[Record], split_file: str | Path) -> list[Record]:
    """splits/{version}/train.txt 같은 파일(이미지 이름 한 줄씩)로 Record를 고른다."""
    names = {
        line.strip() for line in open(resolve_path(split_file), encoding="utf-8") if line.strip()
    }
    return [r for r in records if r["file_name"] in names]


def list_images(image_dir: str | Path) -> list[Path]:
    from src.interfaces import IMAGE_EXTS

    image_dir = resolve_path(image_dir)
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
