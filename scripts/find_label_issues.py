"""라벨이 잘못됐을 수 있는 박스를 자동으로 찾아 목록으로 만든다 (실험 D10, 데이터 B / D11 --ocr, 데이터 A).

학습된 모델의 예측과 정답을 비교해 사람이 확인할 후보만 골라 준다. 232장을 다 보지 않아도 된다.
후보를 눈으로 확인한 뒤, 맞는 줄만 data/annotations/manual_fix.csv 에 옮기면 clean_labels.py가 고친다.

실행:
    uv run python scripts/find_label_issues.py --exp <학습된 실험> --split val
    uv run python scripts/find_label_issues.py --exp <학습된 실험> --split all --draw      # 확인용 그림까지
    uv run python scripts/find_label_issues.py --exp <학습된 실험> --split all --ocr       # 각인 OCR 비교 추가 (D11)

결과 (대회 데이터에서 나온 것이므로 커밋하지 않음):
    outputs/<실험>/label_issues_<split>.csv   후보 목록 (manual_fix.csv와 같은 앞 네 칸 + 근거)
    outputs/<실험>/label_issues/*.jpg         --draw: 초록=정답, 빨강=예측 (번호는 label_idx)

찾는 것
  wrong_class   모델이 높은 확신도로 같은 자리에 다른 알약을 예측 → 클래스가 틀렸을 수 있음
  loose_box     같은 알약을 찾았지만 박스 겹침(IoU)이 0.5~0.75 → 정답 박스가 헐겁거나 어긋났을 수 있음
  missing_label 모델이 확신하는데 근처에 정답 박스가 없음 → 라벨이 빠졌을 수 있음
  not_found     확신도가 낮은 예측까지 봐도 정답 박스 자리에 아무것도 없음 → 라벨 오류이거나 어려운 사진
  ocr_mismatch  (--ocr) 읽은 각인이 정답 알약보다 다른 알약 각인과 훨씬 비슷함
모델이 틀린 경우도 섞여 있으므로 반드시 사진을 보고 판단한다. 검증셋(val)부터 확인한다.
"""

from __future__ import annotations

import argparse
import csv

import cv2
import torch
from torchvision.ops import box_iou

from src.dataset import load_records, load_split, read_image_rgb
from src.infer import load_experiment
from src.interfaces import OUTPUT_DIR
from src.utils import category_to_label, label_to_category, load_class_map, resolve_path

FIELDS = [
    "file_name",
    "category_id",
    "action",
    "new_value",
    "issue",
    "pred_category_id",
    "score",
    "iou",
    "gt_box",
    "pred_box",
    "ocr_text",
]


def _fmt(box) -> str:
    return " ".join(f"{float(v):.1f}" for v in box)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="학습된 실험 이름 (outputs/ 아래)")
    ap.add_argument("--split", choices=["val", "train", "all"], default="val")
    ap.add_argument("--conf", type=float, default=0.5, help="이 확신도 이상 예측만 근거로 씀")
    ap.add_argument("--draw", action="store_true", help="후보 사진을 그림으로 저장")
    ap.add_argument("--ocr", action="store_true", help="각인 OCR로 클래스 비교 (D11)")
    ap.add_argument(
        "--ocr-margin", type=float, default=0.3, help="다른 알약이 이만큼 더 비슷하면 후보"
    )
    args = ap.parse_args()

    cfg, model = load_experiment(args.exp)
    d = cfg["data"]
    records, _ = load_records(d["train_annotations"])
    if args.split != "all":
        records = load_split(records, f"{d['split_dir']}/{d['version']}/{args.split}.txt")
    class_map = load_class_map(d["class_map"])
    c2l, l2c = category_to_label(class_map), label_to_category(class_map)
    image_dir = resolve_path(d["train_images"])

    ocr = table = None
    if args.ocr:
        from src.ocr import PillOCR, imprint_table, match

        ocr = PillOCR()
        table = imprint_table(class_map, resolve_path(d["class_map"]).parent / "drug_info.json")
        print(f"각인 정보가 있는 알약 {len(table)}종 / 전체 {len(class_map)}종")

    out_dir = OUTPUT_DIR / args.exp
    draw_dir = out_dir / "label_issues"
    if args.draw:
        draw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    preds = model.predict([image_dir / r["file_name"] for r in records], conf_thr=0.05)
    for r, p in zip(records, preds, strict=True):
        gt = torch.tensor(r["boxes"], dtype=torch.float32).reshape(-1, 4)
        gt_lbl = torch.tensor([c2l[c] for c in r["category_ids"]], dtype=torch.long)
        keep = p["scores"] >= args.conf
        pb, ps, pl = p["boxes"][keep], p["scores"][keep], p["labels"][keep]
        iou = box_iou(gt, pb) if len(gt) and len(pb) else torch.zeros((len(gt), len(pb)))
        found = []
        for i, cid in enumerate(r["category_ids"]):
            base = {"file_name": r["file_name"], "category_id": cid, "gt_box": _fmt(gt[i])}
            same = (pl == gt_lbl[i]).nonzero().flatten().tolist()
            other = (pl != gt_lbl[i]).nonzero().flatten().tolist()
            best_same = max(same, key=lambda j: iou[i, j], default=None)
            best_other = max(other, key=lambda j: iou[i, j], default=None)
            if best_same is not None and iou[i, best_same] >= 0.75:
                pass  # 정상: 같은 알약을 정확한 자리에서 찾음
            elif best_same is not None and iou[i, best_same] >= 0.5:
                j = best_same
                rows.append(
                    {
                        **base,
                        "action": "box",
                        "new_value": _fmt(pb[j]),
                        "issue": "loose_box",
                        "pred_category_id": cid,
                        "score": f"{ps[j]:.3f}",
                        "iou": f"{iou[i, j]:.3f}",
                        "pred_box": _fmt(pb[j]),
                    }
                )
            elif best_other is not None and iou[i, best_other] >= 0.7:
                j = best_other
                rows.append(
                    {
                        **base,
                        "action": "class",
                        "new_value": l2c[int(pl[j])],
                        "issue": "wrong_class",
                        "pred_category_id": l2c[int(pl[j])],
                        "score": f"{ps[j]:.3f}",
                        "iou": f"{iou[i, j]:.3f}",
                        "pred_box": _fmt(pb[j]),
                    }
                )
            elif not len(p["boxes"]) or float(box_iou(gt[i : i + 1], p["boxes"]).max()) < 0.5:
                # 낮은 확신도 예측까지 봐도 이 자리에 아무것도 없음
                rows.append({**base, "action": "check", "new_value": "", "issue": "not_found"})
            found += [j for j in (best_same, best_other) if j is not None and iou[i, j] >= 0.3]
        for j in range(len(pb)):  # 어떤 정답과도 겹치지 않는 확신 높은 예측
            if j not in found and (len(gt) == 0 or float(iou[:, j].max()) < 0.3):
                rows.append(
                    {
                        "file_name": r["file_name"],
                        "category_id": l2c[int(pl[j])],
                        "action": "add",
                        "new_value": _fmt(pb[j]),
                        "issue": "missing_label",
                        "pred_category_id": l2c[int(pl[j])],
                        "score": f"{ps[j]:.3f}",
                        "pred_box": _fmt(pb[j]),
                    }
                )
        img = None
        if ocr is not None:
            img = read_image_rgb(image_dir / r["file_name"])
            for i, cid in enumerate(r["category_ids"]):
                text, _ = ocr.read(img, gt[i].tolist())
                ranked = match(text, table)
                if not ranked:
                    continue
                own = dict(ranked).get(int(gt_lbl[i]), 0.0)
                best_label, best_sim = ranked[0]
                if best_label != int(gt_lbl[i]) and best_sim - own >= args.ocr_margin:
                    rows.append(
                        {
                            "file_name": r["file_name"],
                            "category_id": cid,
                            "action": "class",
                            "new_value": l2c[best_label],
                            "issue": "ocr_mismatch",
                            "pred_category_id": l2c[best_label],
                            "score": f"{best_sim:.3f}",
                            "gt_box": _fmt(gt[i]),
                            "ocr_text": text,
                        }
                    )
        if args.draw and any(row["file_name"] == r["file_name"] for row in rows):
            img = read_image_rgb(image_dir / r["file_name"]) if img is None else img
            canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            for b, lbl in zip(gt.tolist(), gt_lbl.tolist(), strict=True):
                cv2.rectangle(
                    canvas, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0), 3
                )
                cv2.putText(canvas, f"GT {lbl}", (int(b[0]), int(b[1]) - 6), 0, 0.8, (0, 200, 0), 2)
            for b, lbl in zip(pb.tolist(), pl.tolist(), strict=True):
                cv2.rectangle(
                    canvas, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 230), 2
                )
                cv2.putText(canvas, f"P {lbl}", (int(b[0]), int(b[3]) + 22), 0, 0.8, (0, 0, 230), 2)
            ok, buf = cv2.imencode(".jpg", canvas)
            buf.tofile(str(draw_dir / (r["file_name"].rsplit(".", 1)[0] + ".jpg")))

    out = out_dir / f"label_issues_{args.split}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:  # 엑셀에서 바로 열림
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["issue"]] = counts.get(row["issue"], 0) + 1
    print(f"이미지 {len(records)}장 확인, 후보 {len(rows)}건: {counts or '없음'}")
    print(f"저장: {out}" + (f", 그림: {draw_dir}" if args.draw else ""))
    print(
        "확인 후 맞는 줄의 앞 네 칸(file_name,category_id,action,new_value)만 manual_fix.csv에 옮기세요."
    )


if __name__ == "__main__":
    main()