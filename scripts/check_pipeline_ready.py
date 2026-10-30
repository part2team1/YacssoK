"""run_pipeline.py를 돌리기 전에 레포에 필요한 기능이 다 들어와 있는지 확인한다 (레포 폴더에서 실행).

실행:  uv run python <이 파일 경로>
없음으로 나온 항목은 정답본 zip의 같은 파일과 VS Code에서 비교해 그 부분을 가져온다.
"""

from pathlib import Path

ROOT = Path.cwd()
CHECKS = [
    # (파일, 들어 있어야 하는 글자, 무엇인지 / 어느 가이드 단계)
    ("src/utils.py", "def apply_overrides", "--set 설정 바꾸기 (모델 가이드 X0)"),
    ("src/utils.py", "def yolo_data_yaml", "오버샘플링용 data.yaml (데이터 가이드 D1)"),
    ("scripts/run_train.py", '"--set"', "run_train --set (모델 가이드 X0)"),
    ("scripts/run_train.py", '"--exp-name"', "run_train --exp-name (모델 가이드 X0)"),
    ("scripts/run_train.py", '"--epochs"', "run_train --epochs (모델 가이드 X0)"),
    ("scripts/export_yolo.py", "apply_overrides", "export_yolo --set (모델 가이드 X0)"),
    ("scripts/evaluate.py", '"--val-config"', "evaluate --val-config (모델 가이드 PS)"),
    ("scripts/evaluate.py", '"--class-metrics"', "evaluate --class-metrics (모델 가이드 PS)"),
    ("src/metrics.py", '"classes"', "클래스별 AP에 classes 저장 (metrics.py 수정)"),
    ("scripts/tune_yolo.py", '"--set"', "하이퍼파라미터 탐색 (모델 가이드 T3)"),
    ("scripts/make_submission.py", '"--exp"', "제출 파일 (모델 가이드 PS)"),
    ("scripts/run_matrix.py", "ocr", "모델 × 실험 일괄 실행 (D11 포함 판)"),
    ("scripts/plot_matrix.py", "def load_curve", "결과 그래프"),
    ("scripts/run_pipeline.py", "def write_report", "전체 파이프라인"),
    ("configs/matrix_data.yaml", "copypaste_ocr", "계획 파일 (D11 포함 판)"),
    ("configs/yolo_baseline.yaml", "yolov8s", "모델 yolov8s"),
    ("configs/yolo11s.yaml", "yolo11s", "모델 yolo11s (모델 가이드 M2)"),
    ("configs/yolo26s.yaml", "yolo26s", "모델 yolo26s (모델 가이드 M2)"),
    ("configs/yolo_v8m.yaml", "yolov8m", "모델 yolov8m (모델 가이드 M1)"),
    ("configs/ocr_rerank.yaml", "ocr_rerank", "D11 설정 (선택)"),
    ("src/models/ocr_rerank.py", "class OcrRerankDetector", "D11 모델 (선택)"),
    ("app/demo.py", "def build_app", "웹 데모 (선택)"),
]

missing = 0
for path, needle, what in CHECKS:
    f = ROOT / path
    ok = f.exists() and needle in f.read_text(encoding="utf-8", errors="ignore")
    missing += not ok
    status = "OK  " if ok else ("없음" if not f.exists() else "부족")
    print(f"[{status}] {path:32s} {what}")
print(
    "\n모두 준비됨"
    if not missing
    else f"\n준비 안 된 항목 {missing}개 — 위 '없음/부족' 줄을 정답본 zip에서 가져오세요."
)
