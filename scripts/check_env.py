"""환경설정 확인 스크립트.

실행: uv run python scripts/check_env.py
모든 항목이 OK이면 팀 채널에 결과 화면을 공유하세요.
"""

import importlib
import platform
import sys

PACKAGES = [
    "numpy",
    "pandas",
    "cv2",
    "PIL",
    "matplotlib",
    "yaml",
    "torch",
    "torchvision",
    "torchmetrics",
    "pycocotools",
    "ultralytics",
]


def main() -> None:
    print(f"Python   : {sys.version.split()[0]} ({platform.system()})")
    ok = True
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "-")
            print(f"[OK]   {name:<12} {version}")
        except Exception as e:  # 설치 누락이나 DLL 오류를 모두 보여 주기 위해 넓게 잡음
            ok = False
            print(f"[FAIL] {name:<12} {e}")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"GPU      : {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            print("GPU      : Apple MPS")
        else:
            print("GPU      : 없음 (CPU로 실행됩니다. 학습은 GPU 환경을 쓰세요)")
    except Exception:
        pass

    print(
        "\n환경설정 완료"
        if ok
        else "\n설치 실패 항목이 있습니다. Experimentation Lead에게 화면을 공유하세요."
    )


if __name__ == "__main__":
    main()
