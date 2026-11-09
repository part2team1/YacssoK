"""CSV 원본은 엔진 전용 임시 폴더에 두고, 다음 저장 시 만료 파일을 정리한다."""

import tempfile
import time
from pathlib import Path


class ResultFiles:
    def __init__(self, ttl_seconds=3600):
        self.ttl_seconds = ttl_seconds
        self._folder = None

    def save(self, frame):
        if self._folder is None:
            self._folder = tempfile.TemporaryDirectory(prefix="yacssok-results-")
        folder = Path(self._folder.name)
        cutoff = time.time() - self.ttl_seconds
        for path in folder.glob("pill_result_*.csv"):
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="pill_result_", suffix=".csv", dir=folder, delete=False
        ) as handle:
            path = handle.name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def close(self):
        if self._folder is not None:
            self._folder.cleanup()
            self._folder = None
