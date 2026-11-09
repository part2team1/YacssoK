"""Official information keyed by the source product code, independent of model IDs."""

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = {}
SUPPLEMENTS = {}
PROFILES = {}
NAMES = []
PHOTO_DIRS = []


def configure_catalog(directory):
    """Load MFDS snapshots and restrict photo paths to product-code filenames."""
    DATA.clear()
    SUPPLEMENTS.clear()
    PROFILES.clear()
    NAMES.clear()
    PHOTO_DIRS.clear()
    directory = Path(directory).resolve()
    if not directory.exists():
        return
    files = ("drug_info_91.json", "drug_supplement_7.json", "drug_simple_91.json")
    loaded = [json.loads((directory / name).read_text(encoding="utf-8")) for name in files]
    data, supplements, profiles = loaded
    for item in data.values():
        seq = str(item["seq"])
        if not re.fullmatch(r"\d{9}", seq):
            raise ValueError("Invalid catalog product code")
        if image := item.get("image"):
            # Imported snapshots use assets/official_pills/<code>.png. Only allow
            # that exact product's local file, never an arbitrary config path.
            if str(image["item_seq"]) != seq or Path(image["path"]).name != seq + ".png":
                raise ValueError("Official photo differs from product code")
            photo = directory / "official_pills" / (seq + ".png")
            if not photo.is_file() or not photo.resolve().is_relative_to(directory):
                raise ValueError("Missing or unsafe official photo")
            image["path"] = str(photo)
        if seq not in profiles or str(profiles[seq]["item_seq"]) != seq:
            raise ValueError("Missing verified simple product information")
    DATA.update(data)
    SUPPLEMENTS.update(supplements)
    PROFILES.update(profiles)
    NAMES.extend(data)
    if any(item.get("image") for item in data.values()):
        PHOTO_DIRS.append(directory / "official_pills")


def clean(value):
    return html.escape(re.sub(r"<[^>]+>", "", str(value or "정보 없음")))


def format_date(value):
    text = str(value or "")
    digits = re.sub(r"\D", "", text)
    if len(digits) == 8:
        return f"{digits[:4]}.{digits[4:6]}.{digits[6:]}"
    return text or "일자 정보 없음"


def permit_status(value):
    # This expiry concerns a permit, not a particular tablet's expiry.
    return "허가 유효기간 만료" if value == "유효기간만료" else str(value or "정보 없음")


def render(name):
    item = DATA.get(name)
    if not item:
        return "<p>알약을 선택하면 공식 정보를 볼 수 있습니다.</p>"
    official, historic, seq = item["official"], item["historic"], item["seq"]
    if not re.fullmatch(r"\d{9}", str(seq)):
        raise ValueError("Invalid original product code")
    url = "https://nedrug.mfds.go.kr/pbp/CCBBB01/getItemDetail?itemSeq=" + seq
    checked = format_date(item["lookup_date"])
    if official:
        if str(official["ITEM_SEQ"]) != seq:
            raise ValueError("Official product code does not match source code")
        status = permit_status(official.get("CANCEL_NAME"))
        title = official.get("ITEM_NAME") or item["name"]
        badge = f"식약처 API 정보 · {checked} 조회"
        details = "".join(
            '<div style="margin:14px 0"><b>' + label + "</b>"
            '<p style="margin:5px 0;line-height:1.7;overflow-wrap:anywhere">'
            + clean(official.get(key))
            + "</p></div>"
            for label, key in [
                ("성분·함량", "MATERIAL_NAME"),
                ("성상", "CHART"),
                ("보관방법", "STORAGE_METHOD"),
            ]
        )
        if status != "정상":
            details = (
                '<p class="drug-status" style="color:#9a5a23">품목 상태: '
                + clean(status)
                + " · "
                + clean(format_date(official.get("CANCEL_DATE")))
                + "</p>"
                + details
            )
        if seq == "199501099":
            details = (
                '<p style="color:#67796e">목록 이름: '
                + clean(item["name"])
                + " · 같은 품목코드의 현재 공식 이름입니다.</p>"
                + details
            )
    else:
        if not historic or str(historic["item_seq"]) != seq:
            raise ValueError("Missing verified history for an unlisted product")
        title = item["name"]
        badge = f"공식 품목 이력 · {checked} 확인"
        details = (
            '<p class="drug-status" style="color:#9a5a23">품목 상태: '
            + clean(permit_status(historic["status"]))
            + " · "
            + clean(format_date(historic["status_date"]))
            + "</p>"
            "<p>제품정보 API에서 조회되지 않는 품목입니다. "
            "아래 공식 기록에서 품목 이력을 확인할 수 있습니다.</p>"
        )
        if seq == "200202194":
            details += "<p>외형 확인 필요: 학습 데이터의 흰색 장방형과 공식 기록의 연한 파란색 타원형이 다릅니다.</p>"
        supplement = SUPPLEMENTS.get(seq)
        if supplement:
            if supplement["item_seq"] != seq or supplement["evidence_kind"] != "official_detail":
                raise ValueError("Supplement does not match official product code")
            badge = "의약품안전나라 상세 기록 · " + format_date(supplement["checked_on"]) + " 확인"
            removed = supplement["use_status"] == "removed"
            heading = "효능·효과 삭제 기록" if removed else "이전 허가 기록의 효능·효과"
            details += (
                "<h4>" + heading + "</h4><p>" + html.escape(supplement["indications"]) + "</p>"
            )
            if removed:
                details += (
                    '<p>효과가 입증되지 않아 삭제된 용도를 현재의 효능으로 안내하지 않습니다.</p><a href="'
                    + html.escape(supplement["notice_url"], quote=True)
                    + '" target="_blank" rel="noopener noreferrer">식약처 사용 중지 권고 안내 ↗</a>'
                )
            else:
                details += (
                    "<p>위 내용은 이전 허가 기록이며, 현재 유효한 허가를 뜻하지 않습니다.</p>"
                )
    image = item.get("image")
    if image:
        if str(image["item_seq"]) != seq:
            raise ValueError("Official photo code does not match source code")
        photo = (
            '<figure style="margin:18px 0;padding:16px;background:#f7f9f7;'
            'border-radius:12px;text-align:center"><img loading="lazy" '
            'src="/gradio_api/file='
            + html.escape(image["path"], quote=True)
            + '" alt="'
            + clean(item["name"])
            + ' 공식 식별사진" '
            'style="width:100%;max-width:520px;max-height:250px;object-fit:contain">'
            '<figcaption style="font-size:12px;color:#67796e;margin-top:8px">'
            "식약처 낱알식별 공식 사진 · "
            '<a href="https://www.data.go.kr/data/15057639/openapi.do" '
            'target="_blank" rel="noopener noreferrer">사진 출처</a></figcaption></figure>'
        )
    elif item.get("photo_download_error"):
        photo = '<p style="color:#778278">공식 사진을 불러오지 못했습니다.</p>'
    else:
        photo = (
            '<p style="padding:18px;background:#f7f9f7;border-radius:12px;color:#778278">'
            "현재 연결된 공식 알약 사진이 없습니다.</p>"
        )
    return (
        '<section class="drug-card" data-item-seq="'
        + seq
        + '" style="padding:22px;border:1px solid #dce5df;border-radius:16px;'
        'background:white;overflow-wrap:anywhere"><small style="color:#67796e">'
        + badge
        + '</small><h3 style="margin:9px 0">'
        + clean(title)
        + "</h3>"
        + photo
        + details
        + '<a class="official-record-link" href="'
        + url
        + '" target="_blank" rel="noopener noreferrer">의약품안전나라 공식 기록 ↗</a>'
        '<p style="font-size:12px;color:#7c8780;margin:16px 0 0">품목번호 '
        + seq
        + " · 사진 인식 결과와 실제 제품이 일치하는지 확인해주세요.</p></section>"
    )
