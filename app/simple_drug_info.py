"""Plain Korean representative uses, checked against exact-code official indications."""

import html

from .drug_info import DATA, PROFILES, SUPPLEMENTS, clean, format_date, permit_status, render


def render_simple(name):
    item = DATA.get(name)
    if not item:
        return "<p>알약을 선택하면 공식 정보를 볼 수 있습니다.</p>"
    seq = str(item["seq"])
    supplement = SUPPLEMENTS.get(seq)
    profile = supplement or PROFILES[seq]
    if profile["item_seq"] != seq:
        raise ValueError("Simple information differs from verified official product")
    if supplement:
        if (
            item["official"]
            or supplement["evidence_kind"] != "official_detail"
            or supplement["use_status"] not in ("historic", "removed")
        ):
            raise ValueError("Invalid official supplement")
    elif bool(profile["indications"]) != bool(item["official"]):
        raise ValueError("Simple information differs from verified official product")
    official, historic = item["official"], item["historic"]
    status = (
        permit_status(official.get("CANCEL_NAME"))
        if official
        else permit_status(historic["status"])
    )
    status_date = official.get("CANCEL_DATE") if official else historic["status_date"]
    warning = ""
    if status != "정상":
        warning = (
            '<p class="simple-status">품목 상태: '
            + clean(status)
            + " · "
            + clean(format_date(status_date))
            + "</p>"
        )
    if seq == "200202194":
        warning += '<p class="simple-status">사진의 모양 확인이 필요해요. 학습 사진과 공식 사진의 색과 모양이 다릅니다.</p>'
    photo = ""
    image = item.get("image")
    if image:
        if str(image["item_seq"]) != seq:
            raise ValueError("Official photo code does not match source code")
        photo = (
            '<figure class="simple-photo"><img src="/gradio_api/file='
            + html.escape(image["path"], quote=True)
            + '" alt="'
            + clean(name)
            + ' 공식 사진"><figcaption>식약처 공식 알약 사진</figcaption></figure>'
        )
    else:
        photo = "<p>현재 연결된 공식 알약 사진이 없습니다.</p>"
    url = "https://nedrug.mfds.go.kr/pbp/CCBBB01/getItemDetail?itemSeq=" + seq
    if supplement:
        heading = (
            "공식 기록 확인" if supplement["use_status"] == "removed" else "이전 허가 기록의 용도"
        )
        helper = (
            "삭제된 효능과 식약처 안내를 아래에서 확인하세요."
            if supplement["use_status"] == "removed"
            else "이전 허가 기록의 설명입니다. 현재 허가 상태도 함께 확인해주세요."
        )
    else:
        heading = "대표 용도"
        helper = (
            "공식 정보를 쉽게 줄인 설명입니다. 전체 용도는 아래에서 확인하세요."
            if profile["indications"]
            else "용도를 확인할 공식 자료가 아직 연결되지 않았습니다."
        )
    lookup_notice = ""
    if not official:
        lookup_notice = (
            '<aside class="drug-lookup-alert" role="alert" aria-atomic="true" '
            'style="padding:18px 20px;margin:16px 0;border:2px solid #b77924;'
            'border-radius:12px;background:#fff4e5;color:#704412">'
            '<strong style="font-size:24px">조회되지 않습니다.</strong>'
            '<p style="margin:8px 0 0">현재 식약처 제품정보에서 조회되지 않는 품목입니다. '
            "아래는 공식 상세 기록에서 확인한 내용입니다.</p></aside>"
        )
    return (
        '<section class="drug-card simple-drug-card" data-item-seq="' + seq + '">'
        "<h3>" + clean(name) + "</h3>" + lookup_notice + "<small>" + heading + "</small>"
        '<div class="simple-purpose">' + clean(profile["label"]) + "</div>"
        "<p>"
        + clean(profile["summary"])
        + "</p>"
        + warning
        + photo
        + "<small>"
        + helper
        + "</small>"
        '<details class="simple-details"><summary>자세한 공식 정보 보기</summary>'
        + (
            "<h3>효능·효과 원문</h3><p>" + html.escape(profile["indications"]) + "</p>"
            if profile["indications"] and not supplement
            else ""
        )
        + render(name)
        + '</details><a href="'
        + url
        + '" target="_blank" rel="noopener noreferrer">식약처 공식 정보 보기 ↗</a></section>'
    )
