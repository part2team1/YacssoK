# =====================================================================
# src/eda.py 맨 아래에 붙여넣기 (json 메타데이터 EDA)
# =====================================================================
# 위쪽에 이미 있는 import(pd, Path, resolve_path)를 그대로 씀.
# 아래 json, re import 는 eda.py 맨 위 import 블록으로 옮겨도 됨 (린터 E402 경고 방지).
#
# [핵심] 6자리 숫자(K-코드)가 모든 표를 잇는 "키"임.
#   - 문자열 형태 'K-029451' : 약 폴더명, images[].drug_N, images[].dl_mapping_code
#   - 정수 형태   29451      : annotations[].category_id, categories[].id, images[].dl_idx
#   - 위 값들은 train json 763개 전부 서로 일치함 (annotation_consistency 로 확인).
#   - 코드 1개 = 약 이름 1개 (1:1). 같은 코드면 회사·모양·색·각인도 json 마다 전부 같음.
#   → 그래서 기존 boxes_df / class_df 의 category_id 에 "약 1개 = 1행" 표를 merge 하면 끝.
#
# [표 3단계]
#   pill_df : json 1개 = 박스 1개 = 1행 (763행). 원본 값 + 파생 컬럼
#   drug_df : 약(K-코드) 1개 = 1행 (56행). 이름·회사·외형 + 박스 수
#   merged  : 기존 class_distribution() 결과 + drug_df (56행)
# =====================================================================

import json
import re

# ---------------------------------------------------------------------
# 의미 있다고 고른 필드 (그룹별)
#   뺀 필드와 이유:
#   - width/height, drug_S, back_color 외 상수값 → 값이 1종류뿐
#     (back_color, light_color 는 "정말 상수인지" 보여주려고 촬영조건에 남겨둠)
#   - *_en (영문명), img_key/mark_code_*_img (URL) → 분석에 의미 없음
#   - item_seq, di_edi_code, 각종 날짜 → 행정 정보
#   - mark_code_* → 거의 다 빈 값 (745개 중 642~745개 빈 값)
#   - chart → '황색의 타원형 필름코팅정' 같은 문장. drug_shape/color 로 대체 가능
# ---------------------------------------------------------------------
META_FIELDS = {
    "식별": [
        "dl_mapping_code",  # 'K-029451'. 키
        "dl_name",  # 약 이름
        "dl_company",  # 회사
    ],
    "외형": [  # 모델이 사진에서 실제로 보는 정보
        "drug_shape",  # 원형/타원형/장방형/육각형/팔각형
        "color_class1",  # 주 색상
        "color_class2",  # 두 번째 색 (캡슐 몸통/뚜껑 등). 빈 값 = 단색
        "form_code_name",  # 필름코팅정/나정/연질캡슐제 등 세부 제형
        "dl_custom_shape",  # '정제, 저작정' / '경질캡슐제' 등 큰 분류 → dosage 파생에 씀
        "print_front",  # 앞면 각인 글자. 빈 값 = 각인 없음
        "print_back",  # 뒷면 각인 글자. 빈 값 = 각인 없음
        "line_front",  # 앞면 분할선 ('-' 일자, '+' 십자). 빈 값 = 없음
        "line_back",  # 뒷면 분할선
        "leng_long",  # 실제 긴 변 (mm)
        "leng_short",  # 실제 짧은 변 (mm)
        "thick",  # 두께 (mm)
    ],
    "의약정보": [  # 사진으론 안 보이지만 설명·보고서용
        "di_etc_otc_code",  # 전문의약품/일반의약품
        "di_class_no",  # '[01140]해열.진통.소염제' 형태 약효 분류
        "dl_material",  # 성분명
    ],
    "촬영조건": [  # 이미지 단위 값 (한 이미지 안의 약들은 전부 같음)
        "drug_dir",  # 앞면/뒷면
        "camera_la",  # 카메라 위도(각도) 70/75/90
        "back_color",  # 배경색 (train 은 '연회색 배경' 1종)
        "light_color",  # 조명색 (train 은 '주백색' 1종)
    ],
}

# 빈 문자열("")이 결측이 아니라 '해당 없음'을 뜻하는 필드 → 결측률 계산에서 제외
EMPTY_MEANS_NONE = {
    "color_class2": "단색",
    "print_front": "각인 없음",
    "print_back": "각인 없음",
    "line_front": "분할선 없음",
    "line_back": "분할선 없음",
}

# 숫자인데 0 이 '값 없음'을 뜻하는 필드 (가바토파정 100mg 은 크기가 전부 0.0 으로 들어있음)
ZERO_MEANS_MISSING = ["leng_long", "leng_short", "thick"]

# 파일명 꼬리 '_0_2_1_2_75_000_200' 을 7개 숫자로 쪼개는 정규식
#   (train json 763개로 확인한 의미)
#   3번째 숫자: 0=앞면, 1=뒷면 → drug_dir 와 100% 일치
#   5번째 숫자: 70/75/90       → camera_la 와 100% 일치
#   나머지(0, 2, 2, 000, 200)는 전부 상수라 의미 확인 불가
_SUFFIX_RE = re.compile(r"_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)$")


def _meta_fields() -> list[str]:
    """META_FIELDS 를 한 줄 리스트로 펼침."""
    return [c for cols in META_FIELDS.values() for c in cols]


def _iter_json(ann_dir):
    """조합폴더/약폴더/파일.json 구조의 json 경로를 정렬해서 돌려줌.

    맥에서 압축하면 생기는 '._파일명.json'(AppleDouble, 실제론 json 아님)은 건너뜀.
    """
    for path in sorted(resolve_path(ann_dir).glob("*/*/*.json")):
        if not path.name.startswith("._"):
            yield path


def load_pill_meta(ann_dir) -> pd.DataFrame:
    """json 1개 = 박스 1개 = 1행 표를 만듦 (pill_df).

    - 컬럼: 파일·박스 정보 + META_FIELDS 원본 값 + 파생 컬럼(add_derived_columns)
    - 빈 문자열은 원본 컬럼에 그대로 둠 → missing_value_report 에서 결측/해당없음 구분용
    - json 에 키 자체가 없으면 NaN 이 들어감 (진짜 결측)
    - images[] 가 비어 있는 json 은 행을 만들 수 없어서 건너뜀
      (몇 개 건너뛰었는지는 annotation_consistency 의 'images 1개' 항목에서 보임)
    """
    rows = []
    for path in _iter_json(ann_dir):
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
        if not j.get("images"):
            continue
        img = j["images"][0]  # 이 데이터는 json 1개에 이미지 1개
        ann = j["annotations"][0] if j.get("annotations") else {}  # 박스도 1개
        x, y, w, h = ann.get("bbox", [None] * 4)  # COCO 형식 [x, y, 너비, 높이]
        row = {
            "file_name": img.get("file_name"),
            "image_id": img.get("id"),
            "combo": path.parent.parent.name.removesuffix("_json"),  # 'K-001900-016548-...'
            "pill_folder": path.parent.name,  # 'K-029451'
            "category_id": ann.get("category_id"),
            "x": x,
            "y": y,
            "box_w": w,
            "box_h": h,
        }
        for c in _meta_fields():
            row[c] = img.get(c)  # 키가 없으면 None → DataFrame 에서 NaN
        rows.append(row)
    df = pd.DataFrame(rows)
    return add_derived_columns(df)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """분석용 파생 컬럼 추가. 원본 컬럼은 안 건드림.

    추가 컬럼:
    - code        : 'K-029451'. dl_mapping_code 가 없으면 약 폴더명으로 대신함
    - category_id : 박스가 없는 json 은 비어 있으므로 code 숫자로 채움 (약 단위 집계가 갈라지지 않게)
    - class_name  : di_class_no 에서 '[01140]' 번호 떼고 이름만
    - dosage      : 정제 / 캡슐 (dl_custom_shape 에 '캡슐'이 들어가면 캡슐)
    - look        : '색·모양' (예: '하양·원형'). 닮은꼴 그룹. 값 없으면 '미상'
    - print_side  : 각인이 있는 면 수 (각인 없음 / 한 면 / 양면 / 미상)
    - box_long_px : bbox 긴 변 (px)
    - px_per_mm   : bbox 긴 변 / 실제 긴 변. 크기 0(결측)이면 NaN
    - fn_side, fn_camera_la : 파일명에서 다시 읽은 앞/뒷면, 카메라 각도 (json 값 검증용)
    """
    df = df.copy()
    df["code"] = df["dl_mapping_code"].fillna(df["pill_folder"])
    code_num = pd.to_numeric(df["code"].astype(str).str.removeprefix("K-"), errors="coerce")
    df["category_id"] = df["category_id"].fillna(code_num).astype("Int64")
    df["class_name"] = (
        df["di_class_no"].astype("string").str.replace(r"^\[\d+\]", "", regex=True)
    )
    df["dosage"] = df["dl_custom_shape"].map(
        lambda s: ("캡슐" if "캡슐" in s else "정제") if isinstance(s, str) and s.strip() else None
    )
    # 빈 문자열도 '미상' 으로 (안 그러면 '하양·' 처럼 반쪽 그룹이 생김)
    # (replace("", None) 은 pandas 버전에 따라 앞 값으로 채워버려서 mask 를 씀)
    shape = df["drug_shape"].mask(df["drug_shape"].eq("")).fillna("미상").astype(str)
    color = df["color_class1"].mask(df["color_class1"].eq("")).fillna("미상").astype(str)
    df["look"] = color + "·" + shape

    # 각인: 앞/뒤 각각 글자가 있으면 1, 합계 0/1/2 → 이름 붙임
    has_print = {
        side: df[side].fillna("").astype(str).str.strip().ne("").astype(int)
        for side in ["print_front", "print_back"]
    }
    df["print_side"] = (has_print["print_front"] + has_print["print_back"]).map(
        {0: "각인 없음", 1: "한 면", 2: "양면"}
    )
    df.loc[df["print_front"].isna(), "print_side"] = "미상"  # 필드 자체가 없으면 판단 불가

    df["box_long_px"] = df[["box_w", "box_h"]].max(axis=1)
    leng = pd.to_numeric(df["leng_long"], errors="coerce").replace(0, float("nan"))
    df["px_per_mm"] = df["box_long_px"] / leng  # 0 으로 나누면 inf 가 되니 위에서 NaN 처리

    suf = df["file_name"].astype("string").str.removesuffix(".png").str.extract(_SUFFIX_RE)
    df["fn_side"] = suf[2].map({"0": "앞면", "1": "뒷면"})
    df["fn_camera_la"] = pd.to_numeric(suf[4], errors="coerce")
    return df


def drug_table(pill_df: pd.DataFrame) -> pd.DataFrame:
    """약(K-코드) 1개 = 1행 표 (drug_df). 박스 수 많은 순.

    - 메타 값은 약마다 json 끼리 같으므로 첫 값('first')을 씀
      (first 는 NaN 을 건너뛰므로 일부 json 만 값이 있어도 채워짐)
    - n_boxes : 이 약의 박스(json) 수
    - n_images: 이 약이 찍힌 이미지 수 (보통 n_boxes 와 같음. 한 사진에 같은 약 1개씩)
    - n_back  : 뒷면으로 찍힌 박스 수
    - px_per_mm: 이 약 박스들의 mm당 픽셀 중앙값
    """
    first = [
        "dl_name", "dl_company", "dosage", "form_code_name", "drug_shape",
        "color_class1", "color_class2", "look", "print_side", "print_front", "print_back",
        "leng_long", "leng_short", "thick", "di_etc_otc_code", "class_name",
    ]  # fmt: skip
    out = (
        pill_df.groupby("code")  # 키는 code 하나만. category_id 는 code 와 1:1 이라 first 로 가져옴
        .agg(
            category_id=("category_id", "first"),
            **{c: (c, "first") for c in first},
            n_boxes=("file_name", "size"),
            n_images=("file_name", "nunique"),
            n_back=("drug_dir", lambda s: int((s == "뒷면").sum())),
            px_per_mm=("px_per_mm", "median"),
        )
        .reset_index()
        .sort_values("n_boxes", ascending=False)
    )
    return out.reset_index(drop=True)


def merge_class_meta(class_df: pd.DataFrame, drug_df: pd.DataFrame) -> pd.DataFrame:
    """기존 class_distribution() 결과에 약 메타를 category_id 로 붙임 (1:1).

    - validate="1:1": 한쪽에 같은 category_id 가 2번 나오면 바로 에러 → 키 중복 자동 검사
    - has_meta  : 메타가 붙었는지 (False 면 json 에 그 클래스 메타가 없음)
    - name_match: class_df 의 name 과 json 의 dl_name 이 같은지 (False 면 클래스맵 이름 확인)
    """
    cols = ["category_id", "code", "dl_name", "dl_company", "dosage", "look",
            "print_side", "di_etc_otc_code", "class_name", "leng_long", "n_back"]  # fmt: skip
    m = class_df.merge(drug_df[cols], on="category_id", how="left", validate="1:1", indicator=True)
    m["name_match"] = m["name"] == m["dl_name"]
    m["has_meta"] = m.pop("_merge") == "both"
    return m


def missing_value_report(pill_df: pd.DataFrame) -> pd.DataFrame:
    """필드별 결측 집계 (박스 = json 단위).

    결측 종류를 나눠서 셈:
    - n_nan  : json 에 필드 자체가 없음 → 진짜 결측
    - n_zero : 크기 필드(ZERO_MEANS_MISSING)의 0 값 → 결측으로 봄
    - n_empty: 빈 문자열 "" (공백만 있어도 포함)
               EMPTY_MEANS_NONE 필드면 '해당 없음' 의미라 결측에서 뺌, 나머지 필드는 결측
    - missing_%    : 위 기준으로 결측인 비율
    - empty_means  : 이 필드에서 빈 값이 무슨 뜻인지
    - drugs_missing: 결측이 나온 약 이름 (특정 약에 몰려 있는지 보는 용도)
    """
    rows = []
    for group, cols in META_FIELDS.items():
        for c in cols:
            s = pill_df[c]
            nan = s.isna()
            empty = s.astype("string").str.strip().eq("").fillna(False) & ~nan
            if c in ZERO_MEANS_MISSING:
                zero = pd.to_numeric(s, errors="coerce").eq(0)
            else:
                zero = pd.Series(False, index=s.index)
            bad = nan | zero
            if c not in EMPTY_MEANS_NONE:
                bad = bad | empty
            rows.append(
                {
                    "group": group,
                    "field": c,
                    "n_nan": int(nan.sum()),
                    "n_zero": int(zero.sum()),
                    "n_empty": int(empty.sum()),
                    "missing_%": round(bad.mean() * 100, 1),
                    "empty_means": EMPTY_MEANS_NONE.get(c, "결측"),
                    "n_unique": s.nunique(),
                    "drugs_missing": ", ".join(sorted(pill_df.loc[bad, "dl_name"].dropna().unique())),
                }
            )
    return pd.DataFrame(rows)


def meta_counts(pill_df: pd.DataFrame, col: str) -> pd.DataFrame:
    """메타 값별 개수. 박스 수 많은 순.

    - n_drugs: 그 값을 가진 약 종류 수 (예: '원형' 약이 몇 종)
    - n_boxes: 그 값을 가진 박스 수 (데이터 양. 불균형 확인용)
    - box_share_%: 전체 박스 중 비율
    - NaN 은 '미상', 빈 문자열은 '(빈 값)' 으로 바꿔서 한 칸으로 보여줌
    """
    s = pill_df[col].fillna("미상").astype(str)
    s = s.where(s.str.strip().ne(""), "(빈 값)")
    out = (
        pill_df.assign(**{col: s})
        .groupby(col)
        .agg(n_drugs=("code", "nunique"), n_boxes=("code", "size"))
        .reset_index()
        .sort_values("n_boxes", ascending=False)
    )
    out["box_share_%"] = (out["n_boxes"] / out["n_boxes"].sum() * 100).round(1)
    return out.reset_index(drop=True)


def look_alike_groups(drug_df: pd.DataFrame, min_drugs: int = 2) -> pd.DataFrame:
    """색·모양(look)이 같은 약끼리 묶음. min_drugs 종 이상인 그룹만.

    한 그룹에 약이 여러 개면 색·모양으론 구분 불가 → 각인·크기로만 구분해야 함.
    → 혼동 행렬에서 이 그룹끼리 헷갈리는지 확인할 후보 목록으로 씀.
    """
    out = (
        drug_df.groupby("look")
        .agg(
            n_drugs=("code", "size"),
            n_boxes=("n_boxes", "sum"),
            drugs=("dl_name", lambda s: ", ".join(s.astype(str))),
        )
        .reset_index()
        .query("n_drugs >= @min_drugs")
        .sort_values(["n_drugs", "n_boxes"], ascending=False)
    )
    return out.reset_index(drop=True)


def shooting_conditions(pill_df: pd.DataFrame) -> pd.DataFrame:
    """이미지 단위 촬영 조건 교차표: 카메라 각도(행) x 앞/뒷면(열) 이미지 수.

    촬영 조건은 이미지마다 1개라서 file_name 중복을 지우고 셈 (박스 수 아님).
    """
    img = pill_df.drop_duplicates("file_name")
    return pd.crosstab(img["camera_la"], img["drug_dir"], margins=True, margins_name="합계")


def annotation_consistency(ann_dir, image_dir=None, ext: str = ".png") -> pd.DataFrame:
    """json·폴더·파일명·이미지가 서로 맞게 연결됐는지 항목별로 셈.

    결과 1행 = 검사 항목 1개
    - n_checked: 검사한 개수 (마지막 항목만 이미지 수, 나머지는 json 수)
    - n_fail   : 실패 개수. 0 이면 정상
    - detail   : 실패한 json 경로 예시 (최대 3개)

    image_dir 를 주면 '이미지 파일 존재' 항목도 검사함.
    반대 방향(이미지는 있는데 json 이 없음)은 image_json_pairs / missing_jsons 로 봄.
    """
    ann_dir = resolve_path(ann_dir)
    image_names = None
    if image_dir is not None:
        image_names = {p.name for p in resolve_path(image_dir).rglob(f"*{ext}")}

    checks: dict[str, list[str]] = {}  # 항목 이름 → 실패한 json 경로 목록
    total = 0
    image_ids: dict[str, set] = {}  # 이미지 파일명 → 그 이미지를 가리키는 images.id 들

    def record(name, ok, path):
        checks.setdefault(name, [])  # 실패가 0개여도 항목이 표에 나오게 먼저 만듦
        if not ok:
            checks[name].append(str(path.relative_to(ann_dir)))

    for path in _iter_json(ann_dir):
        total += 1
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
        # 1) json 구조: 이 데이터는 json 1개에 이미지·박스·카테고리 각 1개여야 함
        record("images 1개", len(j.get("images", [])) == 1, path)
        record("annotations 1개", len(j.get("annotations", [])) == 1, path)
        record("categories 1개", len(j.get("categories", [])) == 1, path)
        if not j.get("images") or not j.get("annotations"):
            continue  # 비교할 대상이 없으면 아래 검사 생략

        img, ann = j["images"][0], j["annotations"][0]
        cat = j["categories"][0] if j.get("categories") else {}
        code = path.parent.name  # 약 폴더 'K-029451'
        combo = path.parent.parent.name.removesuffix("_json")  # 'K-001900-016548-019607-029451'
        num = code.removeprefix("K-")
        w, h = img.get("width", 0), img.get("height", 0)
        x, y, bw, bh = ann.get("bbox", [0, 0, 0, 0])
        tests = {
            # 2) 파일명 ↔ json 내용
            "json 파일명 == file_name": f"{path.stem}{ext}" == img.get("file_name"),
            "file_name이 조합폴더로 시작": str(img.get("file_name", "")).startswith(combo + "_"),
            # 3) 폴더 ↔ 약 코드
            "약폴더 코드가 조합에 포함": num in combo.split("-")[1:],
            "약폴더 == dl_mapping_code": img.get("dl_mapping_code") == code,
            "category_id == 코드 숫자": num.isdigit() and ann.get("category_id") == int(num),
            # 4) json 내부 연결
            "categories.id == category_id": cat.get("id") == ann.get("category_id"),
            "categories.name == dl_name": cat.get("name") == img.get("dl_name"),
            "annotation.image_id == images.id": ann.get("image_id") == img.get("id"),
            # 5) 박스 좌표
            "bbox 이미지 안": x >= 0 and y >= 0 and x + bw <= w and y + bh <= h,
            "bbox 크기 > 0": bw > 0 and bh > 0,
        }
        if image_names is not None:
            # 6) 실제 이미지 파일
            tests["이미지 파일 존재"] = img.get("file_name") in image_names
        for name, ok in tests.items():
            record(name, ok, path)
        image_ids.setdefault(img.get("file_name"), set()).add(img.get("id"))

    # 7) 같은 이미지를 가리키는 json 들(약마다 1개씩)이 같은 images.id 를 쓰는지
    last = "같은 이미지면 images.id 동일"
    checks[last] = [f for f, ids in image_ids.items() if len(ids) > 1]
    return pd.DataFrame(
        [
            {
                "check": k,
                "n_checked": len(image_ids) if k == last else total,
                "n_fail": len(v),
                "detail": ", ".join(v[:3]),
            }
            for k, v in checks.items()
        ]
    )


def size_consistency(pill_df: pd.DataFrame, tol: float = 0.25) -> pd.DataFrame:
    """실제 크기(leng_long, mm)와 bbox 긴 변(px)이 맞는지 봄.

    - 같은 카메라 각도면 거리가 같으니 mm당 픽셀(px_per_mm)이 비슷해야 정상
    - 각도별 중앙값 대비 비율(ratio_to_median)이 1±tol 밖이면 suspect=True
      → bbox 를 잘못 그렸거나 크기 정보가 틀렸을 가능성
    - 알약이 비스듬히 놓이면 bbox 긴 변이 실제보다 짧게 잡히는 건 정상이라
      suspect 는 '확인해볼 후보'일 뿐 오류 확정 아님
    - 크기 0/결측 약(가바토파정)은 px_per_mm 가 NaN 이라 제외됨
    """
    df = pill_df.dropna(subset=["px_per_mm"]).copy()
    med = df.groupby("camera_la")["px_per_mm"].transform("median")
    df["ratio_to_median"] = (df["px_per_mm"] / med).round(3)
    df["suspect"] = (df["ratio_to_median"] - 1).abs() > tol
    cols = ["file_name", "code", "dl_name", "camera_la", "leng_long", "box_w", "box_h",
            "px_per_mm", "ratio_to_median", "suspect"]  # fmt: skip
    return df[cols].sort_values("ratio_to_median").reset_index(drop=True)


# =====================================================================
# 노트북에서 쓰는 법 (자세한 건 02_eda_meta.ipynb)
# =====================================================================
# from src.eda import (load_pill_meta, drug_table, merge_class_meta, missing_value_report,
#                      meta_counts, look_alike_groups, shooting_conditions,
#                      annotation_consistency, size_consistency)
#
# pill_df = load_pill_meta(d["train_annotations"])       # 박스 1개 = 1행 (763행)
# drug_df = drug_table(pill_df)                          # 약 1개 = 1행 (56행)
# merged  = merge_class_meta(class_df, drug_df)          # 기존 class_df + 이름·회사·외형
# missing_value_report(pill_df)                          # 필드별 결측
# annotation_consistency(d["train_annotations"], d["train_images"])   # 연결 상태
