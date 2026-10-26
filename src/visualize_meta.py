# =====================================================================
# src/visualize.py 맨 아래에 붙여넣기 (json 메타데이터 시각화)
# =====================================================================
# - 한글 폰트는 노트북 첫 셀의 set_korean_font() 를 그대로 씀
# - 아래 import 는 visualize.py 맨 위에 이미 있으면 지워도 됨
# - 모든 함수는 그림만 그리고 아무것도 안 돌려줌 (표는 src/eda.py 함수가 담당)
# - 색 규칙: 한 가지 값만 보여주는 막대는 전부 파랑 1색
#            여러 종류를 구분할 때만 파랑 → 주황 → 청록 순서로 씀
# =====================================================================

import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # 계열 색 (이 순서로 씀)
# 구분할 종류가 3개보다 많을 때 이어서 쓸 색 (노랑, 분홍, 초록, 보라, 빨강)
SERIES = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED = "#0b0b0b", "#52514e"  # 제목 글자색, 눈금·숫자 글자색
GRID = "#e5e4e0"  # 옅은 격자선


def _clean_axes(ax, axis="x"):
    """위·오른쪽 테두리 지우고 옅은 격자선만 남김 (막대가 잘 보이게)."""
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)  # 격자선을 막대 뒤로
    ax.tick_params(colors=MUTED)


def _barh_with_values(ax, labels, values, fontsize=9):
    """가로 막대 + 막대 끝에 숫자. 위에서부터 큰 값이 오도록 호출 전에 뒤집어서 넘김."""
    ax.barh(labels, values, color=BLUE, height=0.7)
    for yi, v in enumerate(values):
        ax.text(v, yi, f" {v}", va="center", fontsize=fontsize, color=MUTED)


def plot_meta_bar(
    counts_df: pd.DataFrame,
    col: str,
    value: str = "n_boxes",
    title: str | None = None,
    top: int | None = None,
):
    """meta_counts() 결과를 가로 막대로 그림.

    - col  : 값 이름이 들어있는 컬럼 (meta_counts 에 넘긴 컬럼명 그대로)
    - value: 'n_boxes' = 박스 수(데이터 양), 'n_drugs' = 약 종류 수
    - top  : 상위 몇 개만 볼지 (회사처럼 값이 많을 때)
    """
    df = counts_df.sort_values(value, ascending=False)
    if top:
        df = df.head(top)
    df = df.iloc[::-1]  # barh 는 아래부터 그리므로 뒤집어야 큰 값이 위로 감
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(df) + 1))  # 막대 수에 맞춰 높이 조절
    _barh_with_values(ax, df[col].astype(str), df[value])
    unit = "박스 수" if value == "n_boxes" else "약 종류 수"
    ax.set_title(title or f"{col}별 {unit}", loc="left", color=INK)
    _clean_axes(ax)
    plt.tight_layout()
    plt.show()


def plot_drug_bars(merged_df: pd.DataFrame, value: str = "n_boxes", top: int | None = None):
    """약(클래스)별 박스 수. 라벨 = '이름 · 회사  (K-코드)'.

    merge_class_meta() 결과나 drug_table() 결과 둘 다 넣을 수 있음
    (dl_name, dl_company, code 컬럼만 있으면 됨).
    """
    df = merged_df.sort_values(value, ascending=False)
    if top:
        df = df.head(top)
    df = df.iloc[::-1]
    labels = (
        df["dl_name"].fillna("이름 없음") + " · " + df["dl_company"].fillna("회사 없음")
        + "  (" + df["code"].astype(str) + ")"
    )  # fmt: skip
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(df) + 1))
    _barh_with_values(ax, labels, df[value], fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.margins(y=0.01)  # 위아래 빈 공간 줄임
    unit = "박스 수" if value == "n_boxes" else value
    ax.set_title(f"약(클래스)별 {unit}", loc="left", color=INK)
    _clean_axes(ax)
    plt.tight_layout()
    plt.show()


def plot_missing(report: pd.DataFrame):
    """missing_value_report() 결과를 누적 가로 막대로 그림.

    - 주황 칠      : NaN (json 에 필드 자체가 없음, 진짜 결측)
    - 주황 빗금    : 0 (크기 정보 없음, 결측)
    - 파랑         : 빈 문자열 ""
    - 라벨의 [ ]   : 그 필드에서 빈 값의 뜻. [결측] 이 아니면 '해당 없음' 이라 문제 아님
    - 결측이 하나도 없는 필드는 그래프에서 빠짐 (표에는 다 있음)
    """
    df = report[(report["n_nan"] + report["n_empty"] + report["n_zero"]) > 0].iloc[::-1]
    if df.empty:
        print("결측·빈 값이 있는 필드 없음")
        return
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(df) + 1.2))
    labels = df["field"] + "  [" + df["empty_means"] + "]"
    # 막대 3개를 left 로 이어 붙여서 누적 막대로 만듦
    ax.barh(labels, df["n_nan"], color=ORANGE, height=0.7, label="NaN (필드 없음)")
    ax.barh(labels, df["n_zero"], left=df["n_nan"], color=ORANGE, height=0.7, hatch="///",
            edgecolor="white", linewidth=1.5, label="0 (크기 정보 없음)")  # fmt: skip
    ax.barh(labels, df["n_empty"], left=df["n_nan"] + df["n_zero"], color=BLUE, height=0.7,
            edgecolor="white", linewidth=1.5, label='빈 문자열 ""')  # fmt: skip
    ax.set_xlabel("박스(json) 수")
    ax.set_title("필드별 결측 · [ ] 안은 빈 값의 의미", loc="left", color=INK)
    ax.legend(loc="lower right", frameon=False)
    _clean_axes(ax)
    plt.tight_layout()
    plt.show()


def plot_look_heatmap(drug_df: pd.DataFrame):
    """색(행) x 모양(열)별 약 종류 수 히트맵.

    칸 숫자가 클수록 색·모양이 같은 약이 많음 = 생김새로 구분하기 어려움.
    값이 없는 약(가바토파정)은 '미상' 행·열로 따로 보임.
    """
    ct = pd.crosstab(drug_df["color_class1"].fillna("미상"), drug_df["drug_shape"].fillna("미상"))
    fig, ax = plt.subplots(figsize=(1.1 * ct.shape[1] + 2, 0.5 * ct.shape[0] + 1.5))
    im = ax.imshow(ct.values, cmap="Blues", vmin=0)  # 한 색 계열 진하기 = 개수
    ax.set_xticks(range(ct.shape[1]), ct.columns)
    ax.set_yticks(range(ct.shape[0]), ct.index)
    vmax = ct.values.max()
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.values[i, j]
            if v:  # 0 칸은 숫자 생략
                ax.text(j, i, v, ha="center", va="center", fontsize=10,
                        color="white" if v > vmax / 2 else INK)  # 진한 칸은 흰 글자  # fmt: skip
    ax.set_title("색 x 모양별 약 종류 수", loc="left", color=INK)
    fig.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.show()


def plot_size_scatter(pill_df: pd.DataFrame):
    """실제 긴 변(mm, x축) vs bbox 긴 변(px, y축) 산점도. 카메라 각도별 색 구분.

    점들이 한 직선 위에 모이면 정상 (크기가 클수록 박스도 큼).
    직선에서 크게 벗어난 점 = bbox 나 크기 정보 오류 후보 → size_consistency() 표로 확인.
    크기 0(결측)인 약은 x=0 에 찍혀서 오해를 주니 뺌.
    """
    pill_df = pill_df[pd.to_numeric(pill_df["leng_long"], errors="coerce") > 0]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    # 각도 종류가 3개보다 많아도 안 빠지게 SERIES 에서 순서대로 색을 가져옴
    # (zip 으로 색 3개와 묶으면 4번째 각도부터 조용히 사라지는 문제가 있었음)
    for i, la in enumerate(sorted(pill_df["camera_la"].dropna().unique())):
        color = SERIES[i % len(SERIES)]
        g = pill_df[pill_df["camera_la"] == la]
        ax.scatter(g["leng_long"], g["box_long_px"], s=22, color=color, alpha=0.75,
                   edgecolor="white", linewidth=0.6, label=f"camera_la {la}")  # fmt: skip
    ax.set_xlabel("실제 긴 변 leng_long (mm)")
    ax.set_ylabel("bbox 긴 변 (px)")
    ax.set_title("실제 크기 vs 박스 크기", loc="left", color=INK)
    ax.legend(frameon=False)
    _clean_axes(ax, axis="both")
    plt.tight_layout()
    plt.show()
