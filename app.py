# app.py
import os
import math
import sqlite3
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import pandas as pd
import streamlit as st

# =========================
# Streamlit config (MUST be first Streamlit command)
# =========================
st.set_page_config(page_title="성실신고 리스크 & 법인전환 분석", layout="wide")

# =========================
# 기본 설정
# =========================
APP_TITLE = "📊 개인사업자 성실신고 리스크 & 법인전환 전략 분석"

# (1) 엑셀 기본 파일: app.py와 같은 폴더에 두면 자동 인식
DEFAULT_EXCEL_FILENAME = "업종코드-표준산업분류 연계표_기준경비율 코드 작성.xlsx"
DEFAULT_EXCEL_PATH = os.path.join(os.path.dirname(__file__), DEFAULT_EXCEL_FILENAME)

# (2) 로컬 DB (Supabase 없을 때만 사용)
SQLITE_DB_FILE = "users.db"

# (3) 관리자 최초 부트스트랩(배포 시 환경변수/Secrets로 넣는 걸 추천)
# 예) STREAMLIT_SECRETS 또는 OS env로 설정 가능
ADMIN_EMAIL = st.secrets.get("ADMIN_EMAIL", os.getenv("ADMIN_EMAIL", ""))
# 아래 키를 알고 있는 사람만 "관리자 부트스트랩" 버튼을 사용할 수 있음 (선택)
ADMIN_BOOTSTRAP_KEY = st.secrets.get("ADMIN_BOOTSTRAP_KEY", os.getenv("ADMIN_BOOTSTRAP_KEY", ""))

# (4) Supabase (있으면 우선 사용)
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))

USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_KEY)
supabase = None
if USE_SUPABASE:
    try:
        from supabase import create_client  # type: ignore
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        USE_SUPABASE = False
        supabase = None


# =========================
# 공통 유틸
# =========================
def money(n: float) -> str:
    try:
        n = int(round(float(n)))
    except Exception:
        return "-"
    return f"{n:,}원"


def pct(n: float, digits=1) -> str:
    try:
        return f"{float(n):.{digits}f}%"
    except Exception:
        return "-"


@dataclass
class IncomeRateResult:
    industry_code: int
    biz_code: float
    q_value: float
    income_rate: float  # percent


# =========================
# 1) 사용자 DB 레이어 (Supabase 우선, 없으면 SQLite)
# =========================
def sqlite_init():
    conn = sqlite3.connect(SQLITE_DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            approved INTEGER DEFAULT 0,
            is_admin INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def sqlite_get_user(email: str) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(SQLITE_DB_FILE)
    c = conn.cursor()
    c.execute("SELECT email, approved, is_admin FROM users WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {"email": row[0], "approved": bool(row[1]), "is_admin": bool(row[2])}


def sqlite_upsert_user(email: str, approved: Optional[bool] = None, is_admin: Optional[bool] = None):
    conn = sqlite3.connect(SQLITE_DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO users (email, approved, is_admin) VALUES (?, 0, 0)", (email,))
    if approved is not None:
        c.execute("UPDATE users SET approved=? WHERE email=?", (1 if approved else 0, email))
    if is_admin is not None:
        c.execute("UPDATE users SET is_admin=? WHERE email=?", (1 if is_admin else 0, email))
    conn.commit()
    conn.close()


def sqlite_list_users() -> list[Dict[str, Any]]:
    conn = sqlite3.connect(SQLITE_DB_FILE)
    c = conn.cursor()
    rows = c.execute("SELECT email, approved, is_admin FROM users ORDER BY email").fetchall()
    conn.close()
    return [{"email": r[0], "approved": bool(r[1]), "is_admin": bool(r[2])} for r in rows]


def supa_get_user(email: str) -> Optional[Dict[str, Any]]:
    assert supabase is not None
    resp = supabase.table("users").select("*").eq("email", email).execute()
    data = resp.data or []
    return data[0] if data else None


def supa_upsert_user(email: str, approved: Optional[bool] = None, is_admin: Optional[bool] = None):
    assert supabase is not None
    existing = supa_get_user(email)
    if not existing:
        payload = {"email": email, "approved": bool(approved) if approved is not None else False,
                   "is_admin": bool(is_admin) if is_admin is not None else False}
        supabase.table("users").insert(payload).execute()
        return
    payload = {}
    if approved is not None:
        payload["approved"] = bool(approved)
    if is_admin is not None:
        payload["is_admin"] = bool(is_admin)
    if payload:
        supabase.table("users").update(payload).eq("email", email).execute()


def supa_list_users() -> list[Dict[str, Any]]:
    assert supabase is not None
    resp = supabase.table("users").select("*").order("email").execute()
    return resp.data or []


def db_init():
    if USE_SUPABASE:
        # Supabase는 테이블이 이미 있어야 합니다. (SQL: users 테이블 생성)
        return
    sqlite_init()


def db_get_user(email: str) -> Optional[Dict[str, Any]]:
    if USE_SUPABASE:
        return supa_get_user(email)
    return sqlite_get_user(email)


def db_upsert_user(email: str, approved: Optional[bool] = None, is_admin: Optional[bool] = None):
    if USE_SUPABASE:
        return supa_upsert_user(email, approved=approved, is_admin=is_admin)
    return sqlite_upsert_user(email, approved=approved, is_admin=is_admin)


def db_list_users() -> list[Dict[str, Any]]:
    if USE_SUPABASE:
        return supa_list_users()
    return sqlite_list_users()


db_init()


# =========================
# 2) 로그인/승인 게이트
# =========================
def normalize_email(email: str) -> str:
    return email.strip().lower()


def login_and_gate() -> Dict[str, Any]:
    """
    - 이메일 입력 → 사용자 레코드 없으면 자동 생성(approved=False)
    - approved True 일 때만 앱 사용 가능
    - 반환: current_user dict (email, approved, is_admin)
    """
    st.sidebar.markdown("### 🔐 접근 제어")
    st.sidebar.caption("승인된 사용자만 이용 가능합니다.")

    if "email" not in st.session_state:
        st.session_state.email = ""

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        email = st.sidebar.text_input("이메일", value=st.session_state.email, placeholder="name@example.com")
        if st.sidebar.button("로그인", type="primary"):
            email = normalize_email(email)
            if "@" not in email or "." not in email:
                st.sidebar.error("올바른 이메일 형식이 아닙니다.")
                st.stop()

            st.session_state.email = email

            user = db_get_user(email)
            if not user:
                db_upsert_user(email, approved=False, is_admin=False)
                st.sidebar.warning("등록되었습니다. 관리자 승인 대기 중입니다.")
                st.stop()

            # 승인 전이면 차단
            if not bool(user.get("approved", False)):
                st.sidebar.warning("관리자 승인 대기 중입니다.")
                st.stop()

            st.session_state.logged_in = True
            st.rerun()

        st.stop()

    # 로그인 상태면 사용자 로드
    email = normalize_email(st.session_state.email)
    user = db_get_user(email)
    if not user:
        # 아주 예외적인 경우
        st.sidebar.error("사용자 정보를 찾을 수 없습니다. 다시 로그인해주세요.")
        st.session_state.logged_in = False
        st.rerun()

    # 승인 해제되면 즉시 차단
    if not bool(user.get("approved", False)):
        st.sidebar.warning("승인이 해제되었습니다. 관리자에게 문의하세요.")
        st.session_state.logged_in = False
        st.stop()

    st.sidebar.success(f"접속: {email}")
    return user


def admin_bootstrap_ui():
    """
    관리자 이메일을 환경변수 ADMIN_EMAIL로 지정한 경우,
    최초에 관리자 계정을 승인+관리자로 만들어주는 UI.
    (보안을 위해 ADMIN_BOOTSTRAP_KEY를 설정해두면 키 입력이 있어야 실행됩니다.)
    """
    if not BOOTSTRAP_ADMIN_EMAIL:
        return

    with st.sidebar.expander("🛠 관리자 초기설정(최초 1회)"):
        st.caption("최초에 관리자 계정을 승인+관리자로 설정합니다.")
        st.code(f"ADMIN_EMAIL = {BOOTSTRAP_ADMIN_EMAIL}", language="text")

        if BOOTSTRAP_ADMIN_KEY:
            key = st.text_input("부트스트랩 키", type="password", help="환경변수 ADMIN_BOOTSTRAP_KEY")
            ok = st.button("관리자 계정 생성/갱신")
            if ok:
                if key != BOOTSTRAP_ADMIN_KEY:
                    st.error("키가 틀렸습니다.")
                else:
                    db_upsert_user(BOOTSTRAP_ADMIN_EMAIL, approved=True, is_admin=True)
                    st.success("관리자 계정을 승인+관리자로 설정했습니다.")
        else:
            if st.button("관리자 계정 생성/갱신"):
                db_upsert_user(BOOTSTRAP_ADMIN_EMAIL, approved=True, is_admin=True)
                st.success("관리자 계정을 승인+관리자로 설정했습니다.")


admin_bootstrap_ui()
current_user = login_and_gate()


# =========================
# 3) 엑셀 기반 소득율 계산
# =========================
def load_mapping_excel(uploaded_file) -> pd.DataFrame:
    # 업로드 우선
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file)

    # 기본 파일이 app 폴더에 있으면 자동 사용
    if os.path.exists(DEFAULT_EXCEL_PATH):
        return pd.read_excel(DEFAULT_EXCEL_PATH)

    raise FileNotFoundError(
        f"기본 엑셀 파일을 찾지 못했습니다.\n"
        f"- 앱 폴더에 '{DEFAULT_EXCEL_FILENAME}' 파일을 넣거나\n"
        f"- 왼쪽에서 엑셀을 업로드해 주세요."
    )


def calc_income_rate(df: pd.DataFrame, industry_code: int) -> IncomeRateResult:
    """
    - F열에서 산업분류코드 찾기
    - 해당 행의 C열 값을 ‘업종코드’
    - K열에서 업종코드 찾기
    - 해당 행의 Q열 값을 ‘Q값’
    - 소득율 = 100 - Q값
    """
    # A=0 기준 (F=5, C=2, K=10, Q=16)
    row_f = df[df.iloc[:, 5] == industry_code]
    if row_f.empty:
        raise ValueError("F열에서 산업분류코드를 찾지 못했습니다. (산업분류코드 불일치)")

    biz_code = float(row_f.iloc[0, 2])

    row_k = df[df.iloc[:, 10] == biz_code]
    if row_k.empty:
        raise ValueError("K열에서 업종코드를 찾지 못했습니다. (업종코드 매칭 실패)")

    q_value = float(row_k.iloc[0, 16])
    income_rate = 100.0 - q_value

    return IncomeRateResult(industry_code=industry_code, biz_code=biz_code, q_value=q_value, income_rate=income_rate)


# =========================
# 4) 세금/리스크 계산
# =========================
def korean_progressive_income_tax(tax_base: float) -> float:
    """
    종합소득세(국세) 누진세율 계산(단순화: 과세표준=순이익 가정)
    """
    x = max(0.0, float(tax_base))

    brackets = [
        (14_000_000, 0.06, 0),
        (50_000_000, 0.15, 1_260_000),
        (88_000_000, 0.24, 5_760_000),
        (150_000_000, 0.35, 15_440_000),
        (300_000_000, 0.38, 19_940_000),
        (500_000_000, 0.40, 25_940_000),
        (1_000_000_000, 0.42, 35_940_000),
        (math.inf, 0.45, 65_940_000),
    ]

    for upper, rate, deduction in brackets:
        if x <= upper:
            return x * rate - deduction
    return x * 0.45 - 65_940_000


def local_income_tax(national_tax: float) -> float:
    return max(0.0, float(national_tax)) * 0.10


def faithful_report_risk(category: str, sales: float) -> Tuple[str, str]:
    thresholds = {
        "도소매": 1_500_000_000,
        "제조/건설": 750_000_000,
        "서비스/부동산임대": 500_000_000,
    }
    th = thresholds.get(category, 750_000_000)
    s = float(sales)

    if s < th * 0.8:
        return "낮음", f"기준 {money(th)} 대비 여유 구간"
    elif s < th:
        return "보통", f"기준 {money(th)} 근접 (주의)"
    elif s < th * 1.2:
        return "높음", f"기준 {money(th)} 초과 (대상 가능성 높음)"
    else:
        return "매우 높음", f"기준 {money(th)} 크게 초과 (대상 가능성 매우 높음)"


def conservative_disallow_amounts(sales: float) -> Dict[str, float]:
    s = float(sales)
    return {
        "외주가공비": s * 0.02,
        "가족·특수관계인 인건비": s * 0.01,
        "차량·접대 등 사적경비": s * 0.01,
        "무증빙·현금지출": s * 0.005,
    }


def build_report_md(
    result: IncomeRateResult,
    last_sales: float,
    this_sales: float,
    employees: int,
    category: str,
    insurance_rate: float,
    ceo_salary: float,
    corp_tax_rate: float,
    use_disallow: bool,
    disallow_custom: Optional[Dict[str, float]],
) -> str:
    income_rate = result.income_rate / 100.0

    last_profit = float(last_sales) * income_rate
    this_profit = float(this_sales) * income_rate

    nat_tax = korean_progressive_income_tax(this_profit)
    loc_tax = local_income_tax(nat_tax)
    total_tax = nat_tax + loc_tax

    up_profit = float(this_sales) * ((result.income_rate + 1.0) / 100.0)
    down_profit = float(this_sales) * ((result.income_rate - 1.0) / 100.0)

    up_total_tax = korean_progressive_income_tax(up_profit) + local_income_tax(korean_progressive_income_tax(up_profit))
    down_total_tax = korean_progressive_income_tax(down_profit) + local_income_tax(korean_progressive_income_tax(down_profit))
    delta_up = up_total_tax - total_tax
    delta_down = total_tax - down_total_tax

    risk, reason = faithful_report_risk(category, this_sales)

    disallow = {}
    if use_disallow:
        disallow = disallow_custom if disallow_custom else conservative_disallow_amounts(this_sales)

    rows = []
    total_disallow = 0.0
    add_tax_total = 0.0
    add_ins_total = 0.0

    for k, amt in disallow.items():
        amt = max(0.0, float(amt))
        total_disallow += amt

        add_tax_n = korean_progressive_income_tax(this_profit + amt) - korean_progressive_income_tax(this_profit)
        add_tax_l = local_income_tax(korean_progressive_income_tax(this_profit + amt)) - local_income_tax(korean_progressive_income_tax(this_profit))
        add_tax = max(0.0, add_tax_n + add_tax_l)

        add_ins = amt * float(insurance_rate)

        add_tax_total += add_tax
        add_ins_total += add_ins

        rows.append((k, amt, amt, add_tax, add_ins))

    base_annual = total_tax + (this_profit * float(insurance_rate))
    base_3y = base_annual * 3

    strict_annual = base_annual + add_tax_total + add_ins_total
    strict_3y = strict_annual * 3
    strict_3y_inc = strict_3y - base_3y

    corp_tax_base = max(0.0, this_profit - float(ceo_salary))
    corp_tax = corp_tax_base * float(corp_tax_rate)
    corp_3y = corp_tax * 3

    md = []
    md.append("# 개인사업자 성실신고 리스크 및 법인전환 전략 분석 보고서\n\n")

    md.append("## 1) 소득율 산출 결과\n")
    md.append(f"- 산업분류코드: **{result.industry_code}**\n")
    md.append(f"- 업종코드: **{int(result.biz_code)}**\n")
    md.append(f"- Q값: **{result.q_value}**\n")
    md.append(f"- 계산된 소득율: **{pct(result.income_rate, 1)}**\n\n")

    md.append("## 2) 순이익 추정\n")
    md.append(f"- 작년 매출: {money(last_sales)} → 작년 순이익(추정): **{money(last_profit)}**\n")
    md.append(f"- 금년 예상 매출: {money(this_sales)} → 금년 순이익(추정): **{money(this_profit)}**\n\n")

    md.append("## 3) 종합소득세(추정) + 지방소득세 포함\n")
    md.append("- (단순) 과세표준 ≈ 순이익으로 가정\n")
    md.append(f"- 국세(종합소득세): **{money(nat_tax)}**\n")
    md.append(f"- 지방소득세(국세의 10%): **{money(loc_tax)}**\n")
    md.append(f"- 합계: **{money(total_tax)}**\n\n")
    md.append("### 소득율 민감도(±1%p)\n")
    md.append(f"- 소득율 +1%p 시 세금 증가(추정): **{money(delta_up)}**\n")
    md.append(f"- 소득율 -1%p 시 세금 감소(추정): **{money(delta_down)}**\n\n")

    md.append("## 4) 성실신고확인대상 여부 판단\n")
    md.append(f"- 업종 분류: **{category}**\n")
    md.append(f"- 위험도: **{risk}**\n")
    md.append(f"- 근거: {reason}\n\n")

    md.append("## 5) 성실신고 비용 부인 시뮬레이션\n")
    if not use_disallow:
        md.append("- (설정 OFF)\n\n")
    else:
        if not rows:
            md.append("- 부인 가정 항목이 없습니다.\n\n")
        else:
            md.append("| 항목 | 가정 비용 부인 금액 | 과세소득 증가 | 추가 종합소득세(지방세 포함) | 건보 증가(추정) |\n")
            md.append("|---|---:|---:|---:|---:|\n")
            for (k, amt, inc_tax_base, add_tax, add_ins) in rows:
                md.append(f"| {k} | {money(amt)} | {money(inc_tax_base)} | {money(add_tax)} | {money(add_ins)} |\n")
            md.append("\n")
            md.append(f"- 총 비용 부인 금액: **{money(total_disallow)}**\n")
            md.append(f"- 총 추가 세금(추정): **{money(add_tax_total)}**\n")
            md.append(f"- 총 건보 증가(추정): **{money(add_ins_total)}**\n")
            if total_disallow > 0:
                per_100m = (add_tax_total / total_disallow) * 100_000_000
                md.append(f"\n👉 참고: 비용 1억 정리 시 추가 세금(추정) ≈ **{money(per_100m)}**\n")
            md.append("\n")

    md.append("## 6) 3년 누적 리스크(추정)\n")
    md.append(f"- 개인 유지(현재 구조) 3년: **{money(base_3y)}** (세금+건보)\n")
    if use_disallow:
        md.append(f"- 성실신고 비용 정리 발생 3년: **{money(strict_3y)}**\n")
        md.append(f"- 3년 증가분: **{money(strict_3y_inc)}**\n")
        md.append("- 5년 누적 시에는 증가분이 더 커질 수 있습니다(구조적 누적).\n\n")
    else:
        md.append("\n")

    md.append("## 7) 법인 전환 시 비교(단순 모델)\n")
    md.append(f"- 대표 급여 가정: **{money(ceo_salary)}**\n")
    md.append(f"- 법인 과세표준(단순): max(0, 순이익-급여) = **{money(corp_tax_base)}**\n")
    md.append(f"- 법인세(가정 세율 {corp_tax_rate*100:.1f}%): **{money(corp_tax)}**\n\n")

    md.append("### 3년 누적 비교표(단순)\n")
    md.append("| 구분 | 개인 유지(현재) | 성실신고 정리 후 | 법인 전환 |\n")
    md.append("|---|---:|---:|---:|\n")
    md.append(f"| 3년 합계(세금+건보) | {money(base_3y)} | {money(strict_3y) if use_disallow else '-'} | {money(corp_3y)} |\n\n")

    md.append("## 8) 전략적 결론\n")
    md.append("- **매출 규모가 성실신고 기준에 근접/초과하는 업종**에서는 비용 증빙 리스크가 누적됩니다.\n")
    md.append("- 성실신고 국면에서는 ‘비용 정리’가 곧 ‘과세소득 증가’로 연결되어 세금+건보가 함께 상승하는 구조가 됩니다.\n")
    md.append("- 법인 전환은 **급여/비용 구조 설계로 과세를 분산**할 수 있어 ‘리스크 통제’ 목적에서 의미가 있습니다.\n\n")

    md.append("## 1차 미팅 클로징 멘트(샘플)\n")
    md.append(
        "대표님, 지금은 ‘세금이 많다/적다’가 아니라 **구조적으로 성실신고 리스크 구간**에 들어온 상태입니다. "
        "특히 비용 증빙 이슈가 생기면 3년 누적 금액이 크게 벌어질 수 있어, 이번에 **개인 유지 vs 비용정리 vs 법인전환**을 숫자로 비교해서 "
        "가장 안전한 구조로 설계해보시죠.\n"
    )

    return "".join(md)


# =========================
# 5) UI
# =========================
st.title(APP_TITLE)

st.sidebar.markdown("---")
st.sidebar.caption(f"DB 모드: {'Supabase(배포용)' if USE_SUPABASE else 'SQLite(로컬용)'}")

with st.sidebar:
    st.subheader("1) 데이터 입력")
    uploaded = st.file_uploader("업종코드-표준산업분류 연계표 엑셀 업로드(권장)", type=["xlsx"])
    if os.path.exists(DEFAULT_EXCEL_PATH):
        st.caption(f"기본 파일 자동 인식: {DEFAULT_EXCEL_FILENAME}")
    else:
        st.caption("기본 파일이 없으면 업로드가 필요합니다.")

    industry_code = st.number_input("산업분류코드(F열)", min_value=0, step=1, value=25913)
    last_sales = st.number_input("작년 매출(원)", min_value=0, step=10_000_000, value=800_000_000)
    this_sales = st.number_input("금년 예상 매출(원)", min_value=0, step=10_000_000, value=1_000_000_000)
    employees = st.number_input("직원 수(대표 제외)", min_value=0, step=1, value=6)

    st.divider()
    st.subheader("2) 성실신고 기준(업종 분류)")
    category = st.selectbox("업종 분류 선택", ["제조/건설", "도소매", "서비스/부동산임대"], index=0)

    st.divider()
    st.subheader("3) 건보/법인 가정값")
    insurance_rate = st.slider("건강보험 증가 추정률(과세소득 대비)", 0.0, 0.15, 0.05, 0.005)
    ceo_salary = st.number_input("법인 전환 시 대표 급여 가정(원)", min_value=0, step=1_000_000, value=70_000_000)
    corp_tax_rate = st.slider("법인세(단순 가정 세율)", 0.05, 0.25, 0.09, 0.005)

    st.divider()
    st.subheader("4) 비용 부인 시뮬레이션")
    use_disallow = st.checkbox("성실신고 비용 부인 시뮬레이션 ON", value=True)
    use_custom = st.checkbox("부인 금액 직접 입력(커스텀)", value=False)

    disallow_custom = None
    if use_disallow and use_custom:
        st.caption("금년 매출 기준 ‘가정 비용 부인 금액’을 원 단위로 직접 입력하세요.")
        d1 = st.number_input("외주가공비(원)", min_value=0, step=1_000_000, value=int(this_sales * 0.02))
        d2 = st.number_input("가족·특수관계인 인건비(원)", min_value=0, step=1_000_000, value=int(this_sales * 0.01))
        d3 = st.number_input("차량·접대 등 사적경비(원)", min_value=0, step=1_000_000, value=int(this_sales * 0.01))
        d4 = st.number_input("무증빙·현금지출(원)", min_value=0, step=500_000, value=int(this_sales * 0.005))
        disallow_custom = {
            "외주가공비": float(d1),
            "가족·특수관계인 인건비": float(d2),
            "차량·접대 등 사적경비": float(d3),
            "무증빙·현금지출": float(d4),
        }

run = st.button("✅ 보고서 생성", type="primary")

if run:
    try:
        df_map = load_mapping_excel(uploaded)
        r = calc_income_rate(df_map, int(industry_code))

        report_md = build_report_md(
            result=r,
            last_sales=float(last_sales),
            this_sales=float(this_sales),
            employees=int(employees),
            category=category,
            insurance_rate=float(insurance_rate),
            ceo_salary=float(ceo_salary),
            corp_tax_rate=float(corp_tax_rate),
            use_disallow=bool(use_disallow),
            disallow_custom=disallow_custom,
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📌 핵심 결과(요약)")
            st.metric("소득율", pct(r.income_rate, 1))
            st.metric("Q값", f"{r.q_value}")
            st.metric("업종코드", f"{int(r.biz_code)}")
            st.info("보고서 본문은 오른쪽에 출력됩니다. 아래에서 .md로 다운로드도 가능합니다.")

        with col2:
            st.subheader("🧾 보고서")
            st.markdown(report_md)

        st.download_button(
            "⬇️ 보고서 다운로드 (Markdown .md)",
            data=report_md.encode("utf-8"),
            file_name=f"report_{industry_code}.md",
            mime="text/markdown",
        )

    except Exception as e:
        st.error(f"오류 발생: {e}")
        st.stop()

st.caption("※ 본 앱은 ‘순이익=매출×소득율’, ‘과세표준≈순이익’ 등 단순화 가정을 포함합니다. 실제 세무 신고/설계는 공제·경비·소득구성에 따라 달라집니다.")

# =========================
# 6) 관리자 페이지 (DB의 is_admin으로만 판단)
# =========================
st.sidebar.markdown("---")
if bool(current_user.get("is_admin", False)):
    st.sidebar.subheader("👑 관리자 메뉴")
    if st.sidebar.checkbox("사용자 승인/차단 관리"):
        st.subheader("👑 사용자 승인 관리")
        users = db_list_users()

        for u in users:
            email = u["email"]
            approved = bool(u.get("approved", False))
            is_admin = bool(u.get("is_admin", False))

            c1, c2, c3, c4 = st.columns([3, 1.2, 1.2, 1.2])
            c1.write(email)
            c2.write("관리자" if is_admin else "-")

            # 승인/차단
            btn_label = "승인" if not approved else "차단"
            if c3.button(btn_label, key=f"appr_{email}"):
                db_upsert_user(email, approved=(not approved))
                st.rerun()

            # 관리자 토글 (자기 자신 해제 방지)
            if email == current_user["email"]:
                c4.write("본인")
            else:
                if c4.button("관리자ON" if not is_admin else "관리자OFF", key=f"admin_{email}"):
                    db_upsert_user(email, is_admin=(not is_admin))
                    # 관리자 계정은 승인도 같이 켜주는 게 안전
                    if not is_admin:
                        db_upsert_user(email, approved=True)
                    st.rerun()
else:
    st.sidebar.caption("관리자 권한이 없습니다.")


