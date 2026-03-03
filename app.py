import os
import re
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="승인형 성실신고/법인전환 보고서", layout="wide")

# -----------------------------
# Secrets helpers
# -----------------------------
def sget(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)

SUPABASE_URL = sget("SUPABASE_URL").strip()
SUPABASE_KEY = sget("SUPABASE_KEY").strip()  # service_role 권장
OPENAI_API_KEY = sget("OPENAI_API_KEY").strip()

ADMIN_EMAIL = sget("ADMIN_EMAIL", "").strip().lower()
ADMIN_BOOTSTRAP_KEY = sget("ADMIN_BOOTSTRAP_KEY", "").strip()

DAILY_LIMIT = int(sget("DAILY_LIMIT", "10"))
MONTHLY_LIMIT = int(sget("MONTHLY_LIMIT", "100"))

OPENAI_MODEL = sget("OPENAI_MODEL", "gpt-4.1-mini").strip()  # 필요시 변경

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def must_have_secrets():
    missing = []
    if not SUPABASE_URL: missing.append("SUPABASE_URL")
    if not SUPABASE_KEY: missing.append("SUPABASE_KEY")
    if not OPENAI_API_KEY: missing.append("OPENAI_API_KEY")
    if not ADMIN_BOOTSTRAP_KEY: missing.append("ADMIN_BOOTSTRAP_KEY")
    if missing:
        st.error(f"Secrets 설정이 부족합니다.\n\n누락: {', '.join(missing)}")
        st.stop()

must_have_secrets()

# -----------------------------
# Supabase client
# -----------------------------
@st.cache_resource(show_spinner=False)
def sb():
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# OpenAI client
# -----------------------------
@st.cache_resource(show_spinner=False)
def oai():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Utility
# -----------------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def period_keys() -> Tuple[str, str]:
    dt = now_utc()
    return dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m")

def norm_email(x: str) -> str:
    return (x or "").strip().lower()

def valid_email(x: str) -> bool:
    return bool(EMAIL_RE.match(norm_email(x)))

def parse_money_kr(s: str) -> Optional[int]:
    """
    '8억', '10억', '1.2억', '900000000', '9억 5천' 같은 입력을 단순 파싱.
    완벽하진 않지만 실무 입력에 충분히 유용.
    """
    if not s:
        return None
    t = str(s).strip().replace(",", "").replace("원", "").replace(" ", "")
    if t.isdigit():
        return int(t)

    # 억/만 단위
    # 예: 9억5천(=9.5억) 지원 간단화
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)억(?:([0-9]+)천)?$", t)
    if m:
        eok = float(m.group(1))
        cheon = m.group(2)
        val = eok * 100_000_000
        if cheon:
            val += int(cheon) * 10_000_000  # 1천(만) 단순화가 아니라 '천'이 애매하므로 1천=1천만으로 가정 X
        return int(val)

    m2 = re.match(r"^([0-9]+(?:\.[0-9]+)?)억$", t)
    if m2:
        return int(float(m2.group(1)) * 100_000_000)

    # 만 단위
    m3 = re.match(r"^([0-9]+(?:\.[0-9]+)?)만$", t)
    if m3:
        return int(float(m3.group(1)) * 10_000)

    return None

def fmt_won(x: Optional[int]) -> str:
    if x is None:
        return "-"
    return f"{x:,}원"

# -----------------------------
# DB: users
# -----------------------------
def db_get_user(email: str) -> Optional[Dict[str, Any]]:
    r = sb().table("users").select("*").eq("email", email).limit(1).execute()
    data = r.data or []
    return data[0] if data else None

def db_create_user_if_missing(email: str) -> Dict[str, Any]:
    u = db_get_user(email)
    if u:
        return u
    payload = {
        "email": email,
        "approved": False,
        "is_admin": False,
        "created_at": now_utc().isoformat(),
    }
    sb().table("users").insert(payload).execute()
    return db_get_user(email) or payload

def db_list_users() -> List[Dict[str, Any]]:
    r = sb().table("users").select("*").order("created_at", desc=True).execute()
    return r.data or []

def db_set_approved(email: str, approved: bool):
    sb().table("users").update({"approved": bool(approved)}).eq("email", email).execute()

def db_set_admin(email: str, is_admin: bool):
    sb().table("users").update({"is_admin": bool(is_admin)}).eq("email", email).execute()

# -----------------------------
# DB: usage_counters (중복키 방지 upsert)
# -----------------------------
def usage_get(email: str, period_type: str, period_key: str) -> int:
    r = (
        sb()
        .table("usage_counters")
        .select("used_count")
        .eq("email", email)
        .eq("period_type", period_type)
        .eq("period_key", period_key)
        .limit(1)
        .execute()
    )
    data = r.data or []
    return int(data[0]["used_count"]) if data else 0

def usage_inc(email: str) -> Dict[str, int]:
    daily_key, monthly_key = period_keys()
    # upsert로 "첫 사용 자동 생성 + 중복키 방지"
    d_now = usage_get(email, "daily", daily_key)
    m_now = usage_get(email, "monthly", monthly_key)

    sb().table("usage_counters").upsert(
        {
            "email": email,
            "period_type": "daily",
            "period_key": daily_key,
            "used_count": d_now + 1,
            "created_at": now_utc().isoformat(),
            "updated_at": now_utc().isoformat(),
        },
        on_conflict="email,period_type,period_key",
    ).execute()

    sb().table("usage_counters").upsert(
        {
            "email": email,
            "period_type": "monthly",
            "period_key": monthly_key,
            "used_count": m_now + 1,
            "created_at": now_utc().isoformat(),
            "updated_at": now_utc().isoformat(),
        },
        on_conflict="email,period_type,period_key",
    ).execute()

    return {"daily_used": d_now + 1, "monthly_used": m_now + 1}

def quota_status(email: str) -> Dict[str, Any]:
    daily_key, monthly_key = period_keys()
    d = usage_get(email, "daily", daily_key)
    m = usage_get(email, "monthly", monthly_key)
    return {
        "daily_key": daily_key,
        "monthly_key": monthly_key,
        "daily_used": d,
        "monthly_used": m,
        "daily_limit": DAILY_LIMIT,
        "monthly_limit": MONTHLY_LIMIT,
        "daily_remain": max(0, DAILY_LIMIT - d),
        "monthly_remain": max(0, MONTHLY_LIMIT - m),
    }

def ensure_quota(email: str):
    q = quota_status(email)
    if q["daily_used"] >= DAILY_LIMIT:
        st.error("오늘 사용 한도를 초과했습니다.")
        st.stop()
    if q["monthly_used"] >= MONTHLY_LIMIT:
        st.error("이번 달 사용 한도를 초과했습니다.")
        st.stop()

# -----------------------------
# Excel: 소득율 계산 (두 가지 레이아웃 지원)
#  A) 원본형: F(표준산업분류) -> C(업종코드) -> K(업종코드) -> Q(Q값)
#  B) 현재 업로드 파일형: '표준산업 분류' + '업종코드' + '단순경비율(일반율)'
# -----------------------------
def compute_income_rate(df: pd.DataFrame, industry_code: str) -> Tuple[Optional[Dict[str, Any]], str]:
    code = str(industry_code).strip()

    cols = [str(c) for c in df.columns]

    # --- B) 현재 파일형 탐지
    # 표준산업 분류 / 단순경비율(일반율) / 업종코드 같은 컬럼명 존재
    col_std = None
    col_rate = None
    col_biz = None

    for c in df.columns:
        s = str(c)
        if "표준산업" in s and "분류" in s:
            col_std = c
        if "단순경비율" in s:
            col_rate = c
        if s.strip() == "업종코드" or "귀속" in s and "업종코드" in s:
            # 우선순위: 정확히 '업종코드'
            if str(c).strip() == "업종코드":
                col_biz = c

    if col_std is not None and col_rate is not None:
        # 표준산업분류에서 매칭
        m = df[df[col_std].astype(str).str.strip() == code]
        if m.empty:
            return None, f"표준산업분류 컬럼에서 '{code}'를 찾지 못했습니다."
        row = m.iloc[0]
        biz_code = str(row[col_biz]).strip() if col_biz is not None else ""
        q_like = row[col_rate]
        try:
            q_val = float(q_like)
        except Exception:
            return None, f"단순경비율 값이 숫자가 아닙니다: {q_like}"

        income_rate = 100.0 - q_val
        return {
            "industry_code": code,
            "biz_code": biz_code,
            "q_value": q_val,
            "income_rate": income_rate,
            "source": "단순경비율(일반율) 기반(소득율=100-단순경비율)",
        }, "OK"

    # --- A) 원본형(열 위치 기반) 시도
    # 최소 17열 이상 필요(Q=17번째=인덱스16)
    if len(df.columns) >= 17:
        col_C = df.columns[2]   # C
        col_F = df.columns[5]   # F
        col_K = df.columns[10]  # K
        col_Q = df.columns[16]  # Q

        f = df[col_F].astype(str).str.strip()
        m1 = df[f == code]
        if m1.empty:
            return None, f"F열(6번째 컬럼)에서 산업분류코드 '{code}'를 찾지 못했습니다."
        biz_code = str(m1.iloc[0][col_C]).strip()

        k = df[col_K].astype(str).str.strip()
        m2 = df[k == biz_code]
        if m2.empty:
            return None, f"K열(11번째 컬럼)에서 업종코드 '{biz_code}'를 찾지 못했습니다."
        q_raw = m2.iloc[0][col_Q]
        try:
            q_val = float(q_raw)
        except Exception:
            return None, f"Q열 값이 숫자가 아닙니다: {q_raw}"

        income_rate = 100.0 - q_val
        return {
            "industry_code": code,
            "biz_code": biz_code,
            "q_value": q_val,
            "income_rate": income_rate,
            "source": "원본형(F→C→K→Q) 기반(소득율=100-Q)",
        }, "OK"

    return None, "엑셀 컬럼 구조를 인식하지 못했습니다. (표준산업분류/단순경비율 파일 또는 원본형 F/C/K/Q 파일을 사용하세요.)"

# -----------------------------
# Tax calc (간이)
#  - 종합소득세: 2023~2024 귀속 구간(국세청 표 기준)을 코드에 내장
#  - 지방소득세: 산출세액의 10% 가산
# -----------------------------
INCOME_TAX_BRACKETS = [
    (14_000_000, 0.06, 0),
    (50_000_000, 0.15, 1_260_000),
    (88_000_000, 0.24, 5_760_000),
    (150_000_000, 0.35, 15_440_000),
    (300_000_000, 0.38, 19_940_000),
    (500_000_000, 0.40, 25_940_000),
    (1_000_000_000, 0.42, 35_940_000),
    (10_000_000_000_000, 0.45, 65_940_000),
]

def calc_income_tax(pretax_income: int) -> Dict[str, int]:
    """
    매우 단순화: 필요경비/공제 등 미반영.
    과세표준=순이익 가정.
    """
    x = max(0, int(pretax_income))
    rate = 0.0
    deduct = 0
    for limit, r, d in INCOME_TAX_BRACKETS:
        if x <= limit:
            rate = r
            deduct = d
            break
    national = int(x * rate - deduct)
    local = int(national * 0.10)
    total = national + local
    return {"national": max(0, national), "local": max(0, local), "total": max(0, total)}

def risk_level_faithful_filing(category: str, revenue: int) -> Tuple[str, int]:
    """
    국세청 기준(요청하신 구간):
    - 도소매 15억
    - 제조/건설 등 7.5억
    - 서비스/부동산임대 5억
    """
    cat = category
    thr = 0
    if cat == "도소매":
        thr = 1_500_000_000
    elif cat in ("제조", "건설"):
        thr = 750_000_000
    else:
        thr = 500_000_000

    if revenue < thr * 0.8:
        return "낮음", thr
    if revenue < thr:
        return "보통", thr
    if revenue < thr * 1.2:
        return "높음", thr
    return "매우 높음", thr

def cost_denial_simulation(revenue: int) -> List[Dict[str, Any]]:
    """
    보수적 비율(요청하신 제조업 예시의 '하단' 사용)
    """
    items = [
        ("외주가공비", 0.02),
        ("가족·특수관계인 인건비", 0.01),
        ("차량·접대 등 사적경비", 0.01),
        ("무증빙·현금지출", 0.005),
    ]
    out = []
    for name, pct in items:
        denied = int(revenue * pct)
        out.append({"item": name, "pct": pct, "denied": denied})
    return out

def estimate_health_ins_increase(additional_income: int) -> int:
    """
    건강보험은 실제로 소득/재산/자동차 등 복합. 여기서는 '추정'으로
    추가 소득의 7%를 연간 증가분으로 매우 보수적 추정(설명용).
    """
    return int(max(0, additional_income) * 0.07)

# -----------------------------
# UI
# -----------------------------
st.title("✅ 개인사업자 성실신고 리스크 및 법인전환 전략 분석 AI (업로드 포함)")
st.caption("승인된 사용자만 사용 / 하루 5회 / 월 100회 / 엑셀 업로드로 소득율 자동 산출 + 5년 리스크 보고서 생성")

# Session
if "email" not in st.session_state:
    st.session_state.email = ""
if "user" not in st.session_state:
    st.session_state.user = None

def refresh_user():
    if st.session_state.email:
        st.session_state.user = db_get_user(st.session_state.email)

# Sidebar login
with st.sidebar:
    st.header("🔐 접근 제어")

    email_in = st.text_input("이메일", value=st.session_state.email, placeholder="name@example.com")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("로그인", use_container_width=True):
            e = norm_email(email_in)
            if not valid_email(e):
                st.warning("이메일 형식이 올바르지 않습니다.")
            else:
                st.session_state.email = e
                db_create_user_if_missing(e)
                refresh_user()
                st.rerun()
    with c2:
        if st.button("로그아웃", use_container_width=True):
            st.session_state.email = ""
            st.session_state.user = None
            st.rerun()

    refresh_user()
    if st.session_state.user:
        u = st.session_state.user
        st.success(f"로그인: {u['email']}")
        st.write(f"승인: {'✅' if u.get('approved') else '⏳ 승인대기'}")
        st.write(f"관리자: {'👑' if u.get('is_admin') else '-'}")

        q = quota_status(u["email"])
        st.caption("📌 사용량(비용 방어)")
        st.write(f"- 오늘({q['daily_key']}): {q['daily_used']} / {q['daily_limit']} (잔여 {q['daily_remain']})")
        st.write(f"- 이번달({q['monthly_key']}): {q['monthly_used']} / {q['monthly_limit']} (잔여 {q['monthly_remain']})")

    st.divider()
    with st.expander("🛠 관리자 초기설정(최초 1회)"):
        st.caption("ADMIN_BOOTSTRAP_KEY가 맞으면 해당 이메일을 관리자+승인 처리합니다.")
        boot_key = st.text_input("ADMIN_BOOTSTRAP_KEY", type="password")
        admin_email = st.text_input("관리자 이메일", value=ADMIN_EMAIL or "", placeholder="admin@example.com").strip().lower()
        if st.button("관리자 계정 생성/갱신", use_container_width=True):
            if boot_key != ADMIN_BOOTSTRAP_KEY:
                st.error("키가 일치하지 않습니다.")
            elif not valid_email(admin_email):
                st.error("관리자 이메일 형식이 올바르지 않습니다.")
            else:
                db_create_user_if_missing(admin_email)
                db_set_approved(admin_email, True)
                db_set_admin(admin_email, True)
                st.success("관리자 설정 완료!")
                st.rerun()

# Gate
if not st.session_state.user:
    st.info("왼쪽 사이드바에서 이메일로 로그인하세요.")
    st.stop()

user = st.session_state.user
if not user.get("approved") and not user.get("is_admin"):
    st.warning("등록되었습니다. 관리자 승인 대기 중입니다.")
    st.stop()

# Admin panel
if user.get("is_admin"):
    st.subheader("👑 관리자: 사용자 승인/관리")
    users = db_list_users()
    if users:
        st.dataframe(
            [{"email": u["email"], "approved": u.get("approved"), "is_admin": u.get("is_admin"), "created_at": u.get("created_at")} for u in users],
            use_container_width=True,
        )
    tgt = st.text_input("대상 이메일(승인/해제)", key="tgt_email").strip().lower()
    a1, a2 = st.columns(2)
    with a1:
        if st.button("✅ 승인", use_container_width=True):
            if valid_email(tgt):
                db_create_user_if_missing(tgt)
                db_set_approved(tgt, True)
                st.success("승인 완료")
                st.rerun()
            else:
                st.error("이메일이 올바르지 않습니다.")
    with a2:
        if st.button("⛔ 승인 해제", use_container_width=True):
            if valid_email(tgt):
                db_set_approved(tgt, False)
                st.success("승인 해제 완료")
                st.rerun()
            else:
                st.error("이메일이 올바르지 않습니다.")
    st.divider()

# Main inputs
st.subheader("📎 1) 기준 엑셀 업로드")
uploaded = st.file_uploader("업종코드 엑셀 업로드 (.xlsx)", type=["xlsx"])

st.subheader("🧾 2) 입력")
industry_code = st.text_input("산업분류코드(숫자 그대로)", placeholder="예: 25913")
last_sales_s = st.text_input("작년 매출", placeholder="예: 8억")
this_sales_s = st.text_input("금년 예상 매출", placeholder="예: 10억")
employees = st.number_input("직원 수(대표 제외)", min_value=0, step=1, value=6)

category = st.selectbox("업종 분류(성실신고 기준용)", ["제조", "도소매", "건설", "서비스"], index=0)

# Parse money
last_sales = parse_money_kr(last_sales_s)
this_sales = parse_money_kr(this_sales_s)

# Excel reading + income rate
income_pack = None
if uploaded is not None and industry_code.strip():
    try:
        df = pd.read_excel(uploaded)
        income_pack, msg = compute_income_rate(df, industry_code.strip())
        if income_pack is None:
            st.error(msg)
        else:
            st.success(f"소득율 산출 성공 ({income_pack['source']})")
            st.write({
                "산업분류코드": income_pack["industry_code"],
                "업종코드": income_pack.get("biz_code", ""),
                "Q값(또는 단순경비율)": income_pack["q_value"],
                "소득율(%)": round(income_pack["income_rate"], 2),
            })
    except Exception as e:
        st.error(f"엑셀 읽기 실패: {e}")

st.divider()
st.subheader("📝 3) 보고서 생성 (승인된 사용자만 / 사용량 제한 적용)")

btn = st.button("🚀 보고서 생성(OpenAI)", type="primary", use_container_width=True)

if btn:
    # Input validation
    if income_pack is None:
        st.error("엑셀 업로드 + 산업분류코드 입력 후 소득율 산출을 먼저 완료하세요.")
        st.stop()
    if last_sales is None or this_sales is None:
        st.error("매출 입력 형식을 확인하세요. (예: 8억, 10억 또는 숫자)")
        st.stop()

    # Quota check
    ensure_quota(user["email"])

    # ---- Deterministic calculations (숫자 기반) ----
    income_rate = float(income_pack["income_rate"]) / 100.0
    last_profit = int(last_sales * income_rate)
    this_profit = int(this_sales * income_rate)

    tax_last = calc_income_tax(last_profit)
    tax_this = calc_income_tax(this_profit)

    # +1% / -1% sensitivity
    up_profit = int(this_sales * ((income_rate + 0.01)))
    dn_profit = int(this_sales * max(0, (income_rate - 0.01)))
    tax_up = calc_income_tax(up_profit)
    tax_dn = calc_income_tax(dn_profit)

    risk, threshold = risk_level_faithful_filing(category, this_sales)

    # 비용 부인 시뮬 (성실신고 대상 위험이 '보통' 이상이면 가정)
    denial_items = cost_denial_simulation(this_sales)
    denial_rows = []
    total_denied = 0
    total_add_tax = 0
    total_add_health = 0

    for it in denial_items:
        denied = it["denied"]
        add_income = denied
        add_tax = calc_income_tax(this_profit + add_income)["total"] - tax_this["total"]
        add_health = estimate_health_ins_increase(add_income)

        denial_rows.append({
            "항목": it["item"],
            "가정비율": f"{int(it['pct']*1000)/10:.1f}%",
            "가정 비용부인 금액": denied,
            "증가 과세소득": add_income,
            "추가 종합소득세+지방세(추정)": add_tax,
            "건강보험 증가(연, 추정)": add_health,
        })
        total_denied += denied
        total_add_tax += add_tax
        total_add_health += add_health

    # 3년/5년 누적(단순 누적: 동일 조건 반복)
    base_3y = (tax_this["total"] + 0) * 3  # 건보는 별도 추정치를 넣고 싶으면 여기에 추가
    add_3y = (total_add_tax + total_add_health) * 3
    add_5y = (total_add_tax + total_add_health) * 5

    # 법인 전환 비교(단순화: 법인세 9% 가정, 대표급여 6,500만원 가정)
    ceo_salary = 65_000_000
    corp_tax = int(max(0, this_profit) * 0.09)  # 단순화
    # (건보 직장가입 효과는 실제 급여/사업장 구조에 따라 달라 '정성'으로만 언급)

    # 사용량 차감(성공적으로 생성 시도할 때 1회)
    usage_inc(user["email"])

    # ---- OpenAI prompt (어제처럼 “보고서 구조+5년치” 강제) ----
    denial_table_md = "|항목|가정비율|비용부인|과세소득증가|추가세금(추정)|건보증가(연,추정)|\n|---|---:|---:|---:|---:|---:|\n"
    for r in denial_rows:
        denial_table_md += f"|{r['항목']}|{r['가정비율']}|{r['가정 비용부인 금액']:,}|{r['증가 과세소득']:,}|{r['추가 종합소득세+지방세(추정)']:,}|{r['건강보험 증가(연, 추정)']:,}|\n"

    prompt = f"""
너는 “개인사업자 성실신고 리스크 및 법인전환 전략 분석 AI”다.
반드시 숫자 중심, 과장 금지, 상담에 바로 쓰는 제안서 톤으로 작성한다.

[입력/산출값(계산 완료)]
- 산업분류코드: {income_pack['industry_code']}
- 업종코드: {income_pack.get('biz_code','')}
- Q값(또는 단순경비율): {income_pack['q_value']}
- 소득율(%): {income_pack['income_rate']:.2f}

- 작년 매출: {last_sales:,}원
- 금년 매출: {this_sales:,}원
- 직원수(대표 제외): {employees}명
- 업종분류(성실신고 기준): {category}

[순이익 추정]
- 작년 순이익: {last_profit:,}원
- 금년 순이익: {this_profit:,}원

[종합소득세(추정)]
- 작년 세금(국세): {tax_last['national']:,}원 / 지방세: {tax_last['local']:,}원 / 합계: {tax_last['total']:,}원
- 금년 세금(국세): {tax_this['national']:,}원 / 지방세: {tax_this['local']:,}원 / 합계: {tax_this['total']:,}원

[민감도(소득율 ±1%)]
- 소득율 +1% 시 세금(합계): {tax_up['total']:,}원 (증가분: {(tax_up['total']-tax_this['total']):,}원)
- 소득율 -1% 시 세금(합계): {tax_dn['total']:,}원 (감소분: {(tax_this['total']-tax_dn['total']):,}원)

[성실신고확인대상 위험]
- 기준 매출: {threshold:,}원
- 금년 매출: {this_sales:,}원
- 위험도: {risk}

[비용 부인 시뮬레이션(보수적 가정)]
- 총 비용부인 가정: {total_denied:,}원
- 총 추가세금(추정): {total_add_tax:,}원
- 총 건보 증가(연, 추정): {total_add_health:,}원

{denial_table_md}

[누적 리스크(단순 누적)]
- 3년 누적 증가분(세금+건보): {add_3y:,}원
- 5년 누적 증가분(세금+건보): {add_5y:,}원

[법인 전환 단순 비교]
- 법인세(단순 9% 가정, 금년 순이익 기준): {corp_tax:,}원
- 대표 급여 가정: {ceo_salary:,}원

[보고서 출력 순서(반드시 지킬 것)]
1) 소득율 산출 결과(표로)
2) 순이익 추정
3) 종합소득세 계산(지방세 포함, 민감도 포함)
4) 성실신고 대상 여부 판단(기준과 비교, 위험도)
5) 비용 부인 시뮬레이션(항목별 표 + “비용 1억 정리 시 세금 약 ○○원 증가” 구조로 설명)
6) 3년 누적 리스크 + 5년 누적 가능성(숫자 명확히)
7) 법인 전환 비교(개인 유지 vs 정리 후 vs 법인전환) 간단 비교표
8) 전략적 결론(실행 체크리스트)
9) 1차 미팅 클로징 멘트(자연스럽게)

주의:
- 위 숫자는 그대로 사용하고, 추가 추정이 필요하면 “추정”이라고 표시.
- 건강보험은 실제로 재산/자동차 등 반영되므로 “추정치”임을 명시.
""".strip()

    try:
        with st.spinner("보고서 생성 중(OpenAI)..."):
            resp = oai().responses.create(model=OPENAI_MODEL, input=prompt)
            report_text = resp.output_text.strip()

        st.success("보고서 생성 완료")
        st.markdown(report_text)

        st.download_button(
            "⬇️ 보고서(.md) 다운로드",
            data=report_text.encode("utf-8"),
            file_name=f"report_{income_pack['industry_code']}_{period_keys()[0]}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"OpenAI 호출 실패: {e}")
        st.stop()
















