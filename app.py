# app.py
import os
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# Page config (must be first)
# -----------------------------
st.set_page_config(page_title="승인형 제안서 생성기 (업로드+비용방어)", layout="wide")


# =========================================================
# Secrets / Env helpers
# =========================================================
def get_secret(key: str, default: str = "") -> str:
    # Streamlit Cloud: st.secrets, local: env
    if hasattr(st, "secrets") and key in st.secrets:
        return str(st.secrets.get(key, default))
    return os.getenv(key, default)


SUPABASE_URL = get_secret("SUPABASE_URL").strip()
SUPABASE_KEY = get_secret("SUPABASE_KEY").strip()
OPENAI_API_KEY = get_secret("OPENAI_API_KEY").strip()

ADMIN_EMAIL = get_secret("ADMIN_EMAIL", "").strip().lower()
ADMIN_BOOTSTRAP_KEY = get_secret("ADMIN_BOOTSTRAP_KEY", "").strip()

# Usage limits
DAILY_LIMIT = 5
MONTHLY_LIMIT = 100


# =========================================================
# Supabase client
# =========================================================
def get_supabase_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        from supabase import create_client  # type: ignore
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None


sb = get_supabase_client()


# =========================================================
# OpenAI call (robust)
# =========================================================
def call_openai_generate(text_prompt: str) -> str:
    """
    Uses OpenAI API. If quota/billing not set -> raises Exception.
    """
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY가 설정되지 않았습니다.")

    # Prefer official python SDK if available
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=OPENAI_API_KEY)
        # Responses API (recommended)
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=text_prompt,
        )
        # Extract text safely
        out = []
        for item in resp.output:
            if item.type == "message":
                for c in item.content:
                    if c.type == "output_text":
                        out.append(c.text)
        return "\n".join(out).strip() or "응답이 비어있습니다."
    except Exception:
        # Fallback to HTTP if SDK mismatch
        import requests

        r = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": "gpt-4.1-mini", "input": text_prompt},
            timeout=60,
        )
        if r.status_code >= 400:
            raise Exception(f"OpenAI 호출 실패: {r.status_code} / {r.text}")
        data = r.json()
        # Try to parse output text
        out = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        out.append(c.get("text", ""))
        return "\n".join(out).strip() or "응답이 비어있습니다."


# =========================================================
# Auth / Approval (Supabase: users table)
# =========================================================
def ensure_supabase_ready():
    missing = []
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_KEY:
        missing.append("SUPABASE_KEY")
    if missing:
        st.error(
            "Secrets 설정이 부족합니다.\n\n"
            + "누락: " + ", ".join(missing)
            + "\n\nStreamlit Cloud → Manage app → Settings → Secrets에 TOML로 넣어주세요."
        )
        st.stop()
    if sb is None:
        st.error("Supabase 클라이언트를 로드하지 못했습니다. requirements.txt에 supabase가 설치되어 있는지 확인하세요.")
        st.stop()


def db_get_user(email: str) -> Optional[dict]:
    res = sb.table("users").select("*").eq("email", email).execute()
    if res.data:
        return res.data[0]
    return None


def db_upsert_user(email: str, approved: Optional[bool] = None, is_admin: Optional[bool] = None):
    payload = {"email": email}
    if approved is not None:
        payload["approved"] = approved
    if is_admin is not None:
        payload["is_admin"] = is_admin
    sb.table("users").upsert(payload, on_conflict="email").execute()


def db_list_users():
    return sb.table("users").select("*").order("created_at", desc=True).execute().data or []


def db_set_approval(email: str, approved: bool):
    sb.table("users").update({"approved": approved}).eq("email", email).execute()


# =========================================================
# Usage counters (Supabase: usage_counters table)
# - upsert 기반(중복키 방지)
# - 첫 사용 자동 생성
# =========================================================
def get_period_keys() -> Tuple[str, str]:
    now = datetime.utcnow()
    daily_key = now.strftime("%Y-%m-%d")
    monthly_key = now.strftime("%Y-%m")
    return daily_key, monthly_key


def get_usage(email: str, period_type: str, period_key: str) -> int:
    res = (
        sb.table("usage_counters")
        .select("used_count")
        .eq("email", email)
        .eq("period_type", period_type)
        .eq("period_key", period_key)
        .execute()
    )
    if res.data:
        return int(res.data[0].get("used_count", 0))
    return 0


def check_limits(email: str) -> Tuple[bool, str, int, int]:
    daily_key, monthly_key = get_period_keys()
    daily_used = get_usage(email, "daily", daily_key)
    monthly_used = get_usage(email, "monthly", monthly_key)

    if daily_used >= DAILY_LIMIT:
        return False, "오늘 사용 한도를 초과했습니다.", daily_used, monthly_used
    if monthly_used >= MONTHLY_LIMIT:
        return False, "이번 달 사용 한도를 초과했습니다.", daily_used, monthly_used
    return True, "", daily_used, monthly_used


def increment_usage_safe(email: str):
    """
    경쟁 조건에서도 'duplicate key' 에러가 나면 재시도하는 방식으로 안전하게 처리.
    (완전 원자적 increment는 RPC가 필요하지만, 이 정도면 실사용에 충분히 안정적)
    """
    daily_key, monthly_key = get_period_keys()

    for _ in range(2):
        try:
            # DAILY
            daily_now = get_usage(email, "daily", daily_key)
            sb.table("usage_counters").upsert(
                {
                    "email": email,
                    "period_type": "daily",
                    "period_key": daily_key,
                    "used_count": daily_now + 1,
                },
                on_conflict="email,period_type,period_key",
            ).execute()

            # MONTHLY
            monthly_now = get_usage(email, "monthly", monthly_key)
            sb.table("usage_counters").upsert(
                {
                    "email": email,
                    "period_type": "monthly",
                    "period_key": monthly_key,
                    "used_count": monthly_now + 1,
                },
                on_conflict="email,period_type,period_key",
            ).execute()
            return
        except Exception:
            # 재시도
            continue

    raise Exception("사용량 증가 처리 중 오류가 발생했습니다(재시도 실패).")


# =========================================================
# Excel upload → realtime calculation
# 요구: 업종코드(산업분류코드) 입력
# - F열에서 산업분류코드 찾기
# - 같은 행의 C열 = 업종코드(biz_code)
# - K열에서 업종코드 찾기
# - 같은 행의 Q열 = Q값
# - 소득율 = 100 - Q값
# =========================================================
def compute_income_rate_from_excel(df: pd.DataFrame, industry_code: str) -> Tuple[Optional[float], str]:
    """
    Returns (income_rate, message)
    """
    # Excel 컬럼이 A,B,C... 형태로 들어오는 경우 대비:
    # pandas는 컬럼명이 실제 헤더 행에 따라 달라짐.
    # 여기서는 "열 위치" 기반으로 처리 (C=3, F=6, K=11, Q=17) -> 1-index 기준
    # 0-index로는: C=2, F=5, K=10, Q=16
    try:
        col_C = df.columns[2]
        col_F = df.columns[5]
        col_K = df.columns[10]
        col_Q = df.columns[16]
    except Exception:
        return None, "엑셀 형식이 예상과 다릅니다. 최소 Q열(17번째 컬럼)까지 존재해야 합니다."

    # F열에서 산업분류코드 찾기
    # 숫자로 들어오든 문자열로 들어오든 매칭되게 처리
    target = str(industry_code).strip()
    f_series = df[col_F].astype(str).str.strip()

    matches = df[f_series == target]
    if matches.empty:
        return None, f"F열에서 산업분류코드 '{target}'를 찾지 못했습니다."

    biz_code = str(matches.iloc[0][col_C]).strip()
    if not biz_code or biz_code.lower() == "nan":
        return None, "C열 업종코드를 가져오지 못했습니다."

    # K열에서 업종코드 찾기
    k_series = df[col_K].astype(str).str.strip()
    matches2 = df[k_series == biz_code]
    if matches2.empty:
        return None, f"K열에서 업종코드 '{biz_code}'를 찾지 못했습니다."

    q_raw = matches2.iloc[0][col_Q]
    try:
        q_val = float(q_raw)
    except Exception:
        return None, f"Q열 값이 숫자가 아닙니다: {q_raw}"

    income_rate = 100.0 - q_val
    return income_rate, f"업종코드={biz_code}, Q값={q_val} → 소득율={income_rate:.2f}%"


# =========================================================
# UI
# =========================================================
st.title("✅ 승인형 제안서 생성기 (엑셀 업로드 + 비용 방어)")
st.caption("승인된 사용자만 사용 가능 / 하루 5회 / 월 100회 / 업종코드 엑셀 업로드 후 실시간 계산")


# --- Check Supabase required
ensure_supabase_ready()

# --- Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "user_info" not in st.session_state:
    st.session_state.user_info = None

# --- Sidebar: login
with st.sidebar:
    st.header("🔐 접근 제어")

    email_input = st.text_input("이메일", value=st.session_state.user_email or "", placeholder="name@example.com").strip().lower()

    colA, colB = st.columns(2)
    with colA:
        if st.button("로그인", use_container_width=True):
            if not email_input:
                st.warning("이메일을 입력하세요.")
            else:
                # ensure user exists
                u = db_get_user(email_input)
                if u is None:
                    # 최초 로그인: 자동 생성(승인 대기)
                    db_upsert_user(email_input, approved=False, is_admin=False)
                    u = db_get_user(email_input)

                st.session_state.logged_in = True
                st.session_state.user_email = email_input
                st.session_state.user_info = u
                st.success(f"로그인: {email_input}")

    with colB:
        if st.button("로그아웃", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_email = ""
            st.session_state.user_info = None
            st.info("로그아웃 되었습니다.")

    if st.session_state.logged_in and st.session_state.user_info:
        u = db_get_user(st.session_state.user_email)  # refresh
        st.session_state.user_info = u

        st.write(f"로그인: **{st.session_state.user_email}**")
        st.write(f"승인: {'✅' if u.get('approved') else '⏳ 대기'}")
        st.write(f"관리자: {'👑' if u.get('is_admin') else '—'}")

        ok, msg, daily_used, monthly_used = check_limits(st.session_state.user_email)
        st.markdown("### 📌 사용량(비용 방어)")
        st.write(f"- 오늘({datetime.utcnow().strftime('%Y-%m-%d')}): {daily_used} / {DAILY_LIMIT} (잔여 {max(0, DAILY_LIMIT-daily_used)})")
        st.write(f"- 이달({datetime.utcnow().strftime('%Y-%m')}): {monthly_used} / {MONTHLY_LIMIT} (잔여 {max(0, MONTHLY_LIMIT-monthly_used)})")

    # --- Admin bootstrap (최초 1회)
    st.divider()
    with st.expander("🛠 관리자 초기설정(최초 1회)"):
        st.caption("Secrets의 ADMIN_BOOTSTRAP_KEY를 아는 사람만 관리자 지정 가능")
        bootstrap_key = st.text_input("ADMIN_BOOTSTRAP_KEY", type="password", placeholder="Secrets에 넣은 값", key="bootstrap_key_input")
        admin_email = st.text_input("ADMIN_EMAIL(관리자 이메일)", value=ADMIN_EMAIL or "", placeholder="example@gmail.com").strip().lower()

        if st.button("관리자 계정 생성/갱신", use_container_width=True):
            if not admin_email:
                st.error("ADMIN_EMAIL이 비어있습니다. Secrets에 ADMIN_EMAIL을 넣거나 여기 입력하세요.")
            elif not ADMIN_BOOTSTRAP_KEY:
                st.error("Secrets에 ADMIN_BOOTSTRAP_KEY가 없습니다.")
            elif bootstrap_key != ADMIN_BOOTSTRAP_KEY:
                st.error("ADMIN_BOOTSTRAP_KEY가 틀렸습니다.")
            else:
                # Make admin approved + admin
                db_upsert_user(admin_email, approved=True, is_admin=True)
                st.success("관리자 계정을 승인+관리자로 설정했습니다. (이제 관리자 이메일로 로그인하면 관리 화면이 열립니다.)")


# --- Gate: must login
if not st.session_state.logged_in or not st.session_state.user_info:
    st.info("왼쪽 사이드바에서 이메일로 로그인하세요. (최초 로그인 시 자동 등록되며 ‘승인 대기’가 됩니다.)")
    st.stop()

# --- Gate: must approved
user = st.session_state.user_info
if not user.get("approved", False):
    st.warning("현재 ‘승인 대기’ 상태입니다. 관리자가 승인해야 사용할 수 있습니다.")
    st.stop()


# =========================================================
# Main: Excel upload + realtime calc + proposal generation
# =========================================================
left, right = st.columns([1.0, 1.2], gap="large")

with left:
    st.subheader("1) 엑셀 업로드 → 실시간 계산")

    uploaded_file = st.file_uploader("업종코드 엑셀 업로드 (.xlsx)", type=["xlsx"])
    industry_code = st.text_input("산업분류코드 입력 (F열에서 찾음)", placeholder="예: 22232")

    df_excel = None
    if uploaded_file is not None:
        try:
            df_excel = pd.read_excel(uploaded_file)
            st.success(f"업로드 성공: {uploaded_file.name}  (행 {len(df_excel):,} / 열 {len(df_excel.columns):,})")
            with st.expander("미리보기(상위 20행)"):
                st.dataframe(df_excel.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"엑셀 읽기 실패: {e}")
            df_excel = None

    income_rate = None
    income_msg = ""
    if df_excel is not None and industry_code.strip():
        income_rate, income_msg = compute_income_rate_from_excel(df_excel, industry_code.strip())
        if income_rate is None:
            st.error(income_msg)
        else:
            st.success(income_msg)

    st.divider()
    st.subheader("2) 제안서 입력(예시)")
    last_sales = st.text_input("직전년도 매출(예: 9억)", value="")
    this_sales = st.text_input("금년도 예상 매출(예: 11억)", value="")
    employees = st.number_input("직원 수(대표 제외)", min_value=0, step=1, value=5)
    worries = st.text_area("현재 고민/리스크(선택)", value="성실신고, 건강보험료, 비용처리 리스크")

    tone = st.selectbox("문서 톤", ["전문적/숫자중심/리스크체감형", "간결/설득형", "강하게/경고형"], index=0)

with right:
    st.subheader("3) 승인된 사용자만 제안서 생성 + 사용량 제한(비용방어)")

    # show remaining
    ok, msg, daily_used, monthly_used = check_limits(st.session_state.user_email)
    st.write(f"오늘 잔여: **{max(0, DAILY_LIMIT-daily_used)}회** / 이달 잔여: **{max(0, MONTHLY_LIMIT-monthly_used)}회**")

    if st.button("🚀 제안서 생성(OpenAI)", use_container_width=True):
        # limit check
        ok, msg, _, _ = check_limits(st.session_state.user_email)
        if not ok:
            st.error(msg)
            st.stop()

        # increment first (cost defense: 실패해도 카운트할지 정책 선택 가능)
        # 여기서는 "호출 시도" 자체를 비용으로 보고 선차감.
        try:
            increment_usage_safe(st.session_state.user_email)
        except Exception as e:
            st.error(f"사용량 처리 실패: {e}")
            st.stop()

        # Build prompt
        calc_part = ""
        if income_rate is not None:
            calc_part = f"- 업로드 엑셀 기준 소득율(100-Q): {income_rate:.2f}%\n"
        else:
            calc_part = "- 업로드 엑셀 기준 소득율: (미계산)\n"

        prompt = f"""
당신은 한국의 법인전환/세무 리스크 컨설팅 제안서 작성 전문가입니다.
아래 입력을 바탕으로 '컨설팅 제안서'를 한국어로 작성하세요.
숫자/리스크/대안/실행로드맵/기대효과/다음액션을 포함하고, 과장하지 말고 현실적 근거로 설득하세요.

[사용자 입력]
- 직전년도 매출: {last_sales}
- 금년도 예상 매출: {this_sales}
- 직원 수(대표 제외): {employees}
- 산업분류코드: {industry_code}
{calc_part}
- 현재 고민/리스크: {worries}
- 문서 톤: {tone}

[출력 형식]
1) 요약(핵심 5줄)
2) 현재 리스크 진단(세무/건보/성실신고/조사리스크 관점)
3) 법인전환 필요성(왜 지금)
4) 실행 방안(단계별 체크리스트)
5) 예상 효과(정량/정성)
6) 필요 자료 요청 목록
7) 컨설팅 범위/일정(샘플)
8) 면책/유의사항(간단)

주의: 숫자는 사용자가 준 값만 사용하고, 추정치가 필요하면 '추정'임을 명확히 표시.
""".strip()

        try:
            result = call_openai_generate(prompt)
            st.success("생성 완료")
            st.text_area("생성된 제안서", value=result, height=520)
        except Exception as e:
            st.error(str(e))

    st.caption("※ OpenAI 429 insufficient_quota가 뜨면 OpenAI 결제/크레딧(카드등록)이 안 된 상태입니다.")


# =========================================================
# Admin panel (approve users, view usage)
# =========================================================
if user.get("is_admin", False):
    st.divider()
    st.header("👑 관리자: 승인/사용량 관리")

    users = db_list_users()
    if not users:
        st.info("등록된 사용자가 없습니다.")
    else:
        st.subheader("승인 대기/승인 사용자 목록")
        dfu = pd.DataFrame(users)
        st.dataframe(dfu, use_container_width=True)

        st.subheader("승인 변경")
        target_email = st.text_input("대상 이메일", placeholder="승인/해제할 이메일").strip().lower()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ 승인", use_container_width=True):
                if not target_email:
                    st.warning("이메일을 입력하세요.")
                else:
                    db_set_approval(target_email, True)
                    st.success(f"승인 완료: {target_email}")
        with c2:
            if st.button("⛔ 승인 해제", use_container_width=True):
                if not target_email:
                    st.warning("이메일을 입력하세요.")
                else:
                    db_set_approval(target_email, False)
                    st.success(f"승인 해제: {target_email}")

    st.subheader("사용량 카운터(최근)")
    try:
        usage_rows = sb.table("usage_counters").select("*").order("updated_at", desc=True).limit(200).execute().data or []
        if usage_rows:
            st.dataframe(pd.DataFrame(usage_rows), use_container_width=True)
        else:
            st.info("사용량 데이터가 아직 없습니다.")
    except Exception as e:
        st.warning(f"usage_counters 조회 실패: {e}")














