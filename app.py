import os
import re
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

import streamlit as st
from supabase import create_client, Client
from openai import OpenAI

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(page_title="승인형 제안서 생성기", layout="wide")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def sget(key: str, default: Optional[str] = None) -> Optional[str]:
    """Streamlit Secrets 우선, 없으면 환경변수 fallback"""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


SUPABASE_URL = sget("SUPABASE_URL")
SUPABASE_KEY = sget("SUPABASE_KEY")  # service_role 권장(Secrets에만)
OPENAI_API_KEY = sget("OPENAI_API_KEY")
ADMIN_BOOTSTRAP_KEY = sget("ADMIN_BOOTSTRAP_KEY")

DEFAULT_DAILY_LIMIT = int(sget("DEFAULT_DAILY_LIMIT", "5"))
DEFAULT_MONTHLY_LIMIT = int(sget("DEFAULT_MONTHLY_LIMIT", "100"))
OPENAI_MODEL = sget("OPENAI_MODEL", "gpt-5-mini")


def require_secrets():
    missing = []
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_KEY:
        missing.append("SUPABASE_KEY")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not ADMIN_BOOTSTRAP_KEY:
        missing.append("ADMIN_BOOTSTRAP_KEY")
    if missing:
        st.error(
            "Secrets 설정이 부족합니다.\n\n"
            f"누락: {', '.join(missing)}\n\n"
            "Streamlit Cloud → Manage app → Settings → Secrets에 TOML로 넣어주세요."
        )
        st.stop()


require_secrets()


@st.cache_resource(show_spinner=False)
def sb() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


@st.cache_resource(show_spinner=False)
def oai() -> OpenAI:
    # 공식 Python SDK 사용: OpenAI()는 OPENAI_API_KEY 환경변수/설정 사용 가능 :contentReference[oaicite:4]{index=4}
    return OpenAI(api_key=OPENAI_API_KEY)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def period_keys() -> Tuple[str, str]:
    """(daily_key, monthly_key)"""
    dt = now_utc()
    return dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m")


def norm_email(x: str) -> str:
    return (x or "").strip().lower()


def valid_email(x: str) -> bool:
    return bool(EMAIL_RE.match(norm_email(x)))


# -----------------------------
# DB helpers
# -----------------------------
def db_get_user(email: str) -> Optional[Dict[str, Any]]:
    email = norm_email(email)
    res = sb().table("users").select("*").eq("email", email).limit(1).execute()
    data = res.data or []
    return data[0] if data else None


def db_create_user_if_missing(email: str) -> Dict[str, Any]:
    email = norm_email(email)
    u = db_get_user(email)
    if u:
        return u
    payload = {
        "email": email,
        "approved": False,
        "is_admin": False,
        "daily_limit": DEFAULT_DAILY_LIMIT,
        "monthly_limit": DEFAULT_MONTHLY_LIMIT,
    }
    sb().table("users").insert(payload).execute()
    return db_get_user(email) or payload


def db_set_approved(email: str, approved: bool):
    sb().table("users").update({"approved": bool(approved)}).eq("email", norm_email(email)).execute()


def db_set_admin(email: str, is_admin: bool):
    sb().table("users").update({"is_admin": bool(is_admin)}).eq("email", norm_email(email)).execute()


def db_update_limits(email: str, daily_limit: int, monthly_limit: int):
    sb().table("users").update(
        {"daily_limit": int(daily_limit), "monthly_limit": int(monthly_limit)}
    ).eq("email", norm_email(email)).execute()


def db_list_users() -> List[Dict[str, Any]]:
    res = sb().table("users").select("*").order("created_at", desc=True).execute()
    return res.data or []


def usage_get(email: str, period_type: str, period_key: str) -> int:
    res = (
        sb()
        .table("usage")
        .select("count")
        .eq("email", norm_email(email))
        .eq("period_type", period_type)
        .eq("period_key", period_key)
        .limit(1)
        .execute()
    )
    data = res.data or []
    return int(data[0]["count"]) if data else 0


def usage_increment(email: str, period_type: str, period_key: str, by: int = 1) -> int:
    """단순 upsert로 증가(동시성 극단 상황은 드물다고 가정)"""
    email = norm_email(email)
    current = usage_get(email, period_type, period_key)
    new_count = current + by
    payload = {
        "email": email,
        "period_type": period_type,
        "period_key": period_key,
        "count": new_count,
        "updated_at": now_utc().isoformat(),
    }
    sb().table("usage").upsert(payload).execute()
    return new_count


def get_remaining_quota(user: Dict[str, Any]) -> Dict[str, Any]:
    email = user["email"]
    daily_key, monthly_key = period_keys()

    used_today = usage_get(email, "daily", daily_key)
    used_month = usage_get(email, "monthly", monthly_key)

    daily_limit = int(user.get("daily_limit", DEFAULT_DAILY_LIMIT))
    monthly_limit = int(user.get("monthly_limit", DEFAULT_MONTHLY_LIMIT))

    return {
        "daily_key": daily_key,
        "monthly_key": monthly_key,
        "used_today": used_today,
        "used_month": used_month,
        "daily_limit": daily_limit,
        "monthly_limit": monthly_limit,
        "remain_today": max(0, daily_limit - used_today),
        "remain_month": max(0, monthly_limit - used_month),
    }


# -----------------------------
# 세션
# -----------------------------
if "email" not in st.session_state:
    st.session_state.email = ""
if "user" not in st.session_state:
    st.session_state.user = None


def refresh_user():
    if st.session_state.email:
        st.session_state.user = db_get_user(st.session_state.email)


# -----------------------------
# UI
# -----------------------------
st.title("✅ 승인형 제안서 생성기 (OpenAI API + 비용 방어)")

with st.sidebar:
    st.header("🔐 접근 제어")

    email_in = st.text_input("이메일", value=st.session_state.email, placeholder="name@example.com")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("로그인", use_container_width=True):
            e = norm_email(email_in)
            if not valid_email(e):
                st.warning("이메일 형식이 올바르지 않습니다.")
            else:
                st.session_state.email = e
                db_create_user_if_missing(e)
                refresh_user()
                st.rerun()

    with col2:
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

        quota = get_remaining_quota(u)
        st.caption("📌 사용량(비용 방어)")
        st.write(f"- 오늘({quota['daily_key']}): {quota['used_today']} / {quota['daily_limit']} (잔여 {quota['remain_today']})")
        st.write(f"- 이번달({quota['monthly_key']}): {quota['used_month']} / {quota['monthly_limit']} (잔여 {quota['remain_month']})")

    st.divider()

    # 관리자 부트스트랩 (최초 1회)
    with st.expander("🛠 관리자 초기설정(최초 1회)", expanded=False):
        st.caption("ADMIN_BOOTSTRAP_KEY가 맞으면 해당 이메일을 관리자+승인 처리합니다.")
        boot_key = st.text_input("ADMIN_BOOTSTRAP_KEY", type="password")
        admin_email = st.text_input("관리자로 지정할 이메일", placeholder="admin@example.com")

        if st.button("관리자 계정 생성/갱신", use_container_width=True):
            if boot_key != ADMIN_BOOTSTRAP_KEY:
                st.error("키가 일치하지 않습니다.")
            else:
                ae = norm_email(admin_email)
                if not valid_email(ae):
                    st.warning("이메일 형식이 올바르지 않습니다.")
                else:
                    db_create_user_if_missing(ae)
                    db_set_approved(ae, True)
                    db_set_admin(ae, True)
                    st.success("관리자 설정 완료!")
                    if st.session_state.email == ae:
                        refresh_user()
                    st.rerun()


# -----------------------------
# 승인 체크
# -----------------------------
if not st.session_state.user:
    st.info("왼쪽 사이드바에서 이메일로 로그인하세요.")
    st.stop()

user = st.session_state.user

if not user.get("approved") and not user.get("is_admin"):
    st.warning("등록되었습니다. 관리자 승인 대기 중입니다.")
    st.stop()


# -----------------------------
# 관리자 화면: 승인/사용량 제한 관리
# -----------------------------
if user.get("is_admin"):
    st.subheader("👑 관리자: 승인 / 사용량 제한 관리")

    users = db_list_users()
    if users:
        st.dataframe(
            [
                {
                    "email": u["email"],
                    "approved": u.get("approved", False),
                    "is_admin": u.get("is_admin", False),
                    "daily_limit": u.get("daily_limit", DEFAULT_DAILY_LIMIT),
                    "monthly_limit": u.get("monthly_limit", DEFAULT_MONTHLY_LIMIT),
                }
                for u in users
            ],
            use_container_width=True,
        )

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    target = c1.text_input("대상 이메일", placeholder="someone@example.com")
    approved = c2.selectbox("승인", [True, False], index=0)
    is_admin = c3.selectbox("관리자", [False, True], index=0)

    with c4:
        if st.button("권한 적용", use_container_width=True):
            te = norm_email(target)
            if not valid_email(te):
                st.error("대상 이메일이 올바르지 않습니다.")
            else:
                db_create_user_if_missing(te)
                db_set_approved(te, approved)
                db_set_admin(te, is_admin)
                st.success("적용 완료")
                st.rerun()

    st.markdown("### 🎯 사용자별 사용량 제한(override)")
    d1, d2, d3 = st.columns([2, 1, 1])
    lim_email = d1.text_input("제한 변경 대상 이메일", placeholder="someone@example.com", key="lim_email")
    daily_lim = d2.number_input("일 제한", min_value=0, value=int(user.get("daily_limit", DEFAULT_DAILY_LIMIT)), step=1)
    monthly_lim = d3.number_input("월 제한", min_value=0, value=int(user.get("monthly_limit", DEFAULT_MONTHLY_LIMIT)), step=1)

    if st.button("제한 저장", use_container_width=True):
        le = norm_email(lim_email)
        if not valid_email(le):
            st.error("이메일이 올바르지 않습니다.")
        else:
            db_create_user_if_missing(le)
            db_update_limits(le, int(daily_lim), int(monthly_lim))
            st.success("저장 완료")
            st.rerun()

    st.divider()


# -----------------------------
# 제안서 생성 UI (승인된 사람만)
# -----------------------------
st.subheader("📝 제안서 생성")

quota = get_remaining_quota(user)
if quota["remain_today"] <= 0 or quota["remain_month"] <= 0:
    st.error("사용량 한도 초과입니다. (비용 방어) 관리자에게 한도 상향을 요청하세요.")
    st.stop()

left, right = st.columns([1.1, 0.9])

with left:
    st.markdown("#### 입력")
    company = st.text_input("회사명", placeholder="예: (주)엠스페이스")
    industry = st.text_input("업종/업태", placeholder="예: 제조업(금속가공)")
    sales_last = st.text_input("작년 매출", placeholder="예: 8억")
    sales_this = st.text_input("금년 예상 매출", placeholder="예: 10억")
    employees = st.text_input("직원 수(대표 제외)", placeholder="예: 6명")

    pains = st.text_area(
        "현재 고민/리스크(선택)",
        placeholder="예: 성실신고 대상 우려, 건강보험료 증가, 비용 증빙 취약, 세무조사 리스크 등",
        height=120,
    )

    tone = st.selectbox("문서 톤", ["전문적/숫자중심/리스크체감형", "간결한 요약형", "영업설득형"], index=0)

with right:
    st.markdown("#### 생성 설정")
    model = st.text_input("OpenAI 모델", value=OPENAI_MODEL)
    max_len = st.slider("길이(대략)", 600, 2400, 1400, 100)

    st.caption("※ 승인된 사용자만 생성 가능 / 생성 시 사용량 1회 차감")

generate = st.button("🚀 제안서 생성(OpenAI)", type="primary", use_container_width=True)

if generate:
    if not company:
        st.error("회사명은 필수입니다.")
        st.stop()

    # 다시 한 번 사용량 체크(클릭 중복 방어)
    quota = get_remaining_quota(user)
    if quota["remain_today"] <= 0 or quota["remain_month"] <= 0:
        st.error("사용량 한도 초과입니다.")
        st.stop()

    system = (
        "너는 '개인사업자 성실신고 리스크 및 법인전환 전략 컨설팅' 전문가다. "
        "과장 없이, 숫자 중심, 리스크 체감형으로 '제안서 원고'를 작성한다. "
        "구성: 1)요약 2)현황/가정 3)리스크(성실신고/세무조사/건보) 4)대응전략(증빙/구조개편/법인전환) "
        "5)3년 관점 비용/리스크 포인트 6)1차 미팅 클로징 멘트. "
        "표가 필요하면 markdown 표로 제시."
    )

    user_input = f"""
[고객 정보]
- 회사명: {company}
- 업종: {industry}
- 작년 매출: {sales_last}
- 금년 예상 매출: {sales_this}
- 직원 수: {employees}
- 추가 고민/리스크: {pains}

[요청 톤]
{tone}

[출력 길이 가이드]
약 {max_len}자 내외
""".strip()

    try:
        with st.spinner("OpenAI로 제안서 생성 중..."):
            # Responses API (공식 권장) :contentReference[oaicite:5]{index=5}
            resp = oai().responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_input},
                ],
            )
            text = resp.output_text

        # 사용량 차감(일/월 각각 +1)
        daily_key, monthly_key = period_keys()
        usage_increment(user["email"], "daily", daily_key, 1)
        usage_increment(user["email"], "monthly", monthly_key, 1)

        st.success("생성 완료 (사용량 1회 차감)")
        st.markdown("### ✅ 생성된 제안서(초안)")
        st.markdown(text)

        # 다운로드(마크다운)
        st.download_button(
            "⬇️ 제안서(.md) 다운로드",
            data=text.encode("utf-8"),
            file_name=f"proposal_{company}_{daily_key}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"OpenAI 호출 실패: {e}")












