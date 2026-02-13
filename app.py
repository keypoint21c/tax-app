import os
import re
from datetime import datetime
from typing import Optional, Dict, Any, List

import streamlit as st

# Supabase (supabase-py)
from supabase import create_client, Client


# =========================================================
# 0) Streamlit 기본 설정
# =========================================================
st.set_page_config(page_title="성실신고 리스크 & 법인전환 분석", layout="wide")


# =========================================================
# 1) Secrets 로드 (Streamlit Cloud / Local 모두 지원)
#    - Streamlit Cloud: st.secrets 사용
#    - Local: 환경변수 사용
# =========================================================
def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    # Streamlit Cloud secrets
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    # env var
    return os.getenv(key, default)


SUPABASE_URL = _get_secret("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = _get_secret("SUPABASE_SERVICE_ROLE_KEY")
ADMIN_BOOTSTRAP_KEY = _get_secret("ADMIN_BOOTSTRAP_KEY")


def secrets_ready() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and ADMIN_BOOTSTRAP_KEY)


# =========================================================
# 2) Supabase Client
# =========================================================
@st.cache_resource(show_spinner=False)
def get_sb() -> Optional[Client]:
    if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY):
        return None
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# =========================================================
# 3) 유틸
# =========================================================
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def is_valid_email(email: str) -> bool:
    return bool(EMAIL_RE.match(normalize_email(email)))


def now_iso() -> str:
    return datetime.utcnow().isoformat()


# =========================================================
# 4) DB 함수 (users 테이블)
#    users(email text pk, approved boolean, is_admin boolean, created_at timestamp)
# =========================================================
def db_get_user(sb: Client, email: str) -> Optional[Dict[str, Any]]:
    email = normalize_email(email)
    if not email:
        return None
    res = sb.table("users").select("*").eq("email", email).limit(1).execute()
    data = res.data or []
    return data[0] if data else None


def db_upsert_user(sb: Client, email: str, approved: bool = False, is_admin: bool = False) -> None:
    email = normalize_email(email)
    payload = {
        "email": email,
        "approved": bool(approved),
        "is_admin": bool(is_admin),
    }
    # created_at 컬럼이 있다면 서버 기본값으로 두는게 더 깔끔하지만,
    # 없을 수도 있어서 안전하게 넣지 않습니다.
    sb.table("users").upsert(payload).execute()


def db_set_approval(sb: Client, email: str, approved: bool) -> None:
    email = normalize_email(email)
    sb.table("users").update({"approved": bool(approved)}).eq("email", email).execute()


def db_set_admin(sb: Client, email: str, is_admin: bool) -> None:
    email = normalize_email(email)
    sb.table("users").update({"is_admin": bool(is_admin)}).eq("email", email).execute()


def db_list_users(sb: Client) -> List[Dict[str, Any]]:
    res = sb.table("users").select("*").order("email").execute()
    return res.data or []


def db_list_pending(sb: Client) -> List[Dict[str, Any]]:
    res = sb.table("users").select("*").eq("approved", False).order("email").execute()
    return res.data or []


# =========================================================
# 5) 세션 상태
# =========================================================
if "logged_in_email" not in st.session_state:
    st.session_state.logged_in_email = ""
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "approved" not in st.session_state:
    st.session_state.approved = False


def refresh_login_state(sb: Client) -> None:
    email = normalize_email(st.session_state.logged_in_email)
    if not email:
        st.session_state.is_admin = False
        st.session_state.approved = False
        return

    user = db_get_user(sb, email)
    if not user:
        st.session_state.is_admin = False
        st.session_state.approved = False
        return

    st.session_state.is_admin = bool(user.get("is_admin", False))
    st.session_state.approved = bool(user.get("approved", False))


# =========================================================
# 6) UI: 상단 타이틀
# =========================================================
st.title("📊 개인사업자 성실신고 리스크 & 법인전환 전략 분석 (Streamlit)")
st.caption("※ 본 앱은 참고용이며 실제 세무/법률 판단은 전문가 검토가 필요합니다.")


# =========================================================
# 7) Secrets / DB 연결 체크
# =========================================================
sb = get_sb()

if sb is None:
    st.error(
        "Supabase 연결 정보가 없습니다.\n\n"
        "Streamlit Cloud → 앱 Settings → Secrets 에 아래 키를 넣어주세요:\n"
        "- SUPABASE_URL\n"
        "- SUPABASE_SERVICE_ROLE_KEY\n"
        "- ADMIN_BOOTSTRAP_KEY\n"
    )
    st.stop()

# users 테이블 존재 여부를 간단히 체크(없으면 에러나므로 안내)
try:
    _ = sb.table("users").select("email").limit(1).execute()
except Exception as e:
    st.error(
        "Supabase에 `users` 테이블이 없거나 접근이 막혀 있습니다.\n\n"
        "1) Supabase SQL Editor에서 users 테이블 생성\n"
        "2) SERVICE_ROLE_KEY를 Secrets에 넣었는지 확인\n\n"
        f"에러: {e}"
    )
    st.stop()


# =========================================================
# 8) 좌측 사이드바: 로그인 / 관리자 초기설정
# =========================================================
with st.sidebar:
    st.header("🔒 접근 제어")

    # (1) 로그인
    email_input = st.text_input("이메일", value=st.session_state.logged_in_email, placeholder="name@example.com")

    colA, colB = st.columns(2)
    with colA:
        if st.button("로그인", use_container_width=True):
            email = normalize_email(email_input)
            if not is_valid_email(email):
                st.warning("이메일 형식이 올바르지 않습니다.")
            else:
                st.session_state.logged_in_email = email
                # 최초 로그인 시 사용자 등록(승인 false)
                if db_get_user(sb, email) is None:
                    db_upsert_user(sb, email, approved=False, is_admin=False)
                refresh_login_state(sb)

    with colB:
        if st.button("로그아웃", use_container_width=True):
            st.session_state.logged_in_email = ""
            st.session_state.is_admin = False
            st.session_state.approved = False

    # 현재 상태 표시
    if st.session_state.logged_in_email:
        refresh_login_state(sb)
        if st.session_state.is_admin:
            st.success(f"관리자 로그인: {st.session_state.logged_in_email}")
        elif st.session_state.approved:
            st.success(f"승인됨: {st.session_state.logged_in_email}")
        else:
            st.info("등록되었습니다. 관리자 승인 대기 중입니다.")

    st.divider()

    # (2) 관리자 초기설정(최초 1회)
    with st.expander("🛠 관리자 초기설정(최초 1회)", expanded=False):
        st.caption("ADMIN_BOOTSTRAP_KEY를 아는 사람만 관리자 권한을 부여할 수 있습니다.")
        bootstrap_key = st.text_input("ADMIN_BOOTSTRAP_KEY", type="password")
        admin_email = st.text_input("관리자로 만들 이메일", placeholder="admin@example.com")

        if st.button("관리자 계정 생성/갱신", use_container_width=True):
            if not ADMIN_BOOTSTRAP_KEY:
                st.error("Secrets에 ADMIN_BOOTSTRAP_KEY가 없습니다.")
            elif bootstrap_key != ADMIN_BOOTSTRAP_KEY:
                st.error("키가 일치하지 않습니다.")
            else:
                email = normalize_email(admin_email)
                if not is_valid_email(email):
                    st.warning("관리자 이메일 형식이 올바르지 않습니다.")
                else:
                    db_upsert_user(sb, email, approved=True, is_admin=True)
                    st.success(f"관리자 지정 완료: {email}")
                    # 내가 방금 그 이메일이면 즉시 갱신
                    if normalize_email(st.session_state.logged_in_email) == email:
                        refresh_login_state(sb)


# =========================================================
# 9) 승인/권한 체크
# =========================================================
if not st.session_state.logged_in_email:
    st.info("왼쪽 사이드바에서 이메일로 로그인하세요.")
    st.stop()

refresh_login_state(sb)

if not (st.session_state.approved or st.session_state.is_admin):
    st.warning("승인된 사용자만 이용 가능합니다. 현재는 관리자 승인 대기 상태입니다.")
    st.stop()


# =========================================================
# 10) 메인 기능 (여기서는 예시로 입력만 유지 — 당신 기존 로직을 아래에 붙이면 됨)
# =========================================================
st.subheader("1) 입력")
col1, col2, col3, col4 = st.columns(4)

with col1:
    industry_code = st.text_input("산업분류코드(F열)", value="25934")
with col2:
    last_sales = st.number_input("작년 매출(원)", min_value=0, step=1000000, value=800000000)
with col3:
    this_sales = st.number_input("금년 예상 매출(원)", min_value=0, step=1000000, value=1000000000)
with col4:
    emp_cnt = st.number_input("직원 수(대표 제외)", min_value=0, step=1, value=6)

st.button("✅ 보고서 생성", type="primary")


# =========================================================
# 11) 관리자 화면: 승인관리
# =========================================================
if st.session_state.is_admin:
    st.divider()
    st.subheader("🛡 관리자: 승인/권한 관리")

    pending = db_list_pending(sb)
    users = db_list_users(sb)

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### 승인 대기 목록")
        if not pending:
            st.success("승인 대기 사용자가 없습니다.")
        else:
            for u in pending:
                email = u["email"]
                c1, c2, c3 = st.columns([3, 1, 1])
                c1.write(email)
                if c2.button("승인", key=f"appr_{email}"):
                    db_set_approval(sb, email, True)
                    st.rerun()
                if c3.button("삭제", key=f"del_{email}"):
                    # Supabase에서 delete 허용(서비스키면 가능)
                    sb.table("users").delete().eq("email", email).execute()
                    st.rerun()

    with right:
        st.markdown("### 전체 사용자")
        if not users:
            st.info("사용자가 없습니다.")
        else:
            for u in users:
                email = u["email"]
                approved = bool(u.get("approved", False))
                is_admin = bool(u.get("is_admin", False))

                c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                c1.write(email)
                c2.write("✅" if approved else "⏳")
                c3.write("👑" if is_admin else "")
                if c4.button("관리", key=f"manage_{email}"):
                    st.session_state["_manage_email"] = email
                    st.rerun()

        manage_email = st.session_state.get("_manage_email")
        if manage_email:
            u = db_get_user(sb, manage_email)
            if u:
                st.markdown("---")
                st.markdown(f"#### 관리: {manage_email}")

                new_approved = st.checkbox("승인(approved)", value=bool(u.get("approved", False)))
                new_admin = st.checkbox("관리자(is_admin)", value=bool(u.get("is_admin", False)))

                if st.button("저장", key="save_user"):
                    db_set_approval(sb, manage_email, new_approved)
                    db_set_admin(sb, manage_email, new_admin)
                    st.success("저장 완료")
                    st.rerun()
            else:
                st.warning("사용자를 찾을 수 없습니다.")






