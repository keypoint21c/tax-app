from datetime import datetime

DAILY_LIMIT = 5
MONTHLY_LIMIT = 100


def get_period_keys():
    now = datetime.utcnow()
    daily_key = now.strftime("%Y-%m-%d")
    monthly_key = now.strftime("%Y-%m")
    return daily_key, monthly_key


def get_usage(email, period_type, period_key):
    res = (
        sb()
        .table("usage_counters")
        .select("*")
        .eq("email", email)
        .eq("period_type", period_type)
        .eq("period_key", period_key)
        .execute()
    )

    if res.data:
        return res.data[0]["used_count"]
    return 0


def increment_usage(email):
    daily_key, monthly_key = get_period_keys()

    # 🔥 DAILY UPSERT
    sb().table("usage_counters").upsert(
        {
            "email": email,
            "period_type": "daily",
            "period_key": daily_key,
            "used_count": get_usage(email, "daily", daily_key) + 1,
        },
        on_conflict="email,period_type,period_key",
    ).execute()

    # 🔥 MONTHLY UPSERT
    sb().table("usage_counters").upsert(
        {
            "email": email,
            "period_type": "monthly",
            "period_key": monthly_key,
            "used_count": get_usage(email, "monthly", monthly_key) + 1,
        },
        on_conflict="email,period_type,period_key",
    ).execute()


def check_limits(email):
    daily_key, monthly_key = get_period_keys()

    daily_used = get_usage(email, "daily", daily_key)
    monthly_used = get_usage(email, "monthly", monthly_key)

    if daily_used >= DAILY_LIMIT:
        return False, "오늘 사용 한도를 초과했습니다."

    if monthly_used >= MONTHLY_LIMIT:
        return False, "이번 달 사용 한도를 초과했습니다."

    return True, ""











