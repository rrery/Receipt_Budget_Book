"""User-scoped calendar totals and small, explainable spending insights."""
from datetime import date
from fastapi import HTTPException

CATEGORIES = ("식비", "카페·간식", "장보기", "교통", "쇼핑", "생활", "의료·건강", "문화·여가", "기타")


def month_bounds(month):
    try:
        start = date.fromisoformat(month + "-01")
        if len(month) != 7:
            raise ValueError()
        end = date(start.year + (start.month == 12), start.month % 12 + 1, 1)
        return start.isoformat(), end.isoformat()
    except ValueError:
        raise HTTPException(422, "올바른 월을 입력하세요.")


def infer_category(store_name, items):
    text = (store_name or "") + " " + " ".join(item.get("name", "") for item in items)
    groups = {
        "카페·간식": ("커피", "카페", "아메리카노", "라떼", "베이커리", "쿠키", "빵"),
        "장보기": ("마트", "슈퍼", "장보기", "식자재", "채소", "과일"),
        "교통": ("주유", "택시", "버스", "지하철", "교통", "주차"),
        "의료·건강": ("약국", "병원", "의원", "치과", "헬스"),
        "문화·여가": ("영화", "서점", "도서", "공연", "전시", "취미"),
        "쇼핑": ("의류", "옷", "신발", "백화점", "쇼핑"),
        "생활": ("생활", "세제", "다이소", "미용", "세탁"),
        "식비": ("식당", "식사", "점심", "저녁", "김밥", "국밥", "치킨", "분식", "음식"),
    }
    return next((category for category, words in groups.items() if any(w in text for w in words)), "기타")


def calendar_summary(store, owner, month):
    start, end = month_bounds(month)
    current = date.fromisoformat(start)
    previous_end = start
    previous_start = date(current.year - (current.month == 1), (current.month-2) % 12 + 1, 1).isoformat()
    with store.connect() as db:
        rows = db.execute("SELECT r.*,j.status job_status FROM receipts r LEFT JOIN jobs j ON j.receipt_id=r.id WHERE r.owner_id=? AND r.purchased_at>=? AND r.purchased_at<? ORDER BY r.purchased_at", (owner,start,end)).fetchall()
        prev = db.execute("SELECT coalesce(sum(total_amount),0) total,count(*) count FROM receipts WHERE owner_id=? AND purchased_at>=? AND purchased_at<?", (owner,previous_start,previous_end)).fetchone()
        memo_row = db.execute("SELECT memo FROM monthly_memos WHERE owner_id=? AND month=?", (owner,month)).fetchone()
    days, categories = {}, {}
    for row in rows:
        day = days.setdefault(row["purchased_at"], {"date": row["purchased_at"], "total": 0, "count": 0, "pending": 0, "unreviewed": 0, "unknown_amount": 0})
        day["total"] += row["total_amount"] or 0
        day["count"] += 1
        day["pending"] += row["job_status"] in {"queued", "processing"}
        day["unreviewed"] += row["status"] != "confirmed"
        day["unknown_amount"] += row["total_amount"] is None
        category = row["category"]
        categories[category] = categories.get(category, 0) + (row["total_amount"] or 0)
    total = sum(day["total"] for day in days.values())
    ordered = sorted(({"category": k, "total": v} for k,v in categories.items()), key=lambda r:(-r["total"],r["category"]))
    busiest = max(days.values(), key=lambda d:d["total"]) if days else None
    return {
        "month": month, "total": total, "count": len(rows), "days": list(days.values()),
        "reviewed_count": sum(row["status"] == "confirmed" for row in rows),
        "demo_count": sum(row["source"] == "demo" for row in rows),
        "unknown_amount_count": sum(row["total_amount"] is None for row in rows),
        "pending_count": sum(row["job_status"] in {"queued", "processing"} for row in rows),
        "categories": ordered, "top_category": ordered[0] if ordered and ordered[0]["total"] > 0 else None,
        "top_day": busiest if busiest and busiest["total"] > 0 else None,
        "previous_month": {"total": prev["total"], "count": prev["count"], "difference": total-prev["total"], "percent": round((total-prev["total"])*100/prev["total"],1) if prev["total"] else None},
        "memo": memo_row["memo"] if memo_row else "",
    }
