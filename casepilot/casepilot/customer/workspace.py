"""Public synthetic exploration catalog and authorized current-session profile."""

import json
from pathlib import Path

from casepilot.customer.views import decoded, status_label


def load_samples():
    path = Path(__file__).resolve().parents[2] / "datasets/customer_cases/samples.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    if content.get("synthetic") is not True:
        raise ValueError("Only public synthetic sample data is permitted")
    return content["cases"]


def filter_samples(samples, query, status):
    return [
        case
        for case in samples
        if (status == "All" or status_label(case["status"]) == status)
        and matches_query(case, query)
    ]


def matches_query(value, query):
    haystack = json.dumps(value, ensure_ascii=False).casefold()
    return all(term in haystack for term in query.strip().casefold().split())


def profile_view(state, receipts):
    if not state.customer_id:
        return None
    for receipt in reversed(list(receipts.values())):
        value = decoded(receipt)
        if (
            isinstance(receipt, dict)
            and receipt.get("tool_name") in {"get_user_details", "modify_user_address"}
            and isinstance(value, dict)
            and value.get("user_id") == state.customer_id
            and isinstance(receipt.get("arguments"), dict)
            and receipt["arguments"].get("user_id") == state.customer_id
        ):
            name = value.get("name", {})
            return {
                "Customer ID": state.customer_id,
                "Email": value.get("email", ""),
                "Full name": " ".join(str(name.get(k, "")) for k in ("first_name", "last_name"))
                if isinstance(name, dict)
                else "",
                "ZIP code": value.get("address", {}).get("zip", "")
                if isinstance(value.get("address"), dict)
                else "",
            }
    return None
