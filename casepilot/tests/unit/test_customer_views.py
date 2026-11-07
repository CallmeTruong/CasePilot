import json

import pytest

from casepilot.agents.state import CaseState


def receipt(name, arguments, value, error=False):
    return {
        "tool_name": name,
        "arguments": arguments,
        "result": {"content": json.dumps(value), "error": error},
    }


def test_private_cards_require_verified_customer_and_matching_receipts():
    from casepilot.customer.views import project_orders

    state = CaseState(episode_id="ep")
    receipts = {
        "p": receipt("get_user_details", {"user_id": "own"}, {"user_id": "own", "orders": ["#A"]}),
        "o": receipt(
            "get_order_details",
            {"order_id": "#A"},
            {
                "order_id": "#A",
                "user_id": "own",
                "status": "return requested",
                "items": [{"name": "Áo", "price": 20}],
            },
        ),
    }
    assert project_orders(state, receipts) == []
    state.customer_id = "other"
    assert project_orders(state, receipts) == []
    state.customer_id = "own"
    state.owned_order_ids = {"#A"}
    cards = project_orders(state, receipts)
    assert cards[0]["status"] == "Return requested"
    assert cards[0]["items"][0]["name"] == "Áo"
    assert cards[0]["total"] == "20.00 USD"
    receipts["o"]["result"]["error"] = True
    assert project_orders(state, receipts)[0]["items"] == []


def test_mutation_receipt_updates_order_status_without_guessing():
    from casepilot.customer.views import project_orders

    state = CaseState(episode_id="ep", customer_id="own", owned_order_ids={"#A"})
    receipts = {
        "p": receipt("get_user_details", {"user_id": "own"}, {"user_id": "own", "orders": ["#A"]}),
        "o": receipt(
            "get_order_details",
            {"order_id": "#A"},
            {"order_id": "#A", "user_id": "own", "status": "pending"},
        ),
        "mutation": receipt(
            "cancel_pending_order",
            {"order_id": "#A"},
            {"order_id": "#A", "user_id": "own", "status": "cancelled"},
        ),
    }
    assert project_orders(state, receipts)[0]["status"] == "Cancelled"


def test_customer_copy_removes_technical_payloads_and_preserves_truthful_status():
    from casepilot.customer.views import customer_text, status_label

    assert "POLICY_" not in customer_text("Theo chính sách [POLICY_abc123], được trả hàng.")
    assert "###STOP###" not in customer_text("Dừng. ###STOP###")
    assert "<script>" not in customer_text("<script>alert(1)</script>Xin chào")
    assert "api_key" not in customer_text('{"api_key": "test-only"}')
    assert "api_key" not in customer_text(
        'Debug get_user_details returned {"api_key":"test-only"}; model=openai/test'
    )
    assert status_label("return requested") != "Refunded"
    assert status_label("unknown-status") == "Verifying status"


def test_confirmation_shows_observed_variant_and_price_difference_only_for_current_customer():
    from casepilot.agents.state import Confirmation, fingerprint
    from casepilot.customer.views import confirmation_view

    state = CaseState(episode_id="ep", customer_id="own", owned_order_ids={"#A"})
    args = {"order_id": "#A", "item_ids": ["old"], "new_item_ids": ["new"]}
    name = "exchange_delivered_order_items"
    state.confirmation = Confirmation(
        episode_id="ep",
        fingerprint=fingerprint(name, args, "own", 0, "ep"),
        tool_name=name,
        arguments=args,
        plan_revision=0,
        user_turn=1,
    )
    receipts = {
        "p": receipt("get_user_details", {"user_id": "own"}, {"user_id": "own", "orders": ["#A"]}),
        "o": receipt(
            "get_order_details",
            {"order_id": "#A"},
            {
                "order_id": "#A",
                "user_id": "own",
                "status": "delivered",
                "items": [{"item_id": "old", "name": "Áo", "price": 20}],
            },
        ),
        "product": receipt(
            "get_product_details",
            {"product_id": "product"},
            {
                "name": "Áo",
                "variants": {
                    "old": {"price": 22},
                    "new": {"price": 25, "options": {"color": "red"}},
                },
            },
        ),
    }
    view = confirmation_view(state, receipts)
    assert "red" in str(view["details"]) and "5.00 USD" in str(view["details"])
    state.customer_id = "different"
    assert confirmation_view(state, receipts) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("orders", [["#A"]]),
        ("orders", None),
        ("items", None),
        ("status", {"bad": "shape"}),
        ("envelope", None),
    ],
)
def test_malformed_receipts_do_not_crash_customer_cards(field, value):
    from casepilot.customer.views import project_orders

    state = CaseState(episode_id="ep", customer_id="own", owned_order_ids={"#A"})
    profile = {"user_id": "own", "orders": ["#A"]}
    order = {"order_id": "#A", "user_id": "own", "status": "pending", "items": []}
    if field == "orders":
        profile[field] = value
    elif field != "envelope":
        order[field] = value
    receipts = {
        "p": receipt("get_user_details", {"user_id": "own"}, profile),
        "o": receipt("get_order_details", {"order_id": "#A"}, order),
    }
    if field == "envelope":
        receipts["o"] = value
    assert isinstance(project_orders(state, receipts), list)
