"""Customer projections built exclusively from successful authorized tool receipts."""

import json
import re

from casepilot.agents.state import fingerprint

STATUSES = {
    "pending": "Pending",
    "processed": "Processing",
    "delivered": "Delivered",
    "cancelled": "Cancelled",
    "return requested": "Return requested",
    "exchange requested": "Exchange requested",
    "pending (items modifed)": "Items updated",
}
ACTIONS = {
    "cancel_pending_order": "Cancel order",
    "return_delivered_order_items": "Request return",
    "exchange_delivered_order_items": "Request exchange",
    "modify_user_address": "Update your address",
    "modify_pending_order_address": "Update shipping address",
    "modify_pending_order_items": "Change order items",
    "modify_pending_order_payment": "Change payment method",
}


def customer_text(text):
    text = str(text or "")
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    try:
        if isinstance(json.loads(text), (dict, list)):
            return "I need to check more information so I can guide you accurately."
    except (ValueError, TypeError):
        pass
    text = re.sub(r"\[?POLICY_[A-Za-z0-9_]+\]?", "", text)
    if re.search(
        r"[{}]|\b(?:api_key|authorization|traceback|bearer|model\s*=|seed\s*=|"
        r"OPENAI_[A-Z_]+|[a-z]+_[a-z]+_[a-z_]+)\b",
        text,
        re.I,
    ):
        return "I need to check the information again so I can give you clear guidance."
    text = re.sub(r"<[^>]*>", "", text).replace("###STOP###", "")
    text = text.replace("**", "").replace("`", "")
    return text.strip() or "I need more information to continue helping you."


def status_label(value):
    return STATUSES.get(value, "Verifying status") if isinstance(value, str) else "Verifying status"


def decoded(receipt):
    if not isinstance(receipt, dict):
        return None
    result = receipt.get("result", {})
    if not isinstance(result, dict):
        return None
    if result.get("error"):
        return None
    try:
        return json.loads(result.get("content") or "null")
    except (ValueError, TypeError):
        return None


def project_orders(state, receipts):
    if not state.customer_id:
        return []
    authorized = set()
    for receipt in receipts.values():
        value = decoded(receipt)
        arguments = receipt.get("arguments") if isinstance(receipt, dict) else None
        if not isinstance(arguments, dict):
            continue
        if (
            receipt.get("tool_name") == "get_user_details"
            and isinstance(value, dict)
            and value.get("user_id") == state.customer_id
            and arguments.get("user_id") == state.customer_id
        ):
            identifiers = value.get("orders")
            if isinstance(identifiers, list):
                authorized.update(item for item in identifiers if isinstance(item, str))
    authorized &= state.owned_order_ids
    cards = {
        identifier: {"id": identifier, "status": "Details not viewed", "items": [], "total": None}
        for identifier in sorted(authorized)
    }
    for receipt in receipts.values():
        value = decoded(receipt)
        arguments = receipt.get("arguments") if isinstance(receipt, dict) else None
        if not isinstance(arguments, dict):
            continue
        if not isinstance(value, dict) or receipt.get("tool_name") not in {
            "get_order_details",
            *ACTIONS,
        }:
            continue
        identifier = value.get("order_id")
        if (
            not isinstance(identifier, str)
            or identifier not in cards
            or value.get("user_id") != state.customer_id
            or arguments.get("order_id") != identifier
        ):
            continue
        items = [
            {
                "id": str(item.get("item_id", "")),
                "product_id": str(item.get("product_id", "")),
                "name": str(item.get("name") or "Item"),
                "price": item.get("price"),
                "options": item.get("options", {}),
            }
            for item in (value.get("items") if isinstance(value.get("items"), list) else [])
            if isinstance(item, dict)
        ]
        prices = [item["price"] for item in items]
        total = (
            f"{sum(prices):.2f} USD"
            if prices
            and all(
                isinstance(price, (float, int)) and not isinstance(price, bool) for price in prices
            )
            else None
        )
        cards[identifier].update(status=status_label(value.get("status")), items=items, total=total)
    return list(cards.values())


def confirmation_view(state, receipts):
    confirmation = state.confirmation
    if (
        not confirmation
        or not state.customer_id
        or confirmation.status != "pending"
        or confirmation.episode_id != state.episode_id
        or confirmation.fingerprint
        != fingerprint(
            confirmation.tool_name,
            confirmation.arguments,
            state.customer_id,
            state.plan_revision,
            state.episode_id,
        )
    ):
        return None
    args = confirmation.arguments
    lines = []
    if args.get("order_id"):
        lines.append(("Order", args["order_id"]))
    owned_items = {
        item["id"]: item for card in project_orders(state, receipts) for item in card["items"]
    }
    observed = dict(owned_items)
    catalog = {}
    for receipt in receipts.values():
        value = decoded(receipt)
        if not isinstance(receipt, dict):
            continue
        if receipt.get("tool_name") == "get_product_details" and isinstance(value, dict):
            variants = value.get("variants")
            if not isinstance(variants, dict):
                continue
            for identifier, variant in variants.items():
                if isinstance(variant, dict):
                    catalog[str(identifier)] = variant | {"name": value.get("name", "Item")}
    observed.update(catalog)
    names = {
        identifier: str(item.get("name", "Item"))
        + (
            " · " + " / ".join(str(v) for v in item["options"].values())
            if isinstance(item.get("options"), dict) and item["options"]
            else ""
        )
        for identifier, item in observed.items()
    }
    for key, label in (("item_ids", "Item"), ("new_item_ids", "Replacement items")):
        if args.get(key):
            lines.append(
                (
                    label,
                    ", ".join(names.get(str(item), "Item ID " + str(item)) for item in args[key]),
                )
            )
    old_ids, new_ids = args.get("item_ids", []), args.get("new_item_ids", [])
    if old_ids and new_ids and len(old_ids) == len(new_ids):
        old_prices = [owned_items.get(str(identifier), {}).get("price") for identifier in old_ids]
        new_prices = [catalog.get(str(identifier), {}).get("price") for identifier in new_ids]
        if all(isinstance(price, (int, float)) for price in old_prices + new_prices):
            lines.append(("Item price difference", f"{sum(new_prices) - sum(old_prices):.2f} USD"))
    if args.get("reason"):
        reasons = {
            "no longer needed": "No longer needed",
            "ordered by mistake": "Ordered by mistake",
        }
        lines.append(("Reason", reasons.get(args["reason"], str(args["reason"]))))
    address = [
        str(args[key])
        for key in ("address1", "address2", "city", "state", "zip", "country")
        if args.get(key)
    ]
    if address:
        lines.append(("New address", ", ".join(address)))
    for key, label in (("payment_method_id", "Payment method"),):
        if args.get(key):
            lines.append((label, "ID ending in …" + str(args[key])[-4:]))
    return {
        "title": ACTIONS.get(confirmation.tool_name, "Confirm request"),
        "fingerprint": confirmation.fingerprint,
        "details": lines,
        "notice": (
            "This action runs only after you confirm. Amounts and conditions follow the"
            " verified information."
        ),
    }
