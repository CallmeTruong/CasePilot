"""Opt-in live synthetic customer flow. Uses API and confirms cancellation in sample data."""

import json
import uuid
from pathlib import Path

from casepilot.customer.factory import create_session
from casepilot.customer.views import project_orders
from casepilot.observability.events import redact


def main():
    folder = Path("results/customer-ui/live") / uuid.uuid4().hex
    folder.mkdir(parents=True)
    session = create_session()
    try:
        db = session.environment.snapshot()
        order = next(order for order in db["orders"].values() if order["status"] == "pending")
        fixture = {
            "order_id": order["order_id"],
            "email": db["users"][order["user_id"]]["email"],
        }
        session.send(
            f"Email đặt hàng của tôi là {fixture['email']}. Hãy xác minh và cho tôi "
            f"xem chi tiết đơn {fixture['order_id']}.",
            "lookup",
        )
        assert any(
            card["id"] == fixture["order_id"]
            for card in project_orders(session.state, session.receipts)
        )
        session.send(
            f"Tôi muốn hủy đơn {fixture['order_id']} vì không còn nhu cầu. "
            "Bạn chuẩn bị yêu cầu giúp tôi nhé.",
            "cancel",
        )
        assert session.state.confirmation and session.state.confirmation.status == "pending"
        session.decide(True, session.confirmation_token, "approve")
        assert any(
            card["id"] == fixture["order_id"] and card["status"] == "Cancelled"
            for card in project_orders(session.state, session.receipts)
        )
        assert sum(session.environment.mutation_commits.values()) == 1
        print(f"Live customer flow passed: {folder}")
        return 0
    finally:
        (folder / "flow.json").write_text(
            json.dumps(
                redact(
                    {
                        "synthetic": True,
                        "episode_id": session.state.episode_id,
                        "transcript": session.transcript,
                        "orders": project_orders(session.state, session.receipts),
                        "confirmation_status": session.state.confirmation.status
                        if session.state.confirmation
                        else None,
                        "state_status": session.state.status,
                        "mutation_commits": sum(session.environment.mutation_commits.values()),
                        "operations": session.environment.operation_records(),
                    }
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
