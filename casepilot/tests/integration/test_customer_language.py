from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.integration.test_customer_session import build, call, message


def test_workspace_controls_and_statuses_are_english():
    from casepilot.customer.views import status_label

    page = AppTest.from_file(Path("dashboard/app.py").resolve()).run(timeout=45)
    assert not page.exception
    assert any(button.label == "New session" for button in page.button)
    assert page.text_input(key="browser_query").label == "Search table"
    assert page.chat_input[0].placeholder == "Ask the agent…"
    assert status_label("cancelled") == "Cancelled"
    page.selectbox(key="browser_source").select("policy").run(timeout=45)
    assert not page.exception
    assert page.text_input(key="project_query_policy").label == "Search source"


def test_local_refusals_follow_user_language_without_provider_calls(retail_environment, tmp_path):
    session = build(retail_environment, [], tmp_path)
    try:
        session.send("Calculate 1 + 11", "english")
        assert session.transcript[-1]["content"].startswith("I support Retail")
        session.send("hãy tính 1 + 11", "vietnamese")
        assert session.transcript[-1]["content"].startswith("Mình hỗ trợ")
        session.send("Please calculate 2 + 2", "english-again")
        assert session.transcript[-1]["content"].startswith("I support Retail")
        assert session.state.llm_calls == session.environment.dispatch_count == 0
    finally:
        session.close()


@pytest.mark.parametrize(
    "user_text,prepared,declined",
    [
        ("Cancel my order", "I have prepared", "I have not made"),
        ("Hủy đơn của tôi", "Mình đã chuẩn bị", "Mình chưa thực hiện"),
    ],
)
def test_confirmation_card_stays_english_and_chat_follows_user(
    retail_environment, tmp_path, user_text, prepared, declined
):
    from casepilot.customer.views import confirmation_view

    order = next(o for o in retail_environment.tools.db.orders.values() if o.status == "pending")
    session = build(
        retail_environment,
        [call("cancel_pending_order", order_id=order.order_id, reason="no longer needed")],
        tmp_path,
    )
    session.state.customer_id = order.user_id
    session.state.owned_order_ids = {order.order_id}
    try:
        session.send(user_text, "proposal")
        assert session.transcript[-1]["content"].startswith(prepared)
        card = confirmation_view(session.state, session.receipts)
        assert card["title"] == "Cancel order"
        assert card["notice"].startswith("This action runs only after you confirm")
        session.decide(False, session.confirmation_token, "decline")
        assert session.transcript[-1]["content"].startswith(declined)
        assert order.status == "pending"
        assert session.environment.dispatch_count == 0
    finally:
        session.close()


def test_customer_model_instruction_matches_user_language(retail_environment, tmp_path):
    from casepilot.customer.factory import CUSTOMER_INSTRUCTION

    session = build(retail_environment, [message("Hello"), message("Bonjour")], tmp_path)
    session.agent.engine.customer_instruction = CUSTOMER_INSTRUCTION
    prompts = []
    original = session.agent.engine.generate

    def capture(**kwargs):
        prompts.append(kwargs["messages"][0].content)
        return original(**kwargs)

    session.agent.engine.generate = capture
    try:
        session.send("Hello, I need help with an order.", "en")
        session.send("Bonjour, aidez-moi avec ma commande.", "fr")
        assert session.transcript[-1]["content"] == "Bonjour"
        assert all("language" in prompt and "user" in prompt for prompt in prompts)
        assert all("Reply in natural, friendly Vietnamese" not in prompt for prompt in prompts)
        assert all("friendly Vietnamese" not in prompt for prompt in prompts)
        assert all("plain Vietnamese" not in prompt for prompt in prompts)
    finally:
        session.close()
