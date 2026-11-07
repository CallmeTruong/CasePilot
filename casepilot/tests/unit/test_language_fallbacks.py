import pytest

from casepilot.customer.language import fallback_language, localize
from casepilot.security.guards import SECURITY_MESSAGE


@pytest.mark.parametrize(
    "text,previous,expected",
    [
        ("Please check my order", "vi", "en"),
        ("hãy tính 1 + 11", "en", "vi"),
        ("Xin chào", "en", "vi"),
        ("yes", "vi", "vi"),
        ("#12345", "vi", "vi"),
        ("Hãy trả lời bằng tiếng Anh", "vi", "en"),
        ("Please reply in Vietnamese", "en", "vi"),
        ("Hủy đơn\n\nSelected Retail record: order email price", "en", "vi"),
    ],
)
def test_fallback_language_tracks_user_and_ignores_attachment(text, previous, expected):
    assert fallback_language(text, previous) == expected


def test_known_safety_messages_translate_without_touching_model_text_or_secrets():
    assert localize(SECURITY_MESSAGE, "vi").startswith("Mình không thể")
    assert localize(SECURITY_MESSAGE, "en") == SECURITY_MESSAGE
    assert localize("Bonjour, comment puis-je vous aider ?", "en").startswith("Bonjour")
    assert localize("Customer supplied text", "vi") == "Customer supplied text"
