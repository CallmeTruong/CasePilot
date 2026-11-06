"""Bounded arithmetic, customer scope checks and visible-secret scrubbing.

Pattern checks are defense in depth. Authorization is always enforced by the existing policy gate.
"""

import ast
import json
import math
import operator
import re
import unicodedata

SCOPE_MESSAGE = (
    "I support Retail requests such as orders, prices, returns/exchanges and payments. I "
    "only calculate when working with verified price or order information."
)
SECURITY_MESSAGE = (
    "I cannot bypass rules, disclose internal or secret information, or perform "
    "unauthorized actions. I can still help you with your own orders."
)
FINANCIAL = re.compile(
    r"giá|\bgia\b|price|usd|cost|refund|hoàn tiền|hoan tien|chênh lệch|chenh lech|"
    r"tổng tiền|tong tien|tổng giá|\btotal\b|thanh toán|thanh toan|số tiền|so tien|"
    r"đổi|\bdoi\b|exchange",
    re.I,
)
ARITHMETIC = re.compile(r"[\d\s+*/().=\-]+")
KEY_PATTERN = re.compile(r"\b(?:sk-(?:ant-)?[A-Za-z0-9_-]{8,}|Bearer\s+[A-Za-z0-9_.-]{8,})", re.I)


def validate_calculation(expression) -> bool:
    if not isinstance(expression, str) or not 1 <= len(expression) <= 256:
        return False
    if not re.fullmatch(r"[0-9+*/(). \-]+", expression):
        return False
    depth = 0
    for char in expression:
        depth += (char == "(") - (char == ")")
        if not 0 <= depth <= 12:
            return False
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def value(node, level=0):
        if level > 12:
            raise ValueError("depth")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = float(node.value)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            result = value(node.operand, level + 1) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            result = operations[type(node.op)](
                value(node.left, level + 1), value(node.right, level + 1)
            )
        else:
            raise ValueError("operator")
        if not math.isfinite(result) or abs(result) > 1e12:
            raise ValueError("magnitude")
        return result

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        if len(list(ast.walk(tree))) > 64:
            return False
        value(tree.body)
        return True
    except (ValueError, SyntaxError, ArithmeticError, RecursionError):
        return False


def normalized(text):
    return (
        re.sub(
            r"\s+",
            " ",
            re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", unicodedata.normalize("NFKC", text)),
        )
        .strip()
        .casefold()
    )


def request_block(text) -> str | None:
    text = normalized(text)
    if re.search(
        r"ignore .{0,60}(?:previous|prior|system|rules)|"
        r"bỏ qua .{0,60}(?:quy tắc|hướng dẫn|chỉ dẫn)|"
        r"bo qua .{0,60}(?:quy tac|huong dan)|<\s*/?\s*(?:system|developer)\b|"
        r"(?:^|\n)(?:system|developer)\s*:",
        text,
    ):
        return "prompt_injection"
    if re.search(
        r"(?:show|print|dump|reveal|display|give|in ra|tiết lộ|tiet lo|cho.{0,20}xem).{0,90}"
        r"(?:system prompt|developer prompt|instructions|api.?key|secret|password|"
        r"mật khẩu|mat khau|prompt hệ thống)",
        text,
    ):
        return "sensitive_data"
    math_text = re.sub(
        r"^(?:hãy |hay |please )?(?:tính|tinh|calculate|compute|solve|what is)\s*", "", text
    )
    math_text = math_text.rstrip(" ?.!")
    if ARITHMETIC.fullmatch(math_text) and re.search(r"\d", math_text):
        return "scope"
    return None


def scrub_text(text, secrets=()) -> str:
    text = str(text or "")
    for secret in sorted((s for s in secrets if s), key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    return KEY_PATTERN.sub("[REDACTED]", text)


class CustomerSafetyGate:
    def __init__(self, base, receipts):
        self.base, self.receipts = base, receipts
        self.request = ""
        self.inputs = []

    def observe_request(self, text):
        self.request = text
        self.inputs.append(normalized(text))
        self.inputs = self.inputs[-32:]

    def __getattr__(self, name):
        return getattr(self.base, name)

    def check(self, name, arguments, state):
        from casepilot.agents.verifier import GateResult

        verdict = self.base.check(name, arguments, state)
        if not verdict.allowed:
            return verdict
        provided = " ".join(self.inputs)
        if name == "find_user_id_by_email":
            emails = re.findall(r"[\w.+%-]+@[\w.-]+\.[a-z]{2,}", provided)
            if normalized(arguments["email"]) not in emails:
                return GateResult(
                    False, "identity_evidence", "Please provide the identity details yourself."
                )
        if name == "find_user_id_by_name_zip":
            if not all(
                re.search(r"(?<!\w)" + re.escape(normalized(arguments[key])) + r"(?!\w)", provided)
                for key in ("first_name", "last_name", "zip")
            ):
                return GateResult(
                    False, "identity_evidence", "Please provide the identity details yourself."
                )
        if name != "calculate":
            return verdict
        if not FINANCIAL.search(normalized(self.request)) or not state.customer_id:
            return GateResult(False, "scope", SCOPE_MESSAGE)
        for receipt in self.receipts.values():
            if receipt["tool_name"] != "get_order_details" or receipt["result"].get("error"):
                continue
            try:
                order = json.loads(receipt["result"]["content"])
            except (ValueError, TypeError, KeyError):
                continue
            if (
                isinstance(order, dict)
                and order.get("user_id") == state.customer_id
                and order.get("order_id") in state.owned_order_ids
            ):
                return verdict
        return GateResult(False, "scope", SCOPE_MESSAGE)
