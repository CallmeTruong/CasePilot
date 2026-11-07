"""Opt-in live verification of language switching in the real customer session."""

import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Allow four provider conversations")
    parser.add_argument("--output", type=Path, default=Path("results/language/live.json"))
    args = parser.parse_args()
    if not args.live:
        parser.error("Provider calls require --live")
    from casepilot.customer.factory import create_session

    session = create_session()
    original = session.agent.engine.generate

    def bounded(**kwargs):
        return original(
            **(kwargs | {"timeout": 30, "num_retries": 0, "max_completion_tokens": 256})
        )

    session.agent.engine.generate = bounded
    cases = [
        ("en", "Hello, please ask me for the email you need to look up my order."),
        ("vi", "Xin chào, hãy hỏi tôi email cần dùng để tra cứu đơn hàng."),
        ("fr", "Bonjour, demandez-moi l'adresse e-mail nécessaire pour consulter ma commande."),
        ("en", "Please reply in English now. What do you need to look up my order?"),
    ]
    results = []
    try:
        for index, (language, text) in enumerate(cases):
            session.send(text, f"language-{index}")
            answer = session.transcript[-1]["content"]
            if language == "vi":
                matched = bool(re.search(r"[ăâđêôơư\u1ea0-\u1ef9]", answer, re.I))
            elif language == "fr":
                matched = bool(re.search(r"\b(?:vous|votre|adresse|commande)\b", answer, re.I))
            else:
                matched = bool(
                    re.search(r"\b(?:your|please|email|order)\b", answer, re.I)
                ) and not bool(re.search(r"[ăâđêôơư\u1ea0-\u1ef9]", answer, re.I))
            results.append({"language": language, "reply": answer, "matched": matched})
            print(f"Turn {index + 1}: language={language}, matched={matched}", flush=True)
        report = {
            "results": results,
            "llm_calls": session.state.llm_calls,
            "tool_calls": session.environment.dispatch_count,
        }
        output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0 if all(row["matched"] for row in results) else 1
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
