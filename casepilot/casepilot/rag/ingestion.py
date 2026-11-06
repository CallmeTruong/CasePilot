from casepilot.integration.benchmark import load_manifest, verify_source
from casepilot.rag.chunking import chunk_policy


def load_policy(manifest_path="configs/benchmark-manifest.json"):
    """Only the hash-pinned policy file is indexed; no task/simulator files enter retrieval."""
    manifest = load_manifest(manifest_path)
    root = verify_source(manifest, manifest_path)
    path = (root / manifest.policy_path).resolve()
    if not path.is_relative_to(root) or path.name != "policy.md":
        raise ValueError("Policy is outside the allowed corpus")
    return chunk_policy(
        path.read_text(encoding="utf-8"), source=manifest.policy_path, version=manifest.revision
    )
