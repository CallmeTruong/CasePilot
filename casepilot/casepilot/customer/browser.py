"""Read-only projections of the local synthetic project data, separate from agent authority."""

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from casepilot.integration.benchmark import load_manifest, verify_source

ROOT = Path(__file__).resolve().parents[2]
TABLES = {"users": "Customers", "orders": "Orders", "products": "Products", "variants": "SKUs"}


@lru_cache(maxsize=1)
def _initial():
    path = ROOT / "configs/benchmark-manifest.json"
    root = verify_source(load_manifest(path), path)
    return json.loads((root / "data/tau2/domains/retail/db.json").read_text(encoding="utf-8"))


def load_initial() -> dict:
    return deepcopy(_initial())


def snapshot(session=None) -> dict:
    environment = getattr(session, "environment", None)
    if environment is not None and not session.closed:
        return environment.snapshot()
    return load_initial()


def _records(db, table):
    if table not in TABLES:
        raise ValueError("Unsupported data table")
    if table == "variants":
        return {
            key: dict(value, product_id=p["product_id"], name=p["name"])
            for p in db["products"].values()
            for key, value in p["variants"].items()
        }
    return db[table]


def table_rows(db: dict, table: str) -> list[dict]:
    rows = []
    for key, value in _records(db, table).items():
        row = {"id": key}
        if table == "users":
            row.update(
                name=" ".join(value["name"].values()),
                email=value["email"],
                city=value["address"]["city"],
                zip=value["address"]["zip"],
                orders=len(value["orders"]),
            )
        elif table == "orders":
            row.update(
                user_id=value["user_id"],
                email=db["users"][value["user_id"]]["email"],
                status=value["status"],
                items=len(value["items"]),
                total_usd=round(sum(i["price"] for i in value["items"]), 2),
            )
        elif table == "products":
            row.update(name=value["name"], variants=len(value["variants"]))
        else:
            row.update(
                product_id=value["product_id"],
                name=value["name"],
                available=value["available"],
                price_usd=value["price"],
                options=" · ".join(value["options"].values()),
            )
        rows.append(row)
    return rows


def query_rows(rows, query="", filters=None, sort_by="id", descending=False) -> list[dict]:
    terms = query.strip().casefold().split()
    matches = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in (filters or {}).items())
        and all(term in json.dumps(row, ensure_ascii=False).casefold() for term in terms)
    ]
    return sorted(matches, key=lambda row: row.get(sort_by, ""), reverse=descending)


def record_detail(db: dict, table: str, record_id: str) -> dict:
    record = _records(db, table).get(record_id)
    if record is None:
        raise ValueError("Record no longer exists in this snapshot")
    related = []
    if table == "users":
        related = [{"table": "orders", "id": key} for key in record["orders"]]
    elif table == "orders":
        related = [{"table": "users", "id": record["user_id"]}]
        related += [{"table": "products", "id": i["product_id"]} for i in record["items"]]
        related += [{"table": "variants", "id": i["item_id"]} for i in record["items"]]
    elif table == "products":
        related = [{"table": "variants", "id": key} for key in record["variants"]]
    else:
        related = [{"table": "products", "id": record["product_id"]}]
    return {"record": deepcopy(record), "related": related}


def record_context(db: dict, table: str, record_id: str) -> str:
    record = record_detail(db, table, record_id)["record"]
    context = {"table": table, "id": record_id}
    if table in {"users", "orders"}:
        user = record if table == "users" else db["users"][record["user_id"]]
        context.update(user_id=user["user_id"], email=user["email"])
    return "Selected Retail record: " + json.dumps(context, ensure_ascii=False)


@lru_cache(maxsize=1)
def _sources():
    from casepilot.rag.ingestion import load_policy

    sources = {
        name: [
            json.loads(line)
            for line in (ROOT / "datasets" / filename).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for name, filename in {
            "rag_eval": "rag_eval/queries.jsonl",
            "fault_cases": "fault_cases/scenarios.jsonl",
            "memory_cases": "memory_cases/scenarios.jsonl",
        }.items()
    }
    sources["policy"] = [
        c.model_dump(mode="json") for c in load_policy(ROOT / "configs/benchmark-manifest.json")
    ]
    return sources


def project_sources() -> dict[str, list[dict]]:
    return deepcopy(_sources())
