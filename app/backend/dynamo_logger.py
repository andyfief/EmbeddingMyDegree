"""
dynamo_logger.py — Fire-and-forget query logging to DynamoDB.
Errors are printed to stdout but never propagate to the caller.
"""

import uuid
import boto3
from datetime import datetime, timezone


_table = None


def _get_table(table_name: str, region: str):
    global _table
    if _table is None:
        dynamodb = boto3.resource("dynamodb", region_name=region)
        _table   = dynamodb.Table(table_name)
    return _table


def log_query(
    query: str,
    k: int,
    latency_ms: float,
    results: list[dict],
    table_name: str,
    region: str,
) -> None:
    """Write one search event to DynamoDB. Swallows all exceptions."""
    try:
        table = _get_table(table_name, region)
        table.put_item(Item={
            "query_id":   str(uuid.uuid4()),
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "query":      query,
            "k":          k,
            "latency_ms": round(latency_ms, 2),
            "results": [
                {
                    "rank":      r["rank"],
                    "file_path": r["file_path"],
                    "score":     str(round(r["score"], 6)),  # Decimal-safe
                }
                for r in results
            ],
        })
    except Exception as e:
        print(f"[dynamo_logger] WARNING: failed to log query: {e}")
