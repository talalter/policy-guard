"""History and stats endpoints: GET /history, GET /history/{run_id}, GET /stats."""

import logging

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, Header, HTTPException

from backend.api.deps import get_db
from backend.models import Contradiction, HistoryDetail, HistoryItem, StatsResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/history", response_model=list[HistoryItem])
async def get_history(
    db=Depends(get_db),
    x_session_id: str | None = Header(default=None),
) -> list[HistoryItem]:
    """Return the 50 most recent detection runs for this session, newest first."""
    if db is None:
        raise HTTPException(status_code=503, detail="Persistence not available — set MONGODB_URL")
    projection = {
        "context": 1, "faithfulness_score": 1, "method_used": 1,
        "provider": 1, "contradictions": 1, "timestamp": 1,
    }
    query = {"session_id": x_session_id} if x_session_id else {}
    cursor = db.detection_runs.find(query, projection).sort("timestamp", -1).limit(50)
    items = []
    async for doc in cursor:
        items.append(HistoryItem(
            run_id=str(doc["_id"]),
            timestamp=doc["timestamp"].isoformat(),
            faithfulness_score=doc["faithfulness_score"],
            contradiction_count=len(doc.get("contradictions", [])),
            method_used=doc["method_used"],
            provider=doc["provider"],
            context_snippet=doc["context"][:100],
        ))
    return items


@router.get("/history/{run_id}", response_model=HistoryDetail)
async def get_history_item(
    run_id: str,
    db=Depends(get_db),
    x_session_id: str | None = Header(default=None),
) -> HistoryDetail:
    """Return full context, response, and contradictions for a single run."""
    if db is None:
        raise HTTPException(status_code=503, detail="Persistence not available — set MONGODB_URL")
    try:
        oid = ObjectId(run_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid run_id")
    query = {"_id": oid, "session_id": x_session_id} if x_session_id else {"_id": oid}
    doc = await db.detection_runs.find_one(query)
    if doc is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return HistoryDetail(
        run_id=str(doc["_id"]),
        timestamp=doc["timestamp"].isoformat(),
        faithfulness_score=doc["faithfulness_score"],
        method_used=doc["method_used"],
        provider=doc["provider"],
        context=doc["context"],
        response=doc["response"],
        contradictions=[Contradiction(**c) for c in doc.get("contradictions", [])],
    )


@router.delete("/history/{run_id}", status_code=204)
async def delete_history_item(
    run_id: str,
    db=Depends(get_db),
    x_session_id: str | None = Header(default=None),
) -> None:
    """Delete a single detection run and its associated feedback by ID."""
    if db is None:
        raise HTTPException(status_code=503, detail="Persistence not available — set MONGODB_URL")
    try:
        oid = ObjectId(run_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid run_id")
    query = {"_id": oid, "session_id": x_session_id} if x_session_id else {"_id": oid}
    result = await db.detection_runs.delete_one(query)
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Run not found")
    await db.feedback.delete_many({"run_id": oid})


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    db=Depends(get_db),
    x_session_id: str | None = Header(default=None),
) -> StatsResponse:
    """Aggregate detection stats for this session via MongoDB pipeline.

    Uses $group + $size to count total contradictions in a single round-trip,
    then two count_documents calls for confirmed-rate from the feedback
    collection.  The aggregation pipeline demonstrates production MongoDB usage.
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Persistence not available - set MONGODB_URL")
    match = {"$match": {"session_id": x_session_id}} if x_session_id else {"$match": {}}
    pipeline = [
        match,
        {"$group": {
            "_id": None,
            "total_runs": {"$sum": 1},
            "total_contradictions": {"$sum": {"$size": "$contradictions"}},
            "run_ids": {"$push": "$_id"},
        }},
    ]
    agg = await db.detection_runs.aggregate(pipeline).to_list(1)
    if not agg:
        return StatsResponse(total_runs=0, total_contradictions=0, confirmed_rate=0.0)
    run_ids = agg[0]["run_ids"]
    fb_pipeline = [
        {"$match": {"run_id": {"$in": run_ids}}},
        {"$group": {
            "_id": None,
            "total": {"$sum": 1},
            "confirmed": {"$sum": {"$cond": [{"$eq": ["$verdict", "confirmed"]}, 1, 0]}},
        }},
    ]
    fb_agg = await db.feedback.aggregate(fb_pipeline).to_list(1)
    total_feedback = fb_agg[0]["total"] if fb_agg else 0
    confirmed = fb_agg[0]["confirmed"] if fb_agg else 0
    confirmed_rate = confirmed / total_feedback if total_feedback > 0 else 0.0
    return StatsResponse(
        total_runs=agg[0]["total_runs"],
        total_contradictions=agg[0]["total_contradictions"],
        confirmed_rate=round(confirmed_rate, 4),
    )
