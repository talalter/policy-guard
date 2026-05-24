"""Feedback endpoint: POST /feedback/{run_id}."""

import logging
from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import get_db
from backend.models import FeedbackRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/feedback/{run_id}", status_code=204)
async def submit_feedback(
    run_id: str,
    body: FeedbackRequest,
    db=Depends(get_db),
) -> None:
    """Store user verdict on a single violation finding.

    Builds a feedback loop: confirmed/false-positive labels accumulate in the
    'feedback' collection and are surfaced as confirmed_rate in GET /stats.
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Persistence not available - set MONGODB_URL")
    try:
        oid = ObjectId(run_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid run_id")
    await db.feedback.update_one(
        {"run_id": oid, "violation_index": body.violation_index},
        {"$set": {"verdict": body.verdict.value, "timestamp": datetime.now(timezone.utc)}},
        upsert=True,
    )
