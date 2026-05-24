"""Motor (async MongoDB) client and collection accessors.

Motor is the official async driver for MongoDB - it's built on PyMongo but
non-blocking, integrating natively with FastAPI's async event loop.
The client is created once at startup and reused across all requests.
"""
import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from backend.config import settings

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None


async def connect() -> AsyncIOMotorDatabase | None:
    """Connect to MongoDB, verify the connection, and create indexes.

    Returns the database object, or None if MONGODB_URL is not configured
    or the server is unreachable - connection errors are non-fatal so the
    app can still run without history persistence.
    """
    global _client
    if not settings.mongodb_url:
        logger.info("MONGODB_URL not set - history persistence disabled.")
        return None
    _client = AsyncIOMotorClient(settings.mongodb_url, serverSelectionTimeoutMS=5000)
    db = _client.policy_guard
    try:
        await db.command("ping")
    except Exception as exc:
        logger.warning("MongoDB unreachable (%s) - persistence disabled.", exc)
        _client = None
        return None
    await _ensure_indexes(db)
    logger.info("Connected to MongoDB at %s", settings.mongodb_url)
    return db


async def _ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    """Create TTL and lookup indexes if they do not already exist.

    TTL index on detection_runs.timestamp: auto-expires documents after 30 days
    to bound storage growth - no manual cleanup job needed.
    Index on feedback.run_id: makes the per-run feedback lookup O(log n).
    """
    await db.detection_runs.create_index(
        "timestamp",
        expireAfterSeconds=30 * 24 * 60 * 60,
        background=True,
    )
    await db.detection_runs.create_index("session_id", background=True)
    await db.feedback.create_index("run_id", background=True)
    logger.debug("MongoDB indexes ensured")


def disconnect() -> None:
    """Close the Motor client cleanly on application shutdown."""
    global _client
    if _client:
        _client.close()
        _client = None
        logger.info("MongoDB connection closed")
