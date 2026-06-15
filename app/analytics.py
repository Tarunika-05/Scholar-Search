import os
import asyncio
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, text
from app.logger import get_logger

logger = get_logger("analytics")

# Determine DB URL: prioritize environment variable, fallback to local file-based SQLite if running locally without Docker
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/analytics.db")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

class QueryLog(Base):
    __tablename__ = "query_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    query_text = Column(String, index=True)
    search_mode = Column(String)
    cache_hit = Column(Boolean)
    latency_ms = Column(Float)
    dominant_cluster = Column(Integer, nullable=True)

async def init_db():
    """Create the tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Analytics database initialized.")

async def log_query_async(query_text: str, search_mode: str, cache_hit: bool, latency_ms: float, dominant_cluster: int | None = None):
    """Asynchronously log a query to the database."""
    try:
        async with AsyncSessionLocal() as session:
            new_log = QueryLog(
                query_text=query_text,
                search_mode=search_mode,
                cache_hit=cache_hit,
                latency_ms=latency_ms,
                dominant_cluster=dominant_cluster
            )
            session.add(new_log)
            await session.commit()
    except Exception as e:
        logger.error(f"Failed to log query: {e}")

def log_query(query_text: str, search_mode: str, cache_hit: bool, latency_ms: float, dominant_cluster: int | None = None):
    """
    Fire-and-forget query logging.
    Creates an asyncio task that won't block the caller.
    """
    asyncio.create_task(log_query_async(
        query_text=query_text,
        search_mode=search_mode,
        cache_hit=cache_hit,
        latency_ms=latency_ms,
        dominant_cluster=dominant_cluster
    ))

async def get_analytics_stats():
    """Retrieve basic analytics statistics."""
    stats = {}
    try:
        async with AsyncSessionLocal() as session:
            # Total queries
            result = await session.execute(text("SELECT COUNT(*) FROM query_logs"))
            stats["total_queries"] = result.scalar()
            
            # Cache hit rate
            result = await session.execute(text("SELECT COUNT(*) FROM query_logs WHERE cache_hit = true"))
            cache_hits = result.scalar()
            stats["cache_hits"] = cache_hits
            stats["cache_hit_rate"] = round((cache_hits / max(stats["total_queries"], 1)) * 100, 2)
            
            # Average latency
            result = await session.execute(text("SELECT AVG(latency_ms) FROM query_logs"))
            avg_latency = result.scalar()
            stats["average_latency_ms"] = round(avg_latency, 2) if avg_latency else 0.0
            
            # Latency by cache hit
            result_hit = await session.execute(text("SELECT AVG(latency_ms) FROM query_logs WHERE cache_hit = true"))
            avg_latency_hit = result_hit.scalar()
            stats["average_latency_hit_ms"] = round(avg_latency_hit, 2) if avg_latency_hit else 0.0
            
            # Latency by cache miss
            result_miss = await session.execute(text("SELECT AVG(latency_ms) FROM query_logs WHERE cache_hit = false"))
            avg_latency_miss = result_miss.scalar()
            stats["average_latency_miss_ms"] = round(avg_latency_miss, 2) if avg_latency_miss else 0.0
            
            # Recent queries
            result = await session.execute(text("SELECT query_text, search_mode, cache_hit, latency_ms FROM query_logs ORDER BY timestamp DESC LIMIT 5"))
            stats["recent_queries"] = [dict(row._mapping) for row in result.fetchall()]
            
            return stats
    except Exception as e:
        logger.error(f"Failed to fetch analytics stats: {e}")
        return {"error": "Failed to fetch stats"}
