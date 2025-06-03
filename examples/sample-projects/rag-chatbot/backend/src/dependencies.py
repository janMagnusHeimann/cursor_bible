"""
FastAPI Dependencies for RAG Chatbot
Provides dependency injection for services and database connections.
"""

from typing import Generator, Optional
from fastapi import Depends, HTTPException, Header
import redis.asyncio as redis

from .config import Settings, get_settings
from .chat import ChatService
from .exceptions import AuthenticationException, ServiceUnavailableException

# Global service instances (set in main.py)
_vector_store_manager = None
_chat_service = None
_redis_client = None


def set_global_services(vector_store_manager, chat_service):
    """Set global service instances (called from main.py)."""
    global _vector_store_manager, _chat_service
    _vector_store_manager = vector_store_manager
    _chat_service = chat_service


async def get_vector_store():
    """Get vector store manager dependency."""
    if _vector_store_manager is None:
        raise ServiceUnavailableException("Vector store manager not initialized")
    return _vector_store_manager


async def get_chat_service() -> ChatService:
    """Get chat service dependency."""
    if _chat_service is None:
        raise ServiceUnavailableException("Chat service not initialized")
    return _chat_service


async def get_redis_client() -> redis.Redis:
    """Get Redis client dependency."""
    global _redis_client
    
    if _redis_client is None:
        settings = get_settings()
        _redis_client = redis.from_url(
            settings.redis_url,
            password=settings.redis_password,
            db=settings.redis_db,
            encoding="utf-8",
            decode_responses=True
        )
    
    try:
        await _redis_client.ping()
        return _redis_client
    except Exception:
        raise ServiceUnavailableException("Redis service unavailable")


async def verify_api_key(
    x_api_key: Optional[str] = Header(None),
    settings: Settings = Depends(get_settings)
) -> bool:
    """Verify API key for admin endpoints."""
    if not settings.admin_api_key:
        return True  # No API key configured, allow access
    
    if not x_api_key:
        raise AuthenticationException("API key required")
    
    if x_api_key != settings.admin_api_key:
        raise AuthenticationException("Invalid API key")
    
    return True


async def get_current_session_id(
    session_id: Optional[str] = Header(None, alias="X-Session-ID")
) -> Optional[str]:
    """Extract session ID from headers."""
    return session_id


class RateLimiter:
    """Rate limiting dependency."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
    
    async def __call__(
        self,
        client_ip: str = Header(None, alias="X-Forwarded-For"),
        redis_client: redis.Redis = Depends(get_redis_client)
    ):
        """Check rate limit for client."""
        if not client_ip:
            client_ip = "unknown"
        
        key = f"rate_limit:{client_ip}"
        
        try:
            current = await redis_client.get(key)
            if current is None:
                await redis_client.setex(key, 60, 1)
                return True
            
            if int(current) >= self.requests_per_minute:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": "60"}
                )
            
            await redis_client.incr(key)
            return True
            
        except Exception as e:
            # If Redis is down, allow the request
            return True


# Pre-configured rate limiters
standard_rate_limit = RateLimiter(requests_per_minute=60)
chat_rate_limit = RateLimiter(requests_per_minute=30)
admin_rate_limit = RateLimiter(requests_per_minute=100) 