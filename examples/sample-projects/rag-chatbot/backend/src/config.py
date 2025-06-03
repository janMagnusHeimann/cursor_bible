"""
Configuration management for RAG Chatbot
Handles environment variables and application settings.
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field, validator
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    app_name: str = Field(default="RAG Chatbot", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI model name")
    openai_embedding_model: str = Field(default="text-embedding-ada-002", description="Embedding model")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(default=1000, ge=1, le=4000, description="Max tokens per response")
    
    # Vector Store Configuration
    pinecone_api_key: str = Field(..., description="Pinecone API key")
    pinecone_environment: str = Field(..., description="Pinecone environment")
    pinecone_index_name: str = Field(default="rag-chatbot", description="Pinecone index name")
    pinecone_namespace: str = Field(default="default", description="Pinecone namespace")
    vector_dimension: int = Field(default=1536, description="Vector dimension for embeddings")
    
    # RAG Configuration
    retrieval_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    chunk_size: int = Field(default=1000, ge=100, le=5000, description="Document chunk size")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="Chunk overlap size")
    conversation_window_size: int = Field(default=10, ge=1, le=50, description="Conversation memory window")
    
    # Redis Configuration (for session management)
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    session_ttl: int = Field(default=3600, ge=300, le=86400, description="Session TTL in seconds")
    
    # Security Configuration
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    admin_api_key: Optional[str] = Field(default=None, description="Admin API key")
    rate_limit_requests: int = Field(default=100, ge=1, description="Rate limit requests per minute")
    rate_limit_window: int = Field(default=60, ge=1, description="Rate limit window in seconds")
    
    # Performance Configuration
    max_concurrent_requests: int = Field(default=10, ge=1, le=100, description="Max concurrent requests")
    request_timeout: int = Field(default=30, ge=5, le=300, description="Request timeout in seconds")
    embedding_batch_size: int = Field(default=10, ge=1, le=100, description="Embedding batch size")
    vector_cache_ttl: int = Field(default=300, ge=60, le=3600, description="Vector cache TTL")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format: json or text")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    
    # Document Processing Configuration
    allowed_file_types: List[str] = Field(
        default=["txt", "pdf", "docx", "md", "html"],
        description="Allowed file types for upload"
    )
    max_file_size_mb: int = Field(default=10, ge=1, le=100, description="Max file size in MB")
    max_files_per_request: int = Field(default=5, ge=1, le=20, description="Max files per upload")
    
    # Health Check Configuration
    health_check_interval: int = Field(default=30, ge=10, le=300, description="Health check interval")
    dependency_timeout: int = Field(default=5, ge=1, le=30, description="Dependency check timeout")
    
    @validator('openai_api_key')
    def validate_openai_key(cls, v):
        if not v or not v.startswith('sk-'):
            raise ValueError('OpenAI API key must start with sk-')
        return v
    
    @validator('pinecone_api_key')
    def validate_pinecone_key(cls, v):
        if not v or len(v) < 20:
            raise ValueError('Pinecone API key must be at least 20 characters')
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()
    
    @validator('log_format')
    def validate_log_format(cls, v):
        if v not in ['json', 'text']:
            raise ValueError('Log format must be json or text')
        return v
    
    @validator('chunk_overlap')
    def validate_chunk_overlap(cls, v, values):
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError('Chunk overlap must be less than chunk size')
        return v
    
    @validator('allowed_file_types')
    def validate_file_types(cls, v):
        valid_types = ['txt', 'pdf', 'docx', 'md', 'html', 'csv', 'json']
        invalid_types = [t for t in v if t not in valid_types]
        if invalid_types:
            raise ValueError(f'Invalid file types: {invalid_types}')
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Environment variable mappings
        fields = {
            'openai_api_key': {'env': 'OPENAI_API_KEY'},
            'pinecone_api_key': {'env': 'PINECONE_API_KEY'},
            'pinecone_environment': {'env': 'PINECONE_ENVIRONMENT'},
            'redis_url': {'env': 'REDIS_URL'},
            'redis_password': {'env': 'REDIS_PASSWORD'},
            'admin_api_key': {'env': 'ADMIN_API_KEY'},
        }


class DevelopmentSettings(Settings):
    """Development environment specific settings."""
    debug: bool = True
    log_level: str = "DEBUG"
    rate_limit_requests: int = 1000  # Higher limit for development
    enable_metrics: bool = True
    
    class Config:
        env_prefix = "DEV_"


class ProductionSettings(Settings):
    """Production environment specific settings."""
    debug: bool = False
    log_level: str = "INFO"
    log_format: str = "json"
    enable_metrics: bool = True
    request_timeout: int = 15  # Shorter timeout in production
    
    class Config:
        env_prefix = "PROD_"


class TestSettings(Settings):
    """Test environment specific settings."""
    debug: bool = True
    log_level: str = "WARNING"
    pinecone_index_name: str = "test-rag-chatbot"
    pinecone_namespace: str = "test"
    redis_db: int = 1  # Use different Redis DB for tests
    
    # Use mock services in tests
    openai_api_key: str = "sk-test-key-mock"
    pinecone_api_key: str = "test-pinecone-key-mock"
    pinecone_environment: str = "test"
    
    class Config:
        env_prefix = "TEST_"


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings based on environment.
    Uses caching to avoid re-reading environment variables.
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "test":
        return TestSettings()
    else:
        return DevelopmentSettings()


# Convenience function for getting specific config sections
def get_openai_config() -> dict:
    """Get OpenAI specific configuration."""
    settings = get_settings()
    return {
        "api_key": settings.openai_api_key,
        "model": settings.openai_model,
        "embedding_model": settings.openai_embedding_model,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }


def get_pinecone_config() -> dict:
    """Get Pinecone specific configuration."""
    settings = get_settings()
    return {
        "api_key": settings.pinecone_api_key,
        "environment": settings.pinecone_environment,
        "index_name": settings.pinecone_index_name,
        "namespace": settings.pinecone_namespace,
        "dimension": settings.vector_dimension,
    }


def get_redis_config() -> dict:
    """Get Redis specific configuration."""
    settings = get_settings()
    return {
        "url": settings.redis_url,
        "password": settings.redis_password,
        "db": settings.redis_db,
        "session_ttl": settings.session_ttl,
    }


def get_rag_config() -> dict:
    """Get RAG specific configuration."""
    settings = get_settings()
    return {
        "retrieval_k": settings.retrieval_k,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "conversation_window_size": settings.conversation_window_size,
        "embedding_batch_size": settings.embedding_batch_size,
    } 