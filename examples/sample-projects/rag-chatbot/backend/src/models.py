"""
Pydantic models for RAG Chatbot API
Defines request/response schemas with validation.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator
import uuid


class ChatMessage(BaseModel):
    """Individual chat message model."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ['user', 'assistant', 'system']:
            raise ValueError('Role must be user, assistant, or system')
        return v
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        if len(v) > 10000:
            raise ValueError('Content too long (max 10000 characters)')
        return v.strip()


class ConversationHistory(BaseModel):
    """Conversation history container."""
    messages: List[ChatMessage] = Field(default_factory=list)
    session_id: str = Field(..., description="Unique session identifier")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to conversation history."""
        message = ChatMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_recent_messages(self, limit: int = 10) -> List[ChatMessage]:
        """Get recent messages up to limit."""
        return self.messages[-limit:] if self.messages else []


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User's chat message")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_history: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    stream: bool = Field(default=True, description="Whether to stream response")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=1000, ge=1, le=4000)
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        if len(v) > 5000:
            raise ValueError('Message too long (max 5000 characters)')
        return v.strip()
    
    @validator('session_id')
    def validate_session_id(cls, v):
        if not v:
            return str(uuid.uuid4())
        # Basic UUID format validation
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            return str(uuid.uuid4())


class SourceDocument(BaseModel):
    """Source document model for RAG responses."""
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    
    @validator('content')
    def validate_content(cls, v):
        if len(v) > 2000:  # Truncate long content
            return v[:2000] + "..."
        return v


class ChatResponse(BaseModel):
    """Chat response model."""
    message: str = Field(..., description="Assistant's response")
    session_id: str = Field(..., description="Session identifier")
    source_documents: List[SourceDocument] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @validator('message')
    def validate_message(cls, v):
        if not v:
            raise ValueError('Response message cannot be empty')
        return v


class DocumentUpload(BaseModel):
    """Document upload model for ingestion."""
    content: str = Field(..., description="Document content")
    filename: str = Field(..., description="Original filename")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    document_type: str = Field(default="text", description="Document type")
    source_url: Optional[str] = None
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError('Document content cannot be empty')
        if len(v) > 1000000:  # 1MB limit
            raise ValueError('Document too large (max 1MB)')
        return v.strip()
    
    @validator('filename')
    def validate_filename(cls, v):
        if not v or not v.strip():
            raise ValueError('Filename cannot be empty')
        # Basic filename validation
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
        if any(char in v for char in invalid_chars):
            raise ValueError('Filename contains invalid characters')
        return v.strip()


class DocumentStatus(BaseModel):
    """Document processing status model."""
    job_id: str = Field(..., description="Processing job ID")
    status: str = Field(..., description="Processing status")
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    documents_processed: int = Field(default=0, ge=0)
    total_documents: int = Field(default=0, ge=0)
    error_message: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ['pending', 'processing', 'completed', 'failed']
        if v not in valid_statuses:
            raise ValueError(f'Status must be one of: {valid_statuses}')
        return v


class HealthCheck(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Overall service status")
    services: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(default="1.0.0")
    
    @validator('status')
    def validate_status(cls, v):
        if v not in ['healthy', 'unhealthy', 'degraded']:
            raise ValueError('Status must be healthy, unhealthy, or degraded')
        return v


class MetricsResponse(BaseModel):
    """Metrics response model."""
    total_chats: int = Field(default=0, ge=0)
    active_sessions: int = Field(default=0, ge=0)
    average_response_time: float = Field(default=0.0, ge=0.0)
    error_rate: float = Field(default=0.0, ge=0.0, le=100.0)
    total_tokens_used: int = Field(default=0, ge=0)
    documents_indexed: int = Field(default=0, ge=0)
    uptime_seconds: float = Field(default=0.0, ge=0.0)
    memory_usage_mb: float = Field(default=0.0, ge=0.0)
    
    class Config:
        schema_extra = {
            "example": {
                "total_chats": 1547,
                "active_sessions": 23,
                "average_response_time": 1.2,
                "error_rate": 0.5,
                "total_tokens_used": 125000,
                "documents_indexed": 342,
                "uptime_seconds": 86400.0,
                "memory_usage_mb": 512.8
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Message cannot be empty",
                "details": {"field": "message", "type": "value_error"},
                "timestamp": "2023-11-20T10:30:00Z",
                "request_id": "req_123abc"
            }
        }


class ConfigUpdate(BaseModel):
    """Configuration update model."""
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=4000)
    retrieval_k: Optional[int] = Field(None, ge=1, le=20)
    conversation_window_size: Optional[int] = Field(None, ge=1, le=50)
    
    class Config:
        schema_extra = {
            "example": {
                "temperature": 0.8,
                "max_tokens": 1500,
                "retrieval_k": 5,
                "conversation_window_size": 10
            }
        } 