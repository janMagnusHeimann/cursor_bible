"""
Custom exceptions for RAG Chatbot
Defines specific exception types for better error handling.
"""

from typing import Optional, Dict, Any


class RAGChatbotException(Exception):
    """Base exception class for RAG Chatbot."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class LLMException(RAGChatbotException):
    """Exception raised for LLM-related errors."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)
        self.model_name = model_name


class RAGException(RAGChatbotException):
    """Exception raised for RAG pipeline errors."""
    
    def __init__(
        self,
        message: str,
        pipeline_stage: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)
        self.pipeline_stage = pipeline_stage


class VectorStoreException(RAGChatbotException):
    """Exception raised for vector store operations."""
    
    def __init__(
        self,
        message: str,
        store_type: Optional[str] = None,
        operation: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)
        self.store_type = store_type
        self.operation = operation


class DocumentProcessingException(RAGChatbotException):
    """Exception raised for document processing errors."""
    
    def __init__(
        self,
        message: str,
        document_type: Optional[str] = None,
        filename: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)
        self.document_type = document_type
        self.filename = filename


class ConfigurationException(RAGChatbotException):
    """Exception raised for configuration errors."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)
        self.config_key = config_key


class RateLimitException(RAGChatbotException):
    """Exception raised when rate limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        limit_type: Optional[str] = None,
        retry_after: Optional[int] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)
        self.limit_type = limit_type
        self.retry_after = retry_after


class AuthenticationException(RAGChatbotException):
    """Exception raised for authentication errors."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        auth_type: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)
        self.auth_type = auth_type


class ValidationException(RAGChatbotException):
    """Exception raised for input validation errors."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)
        self.field_name = field_name
        self.field_value = field_value


class ServiceUnavailableException(RAGChatbotException):
    """Exception raised when external services are unavailable."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)
        self.service_name = service_name 