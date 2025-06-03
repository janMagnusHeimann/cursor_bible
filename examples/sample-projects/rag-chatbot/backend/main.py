"""
RAG Chatbot FastAPI Backend
Production-ready implementation with streaming, error handling, and monitoring.
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from src.config import Settings
from src.vector_store import VectorStoreManager
from src.chat import ChatService
from src.ingestion import DocumentIngestionService
from src.middleware import RateLimitMiddleware, LoggingMiddleware
from src.dependencies import get_vector_store, get_chat_service
from src.exceptions import RAGException, LLMException
from src.models import ChatRequest, ChatResponse, DocumentUpload, HealthCheck

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global services
vector_store_manager: VectorStoreManager = None
chat_service: ChatService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    global vector_store_manager, chat_service
    
    try:
        # Initialize services
        settings = Settings()
        vector_store_manager = VectorStoreManager(settings)
        await vector_store_manager.initialize()
        
        chat_service = ChatService(vector_store_manager, settings)
        await chat_service.initialize()
        
        logger.info("RAG Chatbot services initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    finally:
        # Cleanup
        if chat_service:
            await chat_service.cleanup()
        if vector_store_manager:
            await vector_store_manager.cleanup()
        logger.info("RAG Chatbot services cleaned up")


# Create FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Production-ready RAG chatbot with streaming responses",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(LoggingMiddleware)


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint with service status."""
    try:
        # Check vector store connection
        vector_health = await vector_store_manager.health_check()
        
        # Check LLM service
        chat_health = await chat_service.health_check()
        
        return HealthCheck(
            status="healthy",
            services={
                "vector_store": vector_health,
                "llm_service": chat_health,
                "api": "healthy"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/chat/stream")
async def stream_chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
) -> StreamingResponse:
    """Stream chat responses with RAG context."""
    
    async def generate_response() -> AsyncIterator[str]:
        """Generate streaming response with proper error handling."""
        try:
            async for chunk in chat_service.stream_chat(
                query=request.message,
                session_id=request.session_id,
                conversation_history=request.conversation_history
            ):
                yield f"data: {chunk}\n\n"
                
        except LLMException as e:
            logger.error(f"LLM error in stream_chat: {e}")
            yield f"data: {{'error': 'LLM service error', 'message': '{str(e)}'}}\n\n"
            
        except RAGException as e:
            logger.error(f"RAG error in stream_chat: {e}")
            yield f"data: {{'error': 'RAG pipeline error', 'message': '{str(e)}'}}\n\n"
            
        except Exception as e:
            logger.error(f"Unexpected error in stream_chat: {e}")
            yield f"data: {{'error': 'Internal server error'}}\n\n"
        
        finally:
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
) -> ChatResponse:
    """Non-streaming chat endpoint for simple interactions."""
    try:
        response = await chat_service.chat(
            query=request.message,
            session_id=request.session_id,
            conversation_history=request.conversation_history
        )
        return response
        
    except LLMException as e:
        logger.error(f"LLM error in chat: {e}")
        raise HTTPException(status_code=502, detail="LLM service error")
        
    except RAGException as e:
        logger.error(f"RAG error in chat: {e}")
        raise HTTPException(status_code=500, detail="RAG pipeline error")
        
    except Exception as e:
        logger.error(f"Unexpected error in chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/documents/ingest")
async def ingest_documents(
    documents: List[DocumentUpload],
    background_tasks: BackgroundTasks,
    vector_store: VectorStoreManager = Depends(get_vector_store)
):
    """Ingest documents for RAG knowledge base."""
    try:
        ingestion_service = DocumentIngestionService(vector_store)
        
        # Process documents in background
        background_tasks.add_task(
            ingestion_service.process_documents,
            documents
        )
        
        return {
            "message": f"Processing {len(documents)} documents",
            "document_count": len(documents),
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Document ingestion error: {e}")
        raise HTTPException(status_code=500, detail="Document ingestion failed")


@app.get("/documents/status/{job_id}")
async def get_ingestion_status(
    job_id: str,
    vector_store: VectorStoreManager = Depends(get_vector_store)
):
    """Get document ingestion job status."""
    try:
        ingestion_service = DocumentIngestionService(vector_store)
        status = await ingestion_service.get_job_status(job_id)
        return status
        
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")


@app.delete("/documents")
async def clear_documents(
    namespace: str = "default",
    vector_store: VectorStoreManager = Depends(get_vector_store)
):
    """Clear all documents from vector store namespace."""
    try:
        await vector_store.clear_namespace(namespace)
        return {"message": f"Cleared documents from namespace: {namespace}"}
        
    except Exception as e:
        logger.error(f"Document clearing error: {e}")
        raise HTTPException(status_code=500, detail="Document clearing failed")


@app.get("/metrics")
async def get_metrics(
    chat_service: ChatService = Depends(get_chat_service)
):
    """Get application metrics for monitoring."""
    try:
        metrics = await chat_service.get_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 