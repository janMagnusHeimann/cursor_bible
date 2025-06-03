"""
Chat Service for RAG Chatbot
Handles LLM interactions, streaming responses, and conversation management.
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, AsyncIterator, Optional
from datetime import datetime
import uuid

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from .config import Settings
from .vector_store import VectorStoreManager
from .exceptions import LLMException, RAGException
from .models import ChatResponse, ConversationHistory
from .metrics import MetricsCollector

logger = logging.getLogger(__name__)


class StreamingCallbackHandler(AsyncCallbackHandler):
    """Custom async callback handler for streaming responses."""
    
    def __init__(self):
        self.tokens = []
        self.is_done = False
        
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Handle new token from LLM."""
        self.tokens.append(token)
        
    async def on_llm_end(self, response, **kwargs) -> None:
        """Handle LLM completion."""
        self.is_done = True


class ChatService:
    """
    Core chat service implementing RAG with LangChain.
    Provides streaming and non-streaming chat capabilities.
    """
    
    def __init__(self, vector_store_manager: VectorStoreManager, settings: Settings):
        self.vector_store_manager = vector_store_manager
        self.settings = settings
        self.llm = None
        self.retrieval_chain = None
        self.memory_store = {}  # In-memory store for conversation history
        self.metrics = MetricsCollector()
        
    async def initialize(self) -> None:
        """Initialize chat service components."""
        try:
            # Initialize LLM with streaming support
            self.llm = ChatOpenAI(
                model_name=self.settings.openai_model,
                openai_api_key=self.settings.openai_api_key,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
                streaming=True,
                verbose=True
            )
            
            # Initialize retrieval chain
            await self._setup_retrieval_chain()
            
            logger.info("Chat service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize chat service: {e}")
            raise LLMException(f"Chat service initialization failed: {e}")
    
    async def _setup_retrieval_chain(self) -> None:
        """Setup the conversational retrieval chain."""
        try:
            # Get retriever from vector store
            retriever = await self.vector_store_manager.get_retriever(
                search_kwargs={"k": self.settings.retrieval_k}
            )
            
            # Create conversation memory
            memory = ConversationBufferWindowMemory(
                k=self.settings.conversation_window_size,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            # Create custom prompt template
            system_template = """You are a helpful AI assistant that answers questions based on the provided context.
            Use the following pieces of context to answer the user's question. If you don't know the answer based on the context, just say that you don't know.
            Always be concise and accurate in your responses.
            
            Context: {context}
            
            Chat History: {chat_history}
            """
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])
            
            # Create retrieval chain
            self.retrieval_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=True,
                verbose=True,
                combine_docs_chain_kwargs={"prompt": prompt}
            )
            
        except Exception as e:
            logger.error(f"Failed to setup retrieval chain: {e}")
            raise RAGException(f"Retrieval chain setup failed: {e}")
    
    async def stream_chat(
        self,
        query: str,
        session_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncIterator[str]:
        """
        Stream chat response with RAG context.
        
        Args:
            query: User's question
            session_id: Unique session identifier
            conversation_history: Previous conversation messages
            
        Yields:
            Streaming response chunks
        """
        start_time = datetime.now()
        
        try:
            # Validate inputs
            if not query.strip():
                raise ValueError("Query cannot be empty")
                
            # Load conversation history
            await self._load_conversation_history(session_id, conversation_history)
            
            # Setup streaming callback
            callback_handler = StreamingCallbackHandler()
            
            # Get relevant documents
            docs = await self._retrieve_documents(query)
            
            # Generate streaming response
            response_chunks = []
            async for chunk in self._generate_streaming_response(
                query, docs, callback_handler
            ):
                response_chunks.append(chunk)
                yield json.dumps({
                    "type": "token",
                    "content": chunk,
                    "session_id": session_id
                })
            
            # Save conversation
            full_response = "".join(response_chunks)
            await self._save_conversation_turn(
                session_id, query, full_response, docs
            )
            
            # Track metrics
            duration = (datetime.now() - start_time).total_seconds()
            await self.metrics.track_chat_completion(
                session_id=session_id,
                query_length=len(query),
                response_length=len(full_response),
                duration=duration,
                source_docs_count=len(docs)
            )
            
            # Send completion signal
            yield json.dumps({
                "type": "completion",
                "session_id": session_id,
                "source_documents": [
                    {
                        "content": doc.page_content[:200],
                        "metadata": doc.metadata
                    } for doc in docs[:3]  # Limit to top 3 sources
                ]
            })
            
        except Exception as e:
            logger.error(f"Error in stream_chat: {e}")
            await self.metrics.track_error("stream_chat", str(e))
            yield json.dumps({
                "type": "error",
                "message": str(e),
                "session_id": session_id
            })
    
    async def chat(
        self,
        query: str,
        session_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> ChatResponse:
        """
        Non-streaming chat response with RAG context.
        
        Args:
            query: User's question
            session_id: Unique session identifier
            conversation_history: Previous conversation messages
            
        Returns:
            Complete chat response
        """
        start_time = datetime.now()
        
        try:
            # Validate inputs
            if not query.strip():
                raise ValueError("Query cannot be empty")
            
            # Load conversation history
            await self._load_conversation_history(session_id, conversation_history)
            
            # Get response from retrieval chain
            result = await self.retrieval_chain.acall({
                "question": query,
                "chat_history": self._get_langchain_history(session_id)
            })
            
            response = result["answer"]
            source_docs = result["source_documents"]
            
            # Save conversation
            await self._save_conversation_turn(
                session_id, query, response, source_docs
            )
            
            # Track metrics
            duration = (datetime.now() - start_time).total_seconds()
            await self.metrics.track_chat_completion(
                session_id=session_id,
                query_length=len(query),
                response_length=len(response),
                duration=duration,
                source_docs_count=len(source_docs)
            )
            
            return ChatResponse(
                message=response,
                session_id=session_id,
                source_documents=[
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_score": getattr(doc, "relevance_score", 0.0)
                    } for doc in source_docs
                ],
                metadata={
                    "response_time": duration,
                    "token_count": len(response.split()),
                    "source_count": len(source_docs)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            await self.metrics.track_error("chat", str(e))
            raise LLMException(f"Chat completion failed: {e}")
    
    async def _retrieve_documents(self, query: str) -> List[Any]:
        """Retrieve relevant documents for the query."""
        try:
            retriever = await self.vector_store_manager.get_retriever()
            docs = await retriever.aget_relevant_documents(query)
            return docs
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            raise RAGException(f"Failed to retrieve documents: {e}")
    
    async def _generate_streaming_response(
        self,
        query: str,
        docs: List[Any],
        callback_handler: StreamingCallbackHandler
    ) -> AsyncIterator[str]:
        """Generate streaming response using LLM."""
        try:
            # Create context from documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create messages
            messages = [
                SystemMessage(content=f"Context: {context}"),
                HumanMessage(content=query)
            ]
            
            # Stream response
            async for token in self.llm.astream(messages):
                if hasattr(token, 'content'):
                    yield token.content
                else:
                    yield str(token)
                    
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise LLMException(f"Failed to generate streaming response: {e}")
    
    async def _load_conversation_history(
        self,
        session_id: str,
        conversation_history: Optional[List[Dict[str, str]]]
    ) -> None:
        """Load conversation history for session."""
        if conversation_history:
            self.memory_store[session_id] = conversation_history
        elif session_id not in self.memory_store:
            self.memory_store[session_id] = []
    
    async def _save_conversation_turn(
        self,
        session_id: str,
        query: str,
        response: str,
        source_docs: List[Any]
    ) -> None:
        """Save conversation turn to memory."""
        if session_id not in self.memory_store:
            self.memory_store[session_id] = []
        
        self.memory_store[session_id].extend([
            {"role": "user", "content": query},
            {
                "role": "assistant",
                "content": response,
                "metadata": {
                    "source_documents": [doc.metadata for doc in source_docs],
                    "timestamp": datetime.now().isoformat()
                }
            }
        ])
        
        # Limit conversation history size
        max_history = self.settings.conversation_window_size * 2  # user + assistant
        if len(self.memory_store[session_id]) > max_history:
            self.memory_store[session_id] = self.memory_store[session_id][-max_history:]
    
    def _get_langchain_history(self, session_id: str) -> List[Any]:
        """Convert conversation history to LangChain message format."""
        history = self.memory_store.get(session_id, [])
        messages = []
        
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        return messages
    
    async def health_check(self) -> str:
        """Check service health."""
        try:
            # Test LLM connection
            test_response = await self.llm.agenerate([[HumanMessage(content="test")]])
            
            if test_response and test_response.generations:
                return "healthy"
            else:
                return "unhealthy"
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return "unhealthy"
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        return await self.metrics.get_metrics()
    
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        try:
            # Clear memory store
            self.memory_store.clear()
            
            # Close any open connections
            if hasattr(self.llm, 'client'):
                await self.llm.client.close()
            
            logger.info("Chat service cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}") 