# RAG Chatbot - Production-Ready Example

A comprehensive RAG (Retrieval-Augmented Generation) chatbot built with FastAPI, LangChain, OpenAI, and Pinecone. This example demonstrates best practices for building production-ready AI applications with proper error handling, streaming responses, and monitoring.

## Features

- 🚀 **Streaming Chat Responses** - Real-time streaming with Server-Sent Events
- 🔍 **Advanced RAG Pipeline** - Semantic search with reranking and context compression
- 📄 **Document Ingestion** - Support for multiple file formats (PDF, DOCX, TXT, MD, HTML)
- 💾 **Session Management** - Redis-based conversation memory with configurable windows
- 📊 **Monitoring & Metrics** - Built-in health checks, performance metrics, and observability
- 🔒 **Production Security** - Rate limiting, input validation, API key authentication
- 🐳 **Docker Ready** - Complete containerization with docker-compose
- ✅ **Comprehensive Testing** - Unit and integration tests with mocking

## Tech Stack

- **Backend**: FastAPI with async/await patterns
- **LLM**: OpenAI GPT-3.5/4 with LangChain integration
- **Vector DB**: Pinecone for semantic search
- **Caching**: Redis for session management
- **Frontend**: React with TypeScript (see frontend/ directory)
- **Monitoring**: Prometheus metrics and structured logging

## Quick Start

### 1. Environment Setup

First, copy the environment template and configure your API keys:

```bash
cp .env.example .env
```

Add the following environment variables to your `.env` file:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-3.5-turbo

# Pinecone Configuration  
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENVIRONMENT=your-pinecone-environment
PINECONE_INDEX_NAME=rag-chatbot

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=your-redis-password-if-needed

# Optional: Admin API Key for management endpoints
ADMIN_API_KEY=your-admin-api-key-here

# Environment
ENVIRONMENT=development
```

### 2. Installation

Using Docker (Recommended):
```bash
docker-compose up -d
```

Using Python directly:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Initialize Vector Database

Create and populate your Pinecone index:

```bash
# Upload sample documents
curl -X POST "http://localhost:8000/documents/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "content": "Your document content here...",
        "filename": "sample.txt",
        "metadata": {"source": "manual_upload"}
      }
    ]
  }'
```

### 4. Test the API

Start chatting:
```bash
# Non-streaming chat
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What information do you have about AI?",
    "session_id": "test-session-123"
  }'

# Streaming chat (use curl or a streaming client)
curl -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about machine learning",
    "session_id": "test-session-123"
  }'
```

## Project Structure

```
rag-chatbot/
├── .cursorrules              # Cursor AI development rules
├── backend/
│   ├── main.py              # FastAPI application entry point
│   ├── src/
│   │   ├── chat.py          # Core chat service with LangChain
│   │   ├── vector_store.py  # Pinecone integration
│   │   ├── ingestion.py     # Document processing pipeline
│   │   ├── config.py        # Configuration management
│   │   ├── models.py        # Pydantic models
│   │   ├── exceptions.py    # Custom exception classes
│   │   ├── middleware.py    # Custom middleware
│   │   ├── dependencies.py  # FastAPI dependencies
│   │   └── metrics.py       # Metrics collection
├── frontend/                # React TypeScript frontend
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── hooks/          # Custom hooks
│   │   ├── services/       # API services
│   │   └── types/          # TypeScript types
├── tests/                   # Comprehensive test suite
├── data/                    # Sample documents
├── docker-compose.yml       # Multi-service setup
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Configuration

The application uses Pydantic settings for configuration management. Key settings include:

### RAG Configuration
- `RETRIEVAL_K`: Number of documents to retrieve (default: 5)
- `CHUNK_SIZE`: Document chunk size (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `CONVERSATION_WINDOW_SIZE`: Memory window size (default: 10)

### Performance Settings
- `MAX_CONCURRENT_REQUESTS`: Concurrent request limit (default: 10)
- `REQUEST_TIMEOUT`: Request timeout in seconds (default: 30)
- `RATE_LIMIT_REQUESTS`: Rate limit per minute (default: 100)

### Security Settings
- `ADMIN_API_KEY`: Admin endpoints authentication
- `API_KEY_HEADER`: Custom API key header name

## API Endpoints

### Chat Endpoints
- `POST /chat/stream` - Streaming chat with RAG
- `POST /chat` - Non-streaming chat
- `GET /health` - Health check with service status

### Document Management
- `POST /documents/ingest` - Upload and process documents
- `GET /documents/status/{job_id}` - Check processing status
- `DELETE /documents` - Clear document namespace

### Monitoring
- `GET /metrics` - Application metrics
- `GET /health` - Service health status

## Error Handling

The application implements comprehensive error handling:

- **LLM Errors**: Automatic retry with exponential backoff
- **Vector Store Errors**: Graceful degradation and circuit breaker pattern
- **Rate Limiting**: Per-user rate limiting with Redis
- **Input Validation**: Pydantic model validation
- **Timeout Handling**: Configurable timeouts for all external calls

## Monitoring and Observability

### Metrics Collection
- Request/response times
- Token usage tracking
- Error rates and types
- Active session counts
- Vector store performance

### Logging
- Structured JSON logging
- Correlation IDs for request tracing
- Performance metrics logging
- Error tracking with context

### Health Checks
- LLM service connectivity
- Vector database status
- Redis connection health
- Memory and CPU usage

## Development

### Cursor Rules Integration

This project includes comprehensive `.cursorrules` that help with:
- LangChain best practices
- FastAPI async patterns
- RAG pipeline optimization
- Error handling strategies
- Testing patterns

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy src/
```

## Production Deployment

### Docker Deployment

```bash
# Build and deploy
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale api=3
```

### Environment Variables for Production

```env
ENVIRONMENT=production
LOG_LEVEL=INFO
LOG_FORMAT=json
DEBUG=false
RATE_LIMIT_REQUESTS=100
REQUEST_TIMEOUT=15
```

### Scaling Considerations

- Use Redis Cluster for high availability
- Implement load balancing for multiple API instances
- Monitor vector database query performance
- Set up proper logging aggregation
- Configure alerts for error rates and response times

## Troubleshooting

### Common Issues

1. **OpenAI API Errors**
   - Check API key validity
   - Monitor rate limits and quotas
   - Verify model availability

2. **Pinecone Connection Issues**
   - Validate API key and environment
   - Check index configuration
   - Monitor query performance

3. **Redis Connection Problems**
   - Verify Redis server status
   - Check connection URL format
   - Monitor memory usage

4. **Document Ingestion Failures**
   - Validate file formats
   - Check file size limits
   - Monitor embedding API calls

### Performance Optimization

- Implement response caching for frequent queries
- Use batch processing for document ingestion
- Optimize chunk size based on your use case
- Monitor and tune vector search parameters
- Implement connection pooling for databases

## Contributing

1. Follow the Cursor rules defined in `.cursorrules`
2. Write tests for new features
3. Update documentation
4. Follow conventional commit format
5. Ensure all CI checks pass

## License

MIT License - see LICENSE file for details. 