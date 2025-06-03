"""
Metrics Collection for RAG Chatbot
Tracks performance, usage, and error metrics.
"""

import time
import asyncio
import psutil
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and provides application metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.chat_completions = 0
        self.total_tokens = 0
        self.active_sessions = set()
        self.errors = defaultdict(int)
        self.response_times = deque(maxlen=1000)  # Keep last 1000 response times
        self.token_usage = deque(maxlen=1000)  # Keep last 1000 token counts
        self.error_history = deque(maxlen=100)  # Keep last 100 errors
        
    async def track_chat_completion(
        self,
        session_id: str,
        query_length: int,
        response_length: int,
        duration: float,
        source_docs_count: int,
        token_count: Optional[int] = None
    ):
        """Track a chat completion event."""
        self.chat_completions += 1
        self.active_sessions.add(session_id)
        self.response_times.append(duration)
        
        if token_count:
            self.total_tokens += token_count
            self.token_usage.append(token_count)
        
        logger.info(
            "Chat completion tracked",
            extra={
                "session_id": session_id,
                "query_length": query_length,
                "response_length": response_length,
                "duration": duration,
                "source_docs_count": source_docs_count,
                "token_count": token_count
            }
        )
    
    async def track_error(self, error_type: str, error_message: str):
        """Track an error occurrence."""
        self.errors[error_type] += 1
        self.error_history.append({
            "type": error_type,
            "message": error_message,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.error(
            "Error tracked",
            extra={
                "error_type": error_type,
                "error_message": error_message
            }
        )
    
    async def track_document_ingestion(
        self,
        document_count: int,
        processing_time: float,
        success_count: int,
        error_count: int
    ):
        """Track document ingestion metrics."""
        logger.info(
            "Document ingestion tracked",
            extra={
                "document_count": document_count,
                "processing_time": processing_time,
                "success_count": success_count,
                "error_count": error_count
            }
        )
    
    async def track_vector_search(
        self,
        query: str,
        results_count: int,
        search_time: float,
        vector_store_type: str
    ):
        """Track vector search performance."""
        logger.info(
            "Vector search tracked",
            extra={
                "query_length": len(query),
                "results_count": results_count,
                "search_time": search_time,
                "vector_store_type": vector_store_type
            }
        )
    
    def get_average_response_time(self) -> float:
        """Calculate average response time."""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    def get_error_rate(self) -> float:
        """Calculate error rate as percentage."""
        if self.chat_completions == 0:
            return 0.0
        total_errors = sum(self.errors.values())
        return (total_errors / (self.chat_completions + total_errors)) * 100
    
    def get_uptime_seconds(self) -> float:
        """Get application uptime in seconds."""
        return time.time() - self.start_time
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def get_cpu_usage_percent(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=1)
        except Exception:
            return 0.0
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "total_chats": self.chat_completions,
            "active_sessions": len(self.active_sessions),
            "average_response_time": self.get_average_response_time(),
            "error_rate": self.get_error_rate(),
            "total_tokens_used": self.total_tokens,
            "uptime_seconds": self.get_uptime_seconds(),
            "memory_usage_mb": self.get_memory_usage_mb(),
            "cpu_usage_percent": self.get_cpu_usage_percent(),
            "recent_errors": list(self.error_history)[-10:],  # Last 10 errors
            "error_breakdown": dict(self.errors),
            "performance": {
                "min_response_time": min(self.response_times) if self.response_times else 0,
                "max_response_time": max(self.response_times) if self.response_times else 0,
                "median_response_time": self._get_median(self.response_times),
                "p95_response_time": self._get_percentile(self.response_times, 95),
                "p99_response_time": self._get_percentile(self.response_times, 99)
            }
        }
    
    def _get_median(self, values: deque) -> float:
        """Calculate median value."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n % 2 == 0:
            return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        return sorted_values[n//2]
    
    def _get_percentile(self, values: deque, percentile: int) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        if index >= len(sorted_values):
            index = len(sorted_values) - 1
        return sorted_values[index]
    
    async def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old session tracking (called periodically)."""
        # In a real implementation, you'd track session timestamps
        # and remove old ones based on actual activity
        if len(self.active_sessions) > 1000:  # Arbitrary limit
            # Keep only the most recent 800 sessions
            self.active_sessions = set(list(self.active_sessions)[-800:])
        
        logger.info(f"Cleaned up old sessions, {len(self.active_sessions)} active")


class PrometheusMetrics:
    """Prometheus-compatible metrics exporter."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.collector = metrics_collector
    
    async def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        metrics = await self.collector.get_metrics()
        
        prometheus_format = []
        
        # Basic counters
        prometheus_format.append(f"rag_chatbot_total_chats {metrics['total_chats']}")
        prometheus_format.append(f"rag_chatbot_active_sessions {metrics['active_sessions']}")
        prometheus_format.append(f"rag_chatbot_total_tokens {metrics['total_tokens_used']}")
        prometheus_format.append(f"rag_chatbot_uptime_seconds {metrics['uptime_seconds']}")
        
        # Performance metrics
        prometheus_format.append(f"rag_chatbot_avg_response_time {metrics['average_response_time']}")
        prometheus_format.append(f"rag_chatbot_error_rate {metrics['error_rate']}")
        prometheus_format.append(f"rag_chatbot_memory_usage_mb {metrics['memory_usage_mb']}")
        prometheus_format.append(f"rag_chatbot_cpu_usage_percent {metrics['cpu_usage_percent']}")
        
        # Response time percentiles
        perf = metrics['performance']
        prometheus_format.append(f"rag_chatbot_response_time_p95 {perf['p95_response_time']}")
        prometheus_format.append(f"rag_chatbot_response_time_p99 {perf['p99_response_time']}")
        
        # Error breakdown
        for error_type, count in metrics['error_breakdown'].items():
            safe_error_type = error_type.replace('-', '_').replace(' ', '_').lower()
            prometheus_format.append(f"rag_chatbot_errors_{{type=\"{safe_error_type}\"}} {count}")
        
        return '\n'.join(prometheus_format) 