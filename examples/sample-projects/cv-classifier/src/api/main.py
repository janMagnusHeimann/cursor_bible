"""
Computer Vision Classifier API
Production-ready image classification service with PyTorch and FastAPI.
"""

import os
import logging
import asyncio
from typing import List, Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

from ..models.classifier import ImageClassifier
from ..models.model_manager import ModelManager
from ..utils.image_processing import ImageProcessor
from ..utils.validation import validate_image_file
from ..api.models import (
    ClassificationResponse, 
    BatchClassificationResponse,
    HealthCheck,
    ModelInfo
)
from ..api.dependencies import get_model_manager, get_image_processor
from ..config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global services
model_manager: ModelManager = None
image_processor: ImageProcessor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    global model_manager, image_processor
    
    try:
        # Initialize services
        settings = Settings()
        
        # Initialize model manager
        model_manager = ModelManager(settings)
        await model_manager.initialize()
        
        # Initialize image processor
        image_processor = ImageProcessor(settings)
        
        # Warm up models
        await model_manager.warmup_models()
        
        logger.info("CV Classifier API initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize CV API: {e}")
        raise
    finally:
        # Cleanup
        if model_manager:
            await model_manager.cleanup()
        logger.info("CV Classifier API cleaned up")


# Create FastAPI app
app = FastAPI(
    title="Computer Vision Classifier API",
    description="Production-ready image classification service",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint with model status."""
    try:
        # Check model availability
        model_status = await model_manager.check_model_health()
        
        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        
        return HealthCheck(
            status="healthy",
            models=model_status,
            gpu_available=gpu_available,
            gpu_count=gpu_count,
            torch_version=torch.__version__
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/classify", response_model=ClassificationResponse)
async def classify_image(
    file: UploadFile = File(...),
    model_name: str = "resnet50",
    top_k: int = 5,
    threshold: float = 0.1,
    model_manager: ModelManager = Depends(get_model_manager),
    image_processor: ImageProcessor = Depends(get_image_processor)
):
    """Classify a single image."""
    try:
        # Validate uploaded file
        await validate_image_file(file)
        
        # Read and process image
        image_bytes = await file.read()
        image = await image_processor.load_image_from_bytes(image_bytes)
        
        # Get model
        classifier = await model_manager.get_model(model_name)
        
        # Perform classification
        results = await classifier.classify_single(
            image=image,
            top_k=top_k,
            threshold=threshold
        )
        
        return ClassificationResponse(
            filename=file.filename,
            model_name=model_name,
            predictions=results["predictions"],
            inference_time=results["inference_time"],
            confidence_threshold=threshold
        )
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post("/classify/batch", response_model=BatchClassificationResponse)
async def classify_batch(
    files: List[UploadFile] = File(...),
    model_name: str = "resnet50",
    top_k: int = 5,
    threshold: float = 0.1,
    background_tasks: BackgroundTasks = None,
    model_manager: ModelManager = Depends(get_model_manager),
    image_processor: ImageProcessor = Depends(get_image_processor)
):
    """Classify multiple images in batch."""
    try:
        if len(files) > 20:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size too large (max 20)")
        
        # Validate all files first
        for file in files:
            await validate_image_file(file)
        
        # Process all images
        images = []
        filenames = []
        
        for file in files:
            image_bytes = await file.read()
            image = await image_processor.load_image_from_bytes(image_bytes)
            images.append(image)
            filenames.append(file.filename)
        
        # Get model
        classifier = await model_manager.get_model(model_name)
        
        # Perform batch classification
        results = await classifier.classify_batch(
            images=images,
            filenames=filenames,
            top_k=top_k,
            threshold=threshold
        )
        
        return BatchClassificationResponse(
            model_name=model_name,
            batch_size=len(files),
            results=results["predictions"],
            total_inference_time=results["total_time"],
            average_inference_time=results["avg_time"],
            confidence_threshold=threshold
        )
        
    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch classification failed: {str(e)}")


@app.get("/models", response_model=List[ModelInfo])
async def list_models(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """List available models."""
    try:
        models = await model_manager.list_available_models()
        return models
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")


@app.post("/models/{model_name}/load")
async def load_model(
    model_name: str,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Load a specific model."""
    try:
        # Load model in background
        background_tasks.add_task(model_manager.load_model, model_name)
        
        return {"message": f"Loading model {model_name}", "status": "loading"}
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.delete("/models/{model_name}/unload")
async def unload_model(
    model_name: str,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Unload a specific model to free memory."""
    try:
        await model_manager.unload_model(model_name)
        return {"message": f"Model {model_name} unloaded", "status": "unloaded"}
    except Exception as e:
        logger.error(f"Error unloading model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")


@app.get("/models/{model_name}/info", response_model=ModelInfo)
async def get_model_info(
    model_name: str,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get detailed information about a specific model."""
    try:
        info = await model_manager.get_model_info(model_name)
        return info
    except Exception as e:
        logger.error(f"Error getting model info for {model_name}: {e}")
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")


@app.post("/models/{model_name}/warmup")
async def warmup_model(
    model_name: str,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Warm up a specific model."""
    try:
        background_tasks.add_task(model_manager.warmup_model, model_name)
        return {"message": f"Warming up model {model_name}", "status": "warming_up"}
    except Exception as e:
        logger.error(f"Error warming up model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to warm up model: {str(e)}")


@app.get("/metrics")
async def get_metrics(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get API performance metrics."""
    try:
        metrics = await model_manager.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 