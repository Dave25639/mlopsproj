"""
FastAPI application for food image classification using trained ViT model.
"""

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from mlopsproj.model import ViTClassifier

logger = logging.getLogger(__name__)

# Global variables for model and classes
model: Optional[ViTClassifier] = None
class_names: List[str] = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_project_root() -> Path:
    """Get the project root directory."""
    # This file is at src/mlopsproj/api.py, so go up 2 levels to get project root
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    return project_root


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    global model, class_names

    # Get project root for resolving relative paths
    project_root = get_project_root()

    # Configuration - adjust these paths as needed
    checkpoint_path = os.getenv(
        "CHECKPOINT_PATH",
        str(project_root / "outputs/checkpoints/epoch=00-val/acc=0.9307.ckpt")  # Update to your best checkpoint
    )
    data_dir = os.getenv("DATA_DIR", str(project_root / "data"))
    num_classes = int(os.getenv("NUM_CLASSES", "101"))
    model_name = os.getenv("MODEL_NAME", "google/vit-base-patch16-224-in21k")

    try:
        # Load class names
        class_names = load_class_names(data_dir)
        if len(class_names) != num_classes:
            logger.warning(
                f"Number of classes in file ({len(class_names)}) "
                f"doesn't match num_classes ({num_classes})"
            )

        # Load model
        # If checkpoint_path is relative, resolve it relative to project root
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_absolute():
            checkpoint = project_root / checkpoint_path
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint}. "
                f"Please update CHECKPOINT_PATH environment variable."
            )

        model = load_model_from_checkpoint(
            str(checkpoint),
            num_classes=len(class_names),
            model_name=model_name,
        )

        logger.info("API startup completed successfully")

    except Exception as e:
        logger.error(f"Failed to initialize API: {e}", exc_info=True)
        raise

    yield  # Application is running

    # Shutdown (if needed, add cleanup code here)
    logger.info("API shutting down")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Food Classification API",
    description="API for classifying food images using Vision Transformer",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class PredictionRequest(BaseModel):
    """Request model for image prediction."""
    image_path: str  # Local file path
    top_k: int = 5


class PredictionItem(BaseModel):
    """Single prediction result."""
    class_name: str
    confidence: float
    rank: int


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[PredictionItem]
    top_prediction: str
    top_confidence: float


# Image preprocessing transform (same as validation/test)
def build_inference_transform():
    """Build the same transform used during validation/test."""
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def load_class_names(data_dir: str = "data") -> List[str]:
    """Load class names from classes.txt file."""
    classes_path = Path(data_dir) / "meta" / "classes.txt"
    if not classes_path.exists():
        raise FileNotFoundError(f"Classes file not found at {classes_path}")

    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(classes)} classes from {classes_path}")
    return classes


def load_model_from_checkpoint(
    checkpoint_path: str,
    num_classes: int,
    model_name: str = "google/vit-base-patch16-224-in21k",
) -> ViTClassifier:
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")

    # Load model with same hyperparameters as training
    model = ViTClassifier.load_from_checkpoint(
        checkpoint_path,
        num_classes=num_classes,
        model_name=model_name,
        strict=False,  # Allow some flexibility in loading
    )

    model.eval()
    model.to(device)
    logger.info(f"Model loaded successfully on device: {device}")
    return model


def load_image_from_path(file_path: str) -> Image.Image:
    """Load image from local file path and return PIL Image."""
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found at {file_path}")

        image = Image.open(path)
        # Convert to RGB if needed (handles RGBA, L, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load image from path: {str(e)}"
        )


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load image from bytes (uploaded file) and return PIL Image."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if needed (handles RGBA, L, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load image from uploaded file: {str(e)}"
        )


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model inference."""
    transform = build_inference_transform()
    tensor = transform(image)
    # Add batch dimension
    return tensor.unsqueeze(0)


def predict_image(image_tensor: torch.Tensor, top_k: int = 5) -> List[PredictionItem]:
    """Run model inference and return top-k predictions."""
    global model, class_names

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the API is properly initialized."
        )

    # Move tensor to device
    image_tensor = image_tensor.to(device)

    # Run inference
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)

    # Get top-k predictions
    top_probs, top_indices = torch.topk(probabilities[0], k=min(top_k, len(class_names)))

    predictions = []
    for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), start=1):
        predictions.append(
            PredictionItem(
                class_name=class_names[idx.item()],
                confidence=prob.item(),
                rank=rank
            )
        )

    return predictions


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Food Classification API",
        "status": "running",
        "model_loaded": model is not None,
        "num_classes": len(class_names) if class_names else 0,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict food class from local image file path.

    Args:
        request: PredictionRequest containing image_path and optional top_k

    Returns:
        PredictionResponse with top-k predictions
    """
    try:
        # Load image from local path
        logger.info(f"Loading image from path: {request.image_path}")
        image = load_image_from_path(request.image_path)

        # Preprocess image
        image_tensor = preprocess_image(image)

        # Get predictions
        predictions = predict_image(image_tensor, top_k=request.top_k)

        # Build response
        response = PredictionResponse(
            predictions=predictions,
            top_prediction=predictions[0].class_name,
            top_confidence=predictions[0].confidence,
        )

        logger.info(
            f"Prediction completed: {response.top_prediction} "
            f"(confidence: {response.top_confidence:.4f})"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/upload", response_model=PredictionResponse)
async def predict_upload(
    file: UploadFile = File(...),
    top_k: int = 5
):
    """
    Predict food class from uploaded image file.

    This endpoint accepts multipart/form-data with an image file.
    Perfect for frontend file uploads.

    Args:
        file: Uploaded image file
        top_k: Number of top predictions to return (default: 5)

    Returns:
        PredictionResponse with top-k predictions
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

        # Read uploaded file
        logger.info(f"Processing uploaded file: {file.filename}")
        image_bytes = await file.read()

        # Load image from bytes
        image = load_image_from_bytes(image_bytes)

        # Preprocess image
        image_tensor = preprocess_image(image)

        # Get predictions
        predictions = predict_image(image_tensor, top_k=top_k)

        # Build response
        response = PredictionResponse(
            predictions=predictions,
            top_prediction=predictions[0].class_name,
            top_confidence=predictions[0].confidence,
        )

        logger.info(
            f"Prediction completed: {response.top_prediction} "
            f"(confidence: {response.top_confidence:.4f})"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/classes")
async def get_classes():
    """Get list of all available classes."""
    return {
        "classes": class_names,
        "num_classes": len(class_names),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "mlopsproj.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
