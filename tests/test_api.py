"""
Functional tests for the FastAPI application.

Tests cover:
- Health check endpoint
- Root endpoint
- Classes endpoint
- Prediction endpoints (with mock/simple image)
- Error handling
"""

import io
import pytest
import torch
from unittest.mock import patch
from PIL import Image
from fastapi.testclient import TestClient
from src.mlopsproj import api


@pytest.fixture(scope="function", autouse=True)
def mock_model_and_classes():
    """Automatically mock the model and class_names for all tests."""
    # Create mock class names (101 classes for Food-101)
    mock_class_names = [f"class_{i:03d}" for i in range(101)]

    # Create a callable mock model that returns logits when called
    # The model is called like: logits = model(image_tensor)
    # We need to return logits with shape (batch_size, num_classes) = (1, 101)
    class MockModel:
        """Mock model class that's callable."""
        def __call__(self, image_tensor):
            batch_size = image_tensor.shape[0]
            return torch.randn(batch_size, 101)

        def eval(self):
            pass

        def to(self, device):
            return self

    mock_model = MockModel()

    # Patch the load functions to return our mocks
    def mock_load_class_names(data_dir):
        return mock_class_names

    def mock_load_model_from_checkpoint(checkpoint_path, num_classes, model_name):
        return mock_model

    # Patch both the load functions and set model/class_names directly
    with patch.object(api, 'load_class_names', mock_load_class_names), \
         patch.object(api, 'load_model_from_checkpoint', mock_load_model_from_checkpoint):
        # Set the model and class_names directly to ensure they're available
        api.model = mock_model
        api.class_names = mock_class_names

        yield mock_model, mock_class_names

        # Cleanup
        api.model = None
        api.class_names = []


@pytest.fixture
def client(mock_model_and_classes):
    """Create a test client for the API with mocked model."""
    return TestClient(api.app)


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    # Create a simple 224x224 RGB image
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "num_classes" in data


def test_classes_endpoint(client):
    """Test the classes endpoint."""
    response = client.get("/classes")
    assert response.status_code == 200
    data = response.json()
    assert "classes" in data
    assert isinstance(data["classes"], list)


def test_predict_upload_endpoint(client, sample_image):
    """Test the predict/upload endpoint with file upload."""
    response = client.post(
        "/predict/upload",
        files={"file": ("test_image.jpg", sample_image, "image/jpeg")},
        data={"top_k": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "top_prediction" in data
    assert "top_confidence" in data
    assert len(data["predictions"]) <= 5
    assert isinstance(data["top_confidence"], float)
    assert 0 <= data["top_confidence"] <= 1


def test_predict_upload_invalid_file(client):
    """Test predict/upload with invalid file type."""
    response = client.post(
        "/predict/upload",
        files={"file": ("test.txt", b"not an image", "text/plain")},
        data={"top_k": 5}
    )
    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_predict_upload_top_k_validation(client, sample_image):
    """Test that top_k parameter works correctly."""
    for top_k in [1, 3, 5, 10]:
        response = client.post(
            "/predict/upload",
            files={"file": ("test_image.jpg", sample_image, "image/jpeg")},
            data={"top_k": top_k}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) <= top_k
