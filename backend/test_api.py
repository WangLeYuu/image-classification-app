import pytest
from fastapi.testclient import TestClient
from main import app
import io
from PIL import Image


client = TestClient(app)


def create_test_image():
    def generate_image():
        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes
    return generate_image()


def test_root_endpoint():
    def check_root():
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data
    check_root()


def test_health_endpoint():
    def check_health():
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "image-classification-api"
    check_health()


def test_predict_endpoint_success():
    def check_prediction():
        img_bytes = create_test_image()
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "top_prediction" in data
        assert len(data["predictions"]) == 5
        assert "class" in data["top_prediction"]
        assert "confidence" in data["top_prediction"]
    check_prediction()


def test_predict_endpoint_invalid_file_type():
    def check_invalid_type():
        response = client.post(
            "/predict",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
        )
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
    check_invalid_type()


def test_predict_endpoint_no_file():
    def check_no_file():
        response = client.post("/predict")
        assert response.status_code == 422
    check_no_file()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
