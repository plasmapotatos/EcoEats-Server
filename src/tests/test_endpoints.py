import base64
import json
import pytest
from PIL import Image
from io import BytesIO
from unittest.mock import patch, MagicMock
from src.server import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def get_dummy_base64_image():
    image = Image.new("RGB", (10, 10), color="red")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# /detect_foods
@patch("src.utils.request_utils.call_ollama")
def test_detect_foods_success(mock_ollama, client):
    mock_ollama.return_value = {
        "message": {
            "content": "```json\n{\"foods\": [\"apple\", \"banana\"]}\n```"
        }
    }
    response = client.post("/detect_foods", json={"base64_image": get_dummy_base64_image()})
    assert response.status_code == 200
    assert "foods" in response.json


def test_detect_foods_no_image(client):
    response = client.post("/detect_foods", json={})
    assert response.status_code == 400
    assert "error" in response.json


# /suggest_alternatives
@patch("src.utils.embedding_matcher.find_top_n_lower_emission_matches")
@patch("src.utils.request_utils.call_ollama")
def test_suggest_alternatives_llm(mock_ollama, mock_find, client):
    mock_find.return_value = [
        {"Name": "Tofu", "CO2": 1.0, "Cosine Similarity": 0.9, "Category": "Plant-based"}
    ]
    mock_ollama.return_value = {
        "message": {
            "content": json.dumps([
                {
                    "Name": "Tofu",
                    "Justification": "Lower CO2",
                    "CO2": 1.0,
                    "Category": "Plant-based"
                }
            ])
        }
    }
    payload = {"foods": ["beef"], "llm_guided": True}
    response = client.post("/suggest_alternatives", json=payload)
    assert response.status_code == 200
    assert "beef" in response.json


@patch("src.utils.embedding_matcher.find_top_n_lower_emission_matches")
def test_suggest_alternatives_basic(mock_find, client):
    mock_find.return_value = [
        {"Name": "Tofu", "CO2": 1.0, "Cosine Similarity": 0.9, "Category": "Plant-based"}
    ]
    payload = {"foods": ["beef"]}
    response = client.post("/suggest_alternatives", json=payload)
    assert response.status_code == 200
    assert "beef" in response.json


def test_suggest_alternatives_missing_foods(client):
    response = client.post("/suggest_alternatives", json={})
    assert response.status_code == 400
    assert "error" in response.json


# /analyze_image
@patch("src.utils.request_utils.call_ollama")
def test_analyze_image_success(mock_ollama, client):
    mock_ollama.return_value = {
        "message": {
            "content": "```json\n{\"summary\": \"This is a healthy meal.\"}\n```"
        }
    }
    response = client.post("/analyze_image", json={"base64_image": get_dummy_base64_image()})
    assert response.status_code == 200
    assert "item" in response.json
    assert "alternatives" in response.json


def test_analyze_image_no_image(client):
    response = client.post("/analyze_image", json={})
    assert response.status_code == 400
    assert "error" in response.json


# /generate_recipe
@patch("src.server.pipe")
@patch("src.utils.request_utils.call_ollama")
def test_generate_recipe_fresh(mock_ollama, mock_pipe, client):
    mock_ollama.return_value = {
        "message": {
            "content": json.dumps({
                "title": "Simple Salad",
                "ingredients": ["lettuce", "tomato"],
                "steps": ["Mix and serve"]
            })
        }
    }
    dummy_img = Image.new("RGB", (10, 10))
    mock_pipe.return_value = MagicMock(images=[dummy_img])
    payload = {"ingredients_text": "lettuce, tomato"}
    response = client.post("/generate_recipe", json=payload)
    assert response.status_code == 200
    assert "title" in response.json
    assert "image_base64" in response.json


@patch("src.server.pipe")
@patch("src.utils.request_utils.call_ollama")
def test_generate_recipe_touchup(mock_ollama, mock_pipe, client):
    mock_ollama.return_value = {
        "message": {
            "content": json.dumps({
                "title": "Vegan Salad",
                "ingredients": ["lettuce", "tomato"],
                "steps": ["Mix and serve"]
            })
        }
    }
    dummy_img = Image.new("RGB", (10, 10))
    mock_pipe.return_value = MagicMock(images=[dummy_img])
    payload = {
        "ingredients_text": "lettuce, tomato",
        "preferences": "vegan",
        "previous_recipe": {
            "title": "Salad",
            "ingredients": ["lettuce", "egg"],
            "steps": ["Boil egg", "Mix"]
        }
    }
    response = client.post("/generate_recipe", json=payload)
    assert response.status_code == 200
    assert "title" in response.json
    assert "image_base64" in response.json


def test_generate_recipe_missing_all(client):
    response = client.post("/generate_recipe", json={})
    assert response.status_code == 400
    assert "error" in response.json
