import base64
import requests
from PIL import Image
from src.utils.request_utils import pil_to_base64
import time


def send_image_to_server(image_path: str):
    """Sends a base64-encoded image to the Flask server for analysis."""
    url = "http://192.168.1.40:5001/analyze_image"

    image = Image.open(image_path)

    base64_image = pil_to_base64(image)

    data = {
        "base64_image": base64_image,
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        response_text = response.json()
        print("Server response:", response_text)
    else:
        print(f"Failed to analyze image. HTTP {response.status_code}:", response.json())


def send_ingredients_to_server(ingredients_text: str = None, image_path: str = None):
    """Sends ingredients text to the Flask server to generate recipe & image. Raises error if neither image nor text provided"""
    url = "http://127.0.0.1:5001/generate_recipe"

    if not ingredients_text and not image_path:
        raise ValueError("Must provide at least ingredients_text or image_path")

    data = {}
    if ingredients_text:
        data["ingredients_text"] = ingredients_text
    if image_path:
        image = Image.open(image_path)
        base64_image = pil_to_base64(image)
        data["base64_image"] = base64_image
        print("image read successfully")

    response = requests.post(url, json=data)

    if response.status_code == 200:
        response_json = response.json()
        print("Generated recipe:", response_json.get("recipe"))
        print("Image prompt:", response_json.get("image_prompt"))

        # Decode the base64 image and save it locally for verification
        image_base64 = response_json.get("image_base64")
        if image_base64:
            image_data = base64.b64decode(image_base64)
            with open("generated_dish_test.jpg", "wb") as f:
                f.write(image_data)
            print("Generated image saved as 'generated_dish_test.jpg'")
    else:
        print(
            f"Failed to generate recipe. HTTP {response.status_code}:", response.json()
        )


def test_detect_foods_endpoint(
    image_path: str, url: str = "http://localhost:5001/detect_foods"
):
    # Open image with PIL
    image = Image.open(image_path)

    # Convert PIL image to base64 string
    base64_image = pil_to_base64(image)

    # Prepare JSON payload
    payload = {"base64_image": base64_image}

    # Send POST request
    response = requests.post(url, json=payload)

    return response.json()


def test_suggest_alternatives(llm_guided=False):
    url = "http://localhost:5001/suggest_alternatives"

    # Example input foods
    test_data = {
        "foods": ["doritos", "ground beef"],
        "llm_guided": llm_guided,
    }

    response = requests.post(url, json=test_data)

    print(f"Status Code: {response.status_code}")
    try:
        print("Response:")
        print(response.json())
    except Exception as e:
        print("Failed to parse JSON:", e)
        print("Raw response:")
        print(response.text)


if __name__ == "__main__":
    # image_path = "ingredients.jpg"
    # send_image_to_server(image_path)

    # Example ingredients to test generate_recipe endpoint
    ingredients = "2 tomatoes, 1 onion, 3 cloves garlic, salt, olive oil"
    start_time = time.time()

    # print(test_detect_foods_endpoint(image_path=image_path))
    # send_ingredients_to_server(image_path=image_path)
    test_suggest_alternatives(llm_guided=True)

    end_time = time.time()
    duration = end_time - start_time
    print(f"\nTotal time taken: {duration:.2f} seconds")
