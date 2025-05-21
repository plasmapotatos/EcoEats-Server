import base64
import requests
from PIL import Image
from src.utils.request_utils import pil_to_base64

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

if __name__ == "__main__":
    image_path = "apple.jpg"
    send_image_to_server(image_path)