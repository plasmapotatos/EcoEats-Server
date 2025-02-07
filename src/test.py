import base64
import requests
from PIL import Image
from src.utils.request_utils import pil_to_base64

def send_image_to_server(image_path: str):
    """Sends a base64-encoded image to the Flask server for analysis."""
    url = "http://localhost:5001/analyze_image"  # URL of your Flask server

    # Load the image from disk
    image = Image.open(image_path)
    
    # Convert the image to base64
    base64_image = pil_to_base64(image)
    
    # Prepare payload with image data
    data = {
        "base64_image": base64_image,
    }
    
    # Send POST request to Flask server
    response = requests.post(url, json=data)

    # Handle the response
    if response.status_code == 200:
        response_text = response.json()['message']['content']
        print("Server response:", response_text)
    else:
        print(f"Failed to analyze image. HTTP {response.status_code}:", response.json())

if __name__ == "__main__":
    # Path to the image file
    image_path = "apple.jpg"  # Make sure this is the correct path to the image
    
    # Send image to the server for analysis
    send_image_to_server(image_path)