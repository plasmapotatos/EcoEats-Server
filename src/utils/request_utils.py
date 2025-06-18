import requests
import json
from PIL import Image

import base64
from io import BytesIO
from ollama import chat

## need to run ollama run MODEL_NAME in a sep terminal

def pil_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a base64-encoded string.

    Inputs:
        image (Image.Image): A PIL image object.

    Outputs:
        str: The base64 string representation of the image.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG") 
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def call_ollama(model: str, prompt: str, images: list = None):
    """Calls the locally hosted Ollama chat model with a prompt and optional images.

    Inputs:
        model (str): The name of the Ollama model to use.
        prompt (str): The user input prompt.
        images (list, optional): A list of base64-encoded images.

    Outputs:
        dict: The response from the Ollama API, or an error message if the server is unreachable.
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
    }

    if images:
        payload["messages"][0]["images"] = images 

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        print(response.json())
        response.raise_for_status() 
        return response.json()
    
    except requests.ConnectionError:
        return {"error": "Ollama server is not running on port 11434."}
    
    except requests.Timeout:
        return {"error": "Request to Ollama timed out."}

    except requests.HTTPError as e:
        return {"error": f"HTTP error: {e}"}

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


if __name__ == "__main__":
    image = Image.open("apple.jpg")
    base64_image = pil_to_base64(image)
    response = call_ollama("llama3.2-vision", "what is in the image", [base64_image])
    print(response['message']['content'])