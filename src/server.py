import base64
from io import BytesIO
from flask import Flask, request, jsonify
from PIL import Image
from utils.request_utils import pil_to_base64, call_ollama
from utils.prompts import ANALYZE_FOOD_PROMPT

app = Flask(__name__)


@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    """Endpoint to analyze a base64-encoded image using Ollama's vision model.
    
    Expected input:
        {
            "base64_image": "<base64-encoded image string>",
        }
    
    The 'image' field should be the base64-encoded string of the image.
    """

    if "base64_image" not in request.json:
        return jsonify({"error": "No image file provided"}), 400

    try:
        base64_image = request.json["base64_image"]
        response = call_ollama("llama3.2-vision", ANALYZE_FOOD_PROMPT, [base64_image])
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == "__main__":
    # image = Image.open("apple.jpg")
    # base64_image = pil_to_base64(image)
    # response = call_ollama("llama3.2-vision", "What is in this image?", [base64_image])
    # print(response)
    app.run(host="0.0.0.0", port=5001, debug=True)