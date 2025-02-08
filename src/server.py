import base64
import re
import json
from io import BytesIO
from flask import Flask, request, jsonify
from PIL import Image
from src.utils.request_utils import pil_to_base64, call_ollama
from src.utils.prompts import ANALYZE_FOOD_PROMPT

app = Flask(__name__)

def parse_llm_output(llm_string):
    # Extract JSON content between ```json and ```
    match = re.search(r'```json\s*(\{.*?\})\s*```', llm_string, re.DOTALL)
    print(match)
    
    if not match:
        raise ValueError("No valid JSON found in the LLM output")
    
    json_content = match.group(1)  # Extract the JSON part
    return json.loads(json_content)  # Parse and return as dictionary

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

        image_data = base64.b64decode(base64_image)

        # Load the image from the base64-encoded string
        image = Image.open(BytesIO(image_data))

        # save image to disk
        image.save("image.jpg")
        while True:
            try:
                response = call_ollama("llava:13b", ANALYZE_FOOD_PROMPT, [base64_image])['message']['content']
                print(response, type(response))
                parsed_response = parse_llm_output(response)
                break
            except Exception as e:
                print(f"Failed to parse LLM output: {str(e)}")
                print("Retrying...")
        return parsed_response

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)