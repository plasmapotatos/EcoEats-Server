import base64
import re
import json
from io import BytesIO
from flask import Flask, request, jsonify
from PIL import Image
from src.utils.request_utils import pil_to_base64, call_ollama
from src.utils.prompts import ANALYZE_FOOD_PROMPT, GENERATE_RECIPE_PROMPT

app = Flask(__name__)

def parse_llm_output(llm_string):
    """Function to parse LLM output
    Inputs:
        llm_string (str): This string should be from the LLM

    Outputs:
        dict: JSON output of the LLM, parsed in a structured manner
    """
    match = re.search(r'```json\s*(\{.*?\})\s*```', llm_string, re.DOTALL)
    print(match)
    
    if not match:
        raise ValueError("No valid JSON found in the LLM output")
    
    json_content = match.group(1) 
    return json.loads(json_content) 

@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    """Endpoint to analyze a base64-encoded image.
    
    Inputs:
        base64_image (str): "<base64-encoded image string>",
    
    The 'image' field should be the base64-encoded string of the image.
    """

    if "base64_image" not in request.json:
        return jsonify({"error": "No image file provided"}), 400

    try:
        base64_image = request.json["base64_image"]

        image_data = base64.b64decode(base64_image)

        image = Image.open(BytesIO(image_data))

        #just to test
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


@app.route("/generate_recipe", methods=["POST"])
def generate_recipe():
    """Endpoint to generate a recipe & image from a natural language ingredient list.

    Inputs:
        "ingredients_text": "<natural language ingredient list>"
    Outputs:
        "recipe": "<generated recipe text>",
        "image_prompt": "<used image prompt>",
        "image_base64": "<base64-encoded image string of the dish>"
    """

    if "ingredients_text" not in request.json:
        return jsonify({"error": "No ingredient list provided"}), 400
    try:
        ingredients = request.json["ingredients_text"]

        #Generate recipe from ingredients
        recipe_prompt = GENERATE_RECIPE_PROMPT.format(ingredients=ingredients)
        recipe_response = call_ollama("llama3:8b", recipe_prompt, [ingredients])['message']['content']
        print("Generated Recipe:\n", recipe_response)

        #Use the generated recipe as the image prompt
        image_prompt = f"A realistic photo of the final dish prepared from the following recipe:\n{recipe_response}"

        image = Image.new("RGB", (512, 512), "lightgray") 
        image_base64 = pil_to_base64(image)

        return jsonify({
            "recipe": recipe_response,
            "image_prompt": image_prompt,
            "image_base64": image_base64
        })
    except Exception as e:
        return jsonify({"error:" f"Failed to generate recipe or image: {str(e)}"}), 500
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

''' TODO: Replace the placeholder image block with a call to API for Stable Diffusion, passing image_prompt. '''

