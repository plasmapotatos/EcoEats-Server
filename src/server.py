import base64
import re
import json
from io import BytesIO
import time
import traceback
from flask import Flask, request, jsonify
from PIL import Image
from src.utils.request_utils import pil_to_base64, call_ollama
from src.utils.prompts import (
    ANALYZE_FOOD_PROMPT,
    DETECT_FOODS_PROMPT,
    GENERATE_RECIPE_PROMPT,
    SUGGEST_ALTERNATIVES_PROMPT,
    TOUCHUP_RECIPE_PROMPT,
)
from diffusers import StableDiffusionPipeline
import torch
import csv
from src.utils.embedding_matcher import (
    load_model,
    load_embeddings,
    find_top_n_lower_emission_matches,
)

# small Stable Diffusion model - check doc for alternate options
pipe = StableDiffusionPipeline.from_pretrained(
    "OFA-Sys/small-stable-diffusion-v0",
    torch_dtype=torch.float32,  # or torch.float16 if your device supports it
)
# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")


model = load_model()
embedding_data = load_embeddings("data/embeddings_output.json")

#alternate stable diffusion model - may not be compatible with all devices
# pipe = DiffusionPipeline.from_pretrained(
#    "black-forest-labs/FLUX.1-dev",
#    torch_dtype=torch.float32 # FOR TIMOTHY: required for MPS (mac GPU) you might hv to change to make compatible
# )
# pipe.load_lora_weights("multimodalart/isometric-skeumorphic-3d-bnb")


# FOR TIMOTHY: check that PyTorch install supports MPS (can verify using below)
print(torch.backends.mps.is_available())  # should be true on Mac Studio

device = "mps" if torch.backends.mps.is_available() else "cpu"
pipe.to(device)

app = Flask(__name__)

def parse_llm_output(llm_string):
    """Helper function to parse llm(in our case ollama) output
    Inputs:
        llm_string (str): This string should be from the LLM

    Outputs:
        dict: JSON output of the LLM, parsed in a structured manner
    """
    match = re.search(r"```json\s*(\{.*?\})\s*```", llm_string, re.DOTALL)
    print(match)

    if not match:
        raise ValueError("No valid JSON found in the LLM output")

    json_content = match.group(1)
    return json.loads(json_content)


@app.route("/detect_foods", methods=["POST"])
def detect_foods():
    """
    Endpoint to analyze a base64-encoded image and detect foods.
    Separated into own endpoint from other two endpoints for cleaner feel.

    Inputs (JSON):
        {
            "base64_image": "<base64-encoded image string>"
        }

    Returns:
        {
            "foods": ["apple", "bread", "cheese", ...]
        }
    """

    print("Received request to detect foods")

    start = time.time()
    json_data = request.json
    print("Accessed request.json in", time.time() - start, "seconds")

    # check for image
    if "base64_image" not in request.json:
        print("No base64_image in request")
        return jsonify({"error": "No image file provided"}), 400

    try:
        # Decode the base64 image
        print("encoding image")
        base64_image = request.json["base64_image"]
        image_data = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_data))

        print("image encoded")
        image.save("debug_image.jpg")
        print("image saved")

        # Calling  model (TODO: decide on model type) - optional route: BLIP (image captioning)
        while True:
            try:
                print("calling ollama")
                response = call_ollama("llava:13b", DETECT_FOODS_PROMPT, [base64_image])
                raw_output = response["message"]["content"]
                print("LLM Output:", raw_output)

                # Parse model output (expects a JSON-like list)
                parsed = parse_llm_output(raw_output)
                return jsonify(parsed), 200
            except Exception as e:
                print(f"Failed to parse LLM output: {e}")
                print("Retrying...")

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500


def llm_generate_alternatives(original_food: str, candidates: list[dict]) -> list[dict]:
    print(candidates)
    candidates_str = "\n".join(
        f"- {c['Name']} (CO2: {c['CO2']:.2f} kg CO2-eq/kg, Similarity: {c['Cosine Similarity']:.4f})"
        for c in candidates
    )
    prompt = (
        f"Original food: '{original_food}'\n"
        f"Candidate alternatives:\n{candidates_str}\n\n" + SUGGEST_ALTERNATIVES_PROMPT
    )
    while True:
        try:
            response = call_ollama("llava:13b", prompt, [])
            content = response["message"]["content"]

            # sometimes llm will use code fences w json, other times not
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            parsed = json.loads(content)
            return parsed
        except Exception as e:
            print(f"[Retrying] Failed to parse LLM output for '{original_food}': {e}")
            print("Response content:", content)
            continue


@app.route("/suggest_alternatives", methods=["POST"])
def suggest_alternatives():
    print("Received request to suggest alternatives")
    data = request.get_json()
    if not data or "foods" not in data:
        return jsonify({"error": "Missing 'foods' field"}), 400

    foods = data["foods"]
    llm_guided = data.get("llm_guided", False)

    with open("data/carbonData.csv", newline="") as f:
        reader = csv.DictReader(f)
        carbon_data = list(reader)

    response = {}

    for food in foods:
        candidates = find_top_n_lower_emission_matches(
            food, embedding_data, carbon_data, model, n=10
        )

        if llm_guided:
            llm_results = llm_generate_alternatives(food, candidates)
            formatted = []
            for alt in llm_results:
                formatted.append(
                    {
                        "Name": alt.get("Name"),
                        "Justification": alt.get("Justification"),
                        "CO2": alt.get("CO2"),
                        "Category": alt.get("Category", "Other"),
                    }
                )
            response[food] = formatted
        else:
            top5 = candidates[:5]
            formatted = [
                {
                    "Name": c["Name"],
                    "Similarity": round(c["Cosine Similarity"], 4),
                    "CO2": round(c["CO2"], 4),
                    "Category": c.get("Category", "Other"),
                }
                for c in top5
            ]
            response[food] = formatted

    print("Response:", response)

    return jsonify(response), 200

# Below is the old route to generate alternatives (irrelevant currently)
@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    """Endpoint to analyze a base64-encoded image.

    Inputs:
        base64_image (str): "<base64-encoded image string>",

    The 'image' field should be the base64-encoded string of the image.
    """

    if "base64_image" not in request.json:
        print("No base64_image in request")
        return jsonify({"error": "No image file provided"}), 400

    try:
        base64_image = request.json["base64_image"]

        image_data = base64.b64decode(base64_image)

        image = Image.open(BytesIO(image_data))

        # just to test
        image.save("image.jpg")
        while True:
            try:
                response = call_ollama(
                    "llava:13b", ANALYZE_FOOD_PROMPT, [base64_image]
                )["message"]["content"]
                print(response, type(response))
                parsed_response = parse_llm_output(response)
                break
            except Exception as e:
                print(f"Failed to parse LLM output: {str(e)}")
                print("Retrying...")
        return parsed_response

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500


def extract_json_from_response(text):
    # Helper function to convert llm response to json so it is easier for frontend to retrieve
    print("Raw response text:", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


@app.route("/generate_recipe", methods=["POST"])
def generate_recipe():
    """Endpoint to generate a recipe & image from a textual ingredient list. Recipe can be "touched-up" with user preferences such as 
    no oven or vegan."""

    data = request.json
    ingredients = data.get("ingredients_text")
    print("ingredients: ", ingredients)
    preferences = data.get("preferences")
    print("\n preferences: ", preferences)
    previous_recipe = data.get("previous_recipe")
    print("\n previous recipe: ", previous_recipe)

    try:
        # if it is second time being pressed then touch up mode activated
        if previous_recipe and preferences:
            # touch up mode
            print("Touch-up mode activated")
            original_recipe_str = json.dumps(previous_recipe, indent=2)
            prompt = TOUCHUP_RECIPE_PROMPT.format(
                original_recipe=original_recipe_str,
                preferences=preferences,
                ingredients=ingredients,
            )
        elif ingredients:
            # no old recipe - new generation mode
            print("Fresh generation mode")
            prompt = GENERATE_RECIPE_PROMPT.format(ingredients=ingredients)
        else:
            return jsonify(
                {
                    "error": "Must provide either ingredients_text or previous_recipe with preferences"
                }
            ), 400

        # call ollama - TIMOTHY
        response = call_ollama("llama3.2:latest", prompt)
        raw_response = response["message"]["content"]
        print("Model Response:\n", raw_response)

        recipe_json = extract_json_from_response(raw_response)
        if recipe_json is None:
            return jsonify(
                {"error": "Failed to parse recipe JSON", "raw_response": raw_response}
            ), 500

        title = recipe_json.get("title", "Generated Dish")
        ingredients_list = recipe_json.get("ingredients", [])
        steps_list = recipe_json.get("steps", [])

        # use the generated recipe as the image prompt using the stable diffusion model initalized above - cuts off at 77 tokens bc of CLIP
        image_prompt = f"A realistic photo of the final dish prepared from this recipe titled '{title}' with the following steps: {' '.join(steps_list)}"
        print("image prompt: ", image_prompt)
        image = pipe(image_prompt, num_inference_steps=10).images[0]
        image.save("generated_dish.jpg")
        image_base64 = pil_to_base64(image)

        return jsonify(
            {
                "title": title,
                "ingredients": ingredients_list,
                "steps": steps_list,
                "image_base64": image_base64,
            }
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to generate recipe or image: {str(e)}"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=50001, debug=True)
