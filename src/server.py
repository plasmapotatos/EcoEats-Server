import base64
import re
import json
from io import BytesIO
from flask import Flask, request, jsonify
from PIL import Image
from src.utils.request_utils import pil_to_base64, call_ollama
from src.utils.prompts import (
    ANALYZE_FOOD_PROMPT,
    DETECT_FOODS_PROMPT,
    GENERATE_RECIPE_PROMPT,
    SUGGEST_ALTERNATIVES_PROMPT,
)
from diffusers import StableDiffusionPipeline
import torch
import csv
from src.utils.embedding_matcher import (
    load_model,
    load_embeddings,
    find_top_n_lower_emission_matches,
)

# small Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "OFA-Sys/small-stable-diffusion-v0",
    torch_dtype=torch.float32,  # or torch.float16 if your device supports it
)


model = load_model()
embedding_data = load_embeddings("data/embeddings_output.json")
# Load Stable Diffusion model - "on top of"
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
    """Function to parse LLM output
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

    Inputs (JSON):
        {
            "base64_image": "<base64-encoded image string>"
        }

    Returns:
        {
            "foods": ["apple", "bread", "cheese", ...]
        }
    """

    if "base64_image" not in request.json:
        print("No base64_image in request")
        return jsonify({"error": "No image file provided"}), 400

    try:
        # Decode the base64 image
        base64_image = request.json["base64_image"]
        image_data = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_data))

        # (Optional) Save for debugging
        image.save("debug_image.jpg")

        # Call the vision-language model (e.g., LLaVA via Ollama)
        while True:
            try:
                response = call_ollama("llava:13b", DETECT_FOODS_PROMPT, [base64_image])
                raw_output = response["message"]["content"]
                print("LLM Output:", raw_output)

                # Parse model output (expects a JSON-like list)
                parsed = parse_llm_output(raw_output)
                return jsonify({"foods": parsed}), 200
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

            # Remove code fences if present
            if "```" in content:
                content = content.split("```")[1].strip()

            parsed = json.loads(content)
            return parsed
        except Exception as e:
            print(f"[Retrying] Failed to parse LLM output for '{original_food}': {e}")
            print("Response content:", content)
            continue


@app.route("/suggest_alternatives", methods=["POST"])
def suggest_alternatives():
    data = request.get_json()
    if not data or "foods" not in data:
        return jsonify({"error": "Missing 'foods' field"}), 400

    foods = data["foods"]
    llm_guided = data.get("llm_guided", False)

    with open("data/carbonData.csv", newline="") as f:
        reader = csv.DictReader(f)
        carbon_data = list(reader)

    response = []

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
                        "CO2-eq/kg": alt.get("CO2"),
                    }
                )
            response.append({"original": food, "matches": formatted})

        else:
            top5 = candidates[:5]
            formatted = [
                {
                    "Name": c["Name"],
                    "Similarity": round(c["Cosine Similarity"], 4),
                    "CO2-eq/kg": round(c["CO2"], 4),
                }
                for c in top5
            ]
            response.append({"original": food, "matches": formatted})

    return jsonify({"alternatives": response}), 200


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


@app.route("/generate_recipe", methods=["POST"])
def generate_recipe():
    """Endpoint to generate a recipe & image from a natural language ingredient list or image of ingredient(s).

    Inputs:
        "ingredients_text": "<natural language ingredient list>" (optional)
    Outputs:
        "recipe": "<generated recipe text>",
        "image_base64": "<base64-encoded image string of the dish>"
    """

    if "ingredients_text" not in request.json:
        return jsonify({"error": "No ingredient list provided"}), 400
    try:
        ingredients = request.json["ingredients_text"]


        #Generate recipe from ingredients
        recipe_prompt = GENERATE_RECIPE_PROMPT.format(ingredients=ingredients)
        recipe_response = call_ollama("llama3.2-vision:latest", recipe_prompt)['message']['content']
        
        #print("Ollama response:", recipe_response)        
        #recipe_response = check notes app
        
        print("Generated Recipe:\n", recipe_response)

        #Use the generated recipe as the image prompt
        image_prompt = f"A realistic photo of the final dish prepared from the following recipe:\n{recipe_response}"
        print("Image prompt:", image_prompt)

        #image = pipe(image_prompt).images[0]

        #updated image calling
        image = pipe(image_prompt, num_inference_steps=10, guidance_scale=7.5).images[0]

        image.save("generated_dish.jpg")
        image_base64 = pil_to_base64(image)

        return jsonify({
            "recipe": recipe_response,
            "image_prompt": image_prompt,
            "image_base64": image_base64
        })
    except Exception as e:
        return jsonify({"error": f"Failed to generate recipe or image: {str(e)}"}), 500 
    
    '''
    try:
        ingredients_text = request.json.get("ingredients_text")
        base64_image = request.json.get("base64_image")

        if not ingredients_text and not base64_image:
            return jsonify({"error": "No ingredient text or image provided."}), 400

        # Choose model based on input type
        model_name = "llava:13b" if base64_image else "llama3.2:latest"

        if base64_image and ingredients_text:
            # Both inputs present — prompt that includes text and image
            prompt = GENERATE_RECIPE_PROMPT.format(
                ingredients=(
                    f"Ingredients text: {ingredients_text}\n"
                    "Please also check the attached image for ingredients. "
                    "Create a proper ingredient list by combining both the image and text. "
                    "(Avoid duplicates as best as possible.)"
                )
            )
            inputs = [base64_image]
        elif base64_image:
            # Only image provided — placeholder text prompt + image input
            prompt = GENERATE_RECIPE_PROMPT.format(
                ingredients="Please check the attached image for ingredients."
            )
            inputs = [base64_image]
        else:
            # Only text provided — no image input
            prompt = GENERATE_RECIPE_PROMPT.format(ingredients=ingredients_text)
            inputs = []

        # Call Ollama with prompt + optional image(s)
        # recipe_response = call_ollama("llama3.2-vision:latest", prompt, inputs)['message']['content']
        response = call_ollama(model_name, prompt, inputs)
        print("Ollama raw response:", response)
        if "error" in response:
            return jsonify({"error": f"Ollama error: {response['error']}"}), 500
        if "message" not in response or "content" not in response["message"]:
            return jsonify(
                {"error": f"Unexpected Ollama response format: {response}"}
            ), 500

        recipe_response = response["message"]["content"]
        print("Generated Recipe:\n", recipe_response)

        # print("Generated Recipe:\n", recipe_response)

        # Generate image prompt and image from recipe text
        image_prompt = f"A realistic photo of the final dish prepared from the following recipe:\n{recipe_response}"
        image = pipe(image_prompt, num_inference_steps=10, guidance_scale=7.5).images[0]

        image.save("generated_dish.jpg")
        image_base64_out = pil_to_base64(image)

        return jsonify(
            {
                "recipe": recipe_response,
                "image_prompt": image_prompt,
                "image_base64": image_base64_out,
            }
        )
    except Exception as e:
        return jsonify({"error": f"Failed to generate recipe or image: {str(e)}"}), 500'''

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

""" TODO: Replace the placeholder image block with a call to API for Stable Diffusion, passing image_prompt. """
