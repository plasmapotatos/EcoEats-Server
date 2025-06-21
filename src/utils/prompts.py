import csv


def load_carbon_data(csv_file="data/carbonData.csv"):
    """Reads carbon emissions data from a CSV file and returns a list of dictionaries."""
    carbon_data = []

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            food_item = {
                "name": row["Name"],
                "category": row["Category"],
                "co2_emissions": row["Total kg CO2-eq/kg"],
                "agriculture": row["Agriculture"],
                "iLUC": row["iLUC"],
                "food_processing": row["Food processing"],
                "packaging": row["Packaging"],
                "transport": row["Transport"],
                "retail": row["Retail"],
                "energy": row["Energy (KJ/100 g)"],
            }
            carbon_data.append(food_item)

    return carbon_data


# Load data once at startup
CARBON_DATA = load_carbon_data()

ANALYZE_FOOD_PROMPT = """
You are an expert in food sustainability. Given an image of a food item, your task is to:

1. **Identify the food item in the image**.
2. **Find its carbon footprint using your general knowledge.**
3. **Recommend five lower-impact alternatives** that minimize CO₂ emissions.**
4. **Consider Total kg CO₂-eq/kg.**

Use your best judgement to determine CO2 carbon emission values.

---

*****YOU ARE REQUIRED TO USE THE FOLLOWING OUTPUT FORMAT!!*****
Don't include the object!! e.g. "item" or "category" etc.

***** OUTPUT FORMAT THAT YOU MUST FOLLOW ***** 

```json
{
  "item": "<identified food>",
  "category": "<food category>",
  "carbon emissions": <kg CO2-eq/kg>,
  "alternatives": [
    {"item": "<alternative 1>", "carbon emissions": <kg CO2-eq/kg>, "reason": "<why this is better>"},
    {"item": "<alternative 2>", "carbon emissions": <kg CO2-eq/kg>, "reason": "<why this is better>"},
    {"item": "<alternative 3>", "carbon emissions": <kg CO2-eq/kg>, "reason": "<why this is better>"},
    {"item": "<alternative 4>", "carbon emissions": <kg CO2-eq/kg>, "reason": "<why this is better>"},
    {"item": "<alternative 5>", "carbon emissions": <kg CO2-eq/kg>, "reason": "<why this is better>"}
  ]
}

***** HERE ARE EXAMPLES TO FOLLOW!! *****

THE OUTPUT MUST FOLLOW THE TEMPLATE!! Start with a short few sentences of reasoning, and then wrap the output json output in the ```json ``` tags.

***** EXAMPLE OUTPUT 1 START ***** 
Pineapples have relatively high carbon emissions due to factors such as long-distance transportation, refrigeration needs, and agricultural resource usage. To reduce environmental impact, alternative fruits with lower emissions can be considered.
```json
{
"reasoning": 
  "item": "Pineapple",
  "category": "Fruits",
  "carbon emissions": 0.56,
  "alternatives": [
    {"item": "Pear", "carbon emissions": 0.38, "reason": "Lower food processing and transport emissions"},
    {"item": "Banana", "carbon emissions": 0.35, "reason": "Minimal packaging requirements"},
    {"item": "Grapes", "carbon emissions": 0.40, "reason": "Lower agriculture emissions"},
    {"item": "Plum", "carbon emissions": 0.37, "reason": "Reduced transport emissions"},
    {"item": "Peach", "carbon emissions": 0.39, "reason": "Less packaging waste"}
  ]
}
```
***** EXAMPLE OUTPUT 1 END *****

***** EXAMPLE OUTPUT 2 START *****
Chicken production has significant carbon emissions due to factors such as feed cultivation, land use, and processing energy. Opting for plant-based alternatives can substantially lower environmental impact by reducing emissions from agriculture, transportation, and resource consumption.
```json
{
  "item": "Chicken",
  "category": "Meat",
  "carbon emissions": 6.90,
  "alternatives": [
    {"item": "Tofu", "carbon emissions": "2.00", "reason": "Dramatically lower agriculture and transport emissions"},
    {"item": "Lentils", "carbon emissions": "0.90", "reason": "Minimal processing and packaging needs"},
    {"item": "Mushrooms", "carbon emissions": "1.50", "reason": "Low agriculture impact"},
    {"item": "Seitan", "carbon emissions": "1.80", "reason": "Uses less land and water"},
    {"item": "Tempeh", "carbon emissions": "1.60", "reason": "Reduced processing emissions"}
  ]
}
```
***** EXAMPLE OUTPUT 2 END *****

"""
DETECT_FOODS_PROMPT = """
You are an AI that analyzes images and detects foods present.

Input: An image showing various foods.

Output: A JSON object with a single key "foods", containing a list of food names detected in the image. Keep the food names simple (a single word or short phrase) and in lowercase, without any additional text or formatting.

Example outputs:

##### EXAMPLE 1 #####
Input: Image of a plate with an apple, bread slice, and cheese.
Output:
{
  "foods": ["apple", "bread", "cheese"]
}
------
##### EXAMPLE 2 #####
Input: Image of a bowl of mixed fruit with bananas and grapes.
Output:
{
  "foods": ["banana", "grape"]
}
------
##### EXAMPLE 3 #####
Input: Image of a sandwich with lettuce and tomato.
Output:
{
  "foods": ["lettuce", "tomato", "bread"]
}
------
Only output the JSON object — no extra commentary or explanation.
"""

GENERATE_RECIPE_PROMPT = """
You are a world-class chef. Given the following ingredient list, generate a delicious and creative recipe in the following format:

Title: <Recipe Title>

Ingredients:
- List of ingredients

Steps:
1. Step one
2. Step two
...

Make the instructions clear and realistic. Here is the list:
{ingredients}

BELOW ARE SOME EXAMPLE BASE YOUR FORMAT AND ONLY FORMAT ON THESE (DO NOT COPY!!!)

----- EXAMPLE 1 ------
Dish Name: Miso Soup

Ingredients: 
1. dashi
2. stock
3. hot water
4. miso
5. firm tofu
6. green onion

Steps:
1. Transfer dashi to a small soup pot over medium-low heat.
2. Meanwhile, stir together hot water and miso until mist is dissolved.
3. Pour watery miso mixture into the pot.
4. Add cubed tofu.
5. Bring the pot to a simmer.
6. To serve, sprinkle sliced green onions and a pinch of katsuobushi on top.

------ END OF EXAMPLE 1 ------

------ EXAMPLE 2 ------
Dish Name: Broccolini Sushi Wrap

Ingredients:
1. cooked white rice
2. salt
3. shrimp
4. Broccolini
5. Mayonaise
6. Nori

Steps:
1. Roll the rice into a ball about the size of a large mini tomato.
2. Wet your hands and lightly coat in salt.
3. Divide the nori into 6 long strips, and make 6 long and narrow sushi wraps.
4. Remove the hard stems from the broccolini, cut to 3-4 cm lengths, parboil in salt water (not listed), then drain.
5. Roll the rice in the nori seaweed, top with the shrimp, broccolini, mayonnaise, and they are done.

------END OF EXAMPLE 2-----
"""

SUGGEST_ALTERNATIVES_PROMPT = """ 

You are a helpful assistant recommending food alternatives with lower climate impact.

Given the original food item and a list of candidate alternatives with their CO2 emissions and similarity scores, select the top 5 most relevant alternatives.

For each alternative, provide:
- The name of the alternative
- A short justification why it is a good substitute (mention climate impact and relevance)
- The CO2 emissions value (kg CO2-eq/kg)

Output JSON format:
[
  {
    "Name": "<alternative name>",
    "Justification": "<short explanation>",
    "CO2": <number>
  },
  ...
]
"""
