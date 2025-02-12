import csv
import json

def load_carbon_data(csv_file="src/utils/carbonData.csv"):
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

ANALYZE_FOOD_PROMPT = f"""
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
{{
  "item": "<identified food>",
  "category": "<food category>",
  "carbon emissions": <kg CO2-eq/kg>,
  "alternatives": [
    {{"item": "<alternative 1>", "carbon emissions": <kg CO2-eq/kg>, "reason": "<why this is better>"}},
    {{"item": "<alternative 2>", "carbon emissions": <kg CO2-eq/kg>, "reason": "<why this is better>"}},
    {{"item": "<alternative 3>", "carbon emissions": <kg CO2-eq/kg>, "reason": "<why this is better>"}},
    {{"item": "<alternative 4>", "carbon emissions": <kg CO2-eq/kg>, "reason": "<why this is better>"}},
    {{"item": "<alternative 5>", "carbon emissions": <kg CO2-eq/kg>, "reason": "<why this is better>"}}
  ]
}}

***** HERE ARE EXAMPLES TO FOLLOW!! *****

THE OUTPUT MUST FOLLOW THE TEMPLATE!! Start with a short few sentences of reasoning, and then wrap the output json output in the ```json ``` tags.

***** EXAMPLE OUTPUT 1 START ***** 
Pineapples have relatively high carbon emissions due to factors such as long-distance transportation, refrigeration needs, and agricultural resource usage. To reduce environmental impact, alternative fruits with lower emissions can be considered.
```json
{{
"reasoning": 
  "item": "Pineapple",
  "category": "Fruits",
  "carbon emissions": 0.56,
  "alternatives": [
    {{"item": "Pear", "carbon emissions": 0.38, "reason": "Lower food processing and transport emissions"}},
    {{"item": "Banana", "carbon emissions": 0.35, "reason": "Minimal packaging requirements"}},
    {{"item": "Grapes", "carbon emissions": 0.40, "reason": "Lower agriculture emissions"}},
    {{"item": "Plum", "carbon emissions": 0.37, "reason": "Reduced transport emissions"}},
    {{"item": "Peach", "carbon emissions": 0.39, "reason": "Less packaging waste"}}
  ]
}}
```
***** EXAMPLE OUTPUT 1 END *****

***** EXAMPLE OUTPUT 2 START *****
Chicken production has significant carbon emissions due to factors such as feed cultivation, land use, and processing energy. Opting for plant-based alternatives can substantially lower environmental impact by reducing emissions from agriculture, transportation, and resource consumption.
```json
{{
  "item": "Chicken",
  "category": "Meat",
  "carbon emissions": 6.90,
  "alternatives": [
    {{"item": "Tofu", "carbon emissions": "2.00", "reason": "Dramatically lower agriculture and transport emissions"}},
    {{"item": "Lentils", "carbon emissions": "0.90", "reason": "Minimal processing and packaging needs"}},
    {{"item": "Mushrooms", "carbon emissions": "1.50", "reason": "Low agriculture impact"}},
    {{"item": "Seitan", "carbon emissions": "1.80", "reason": "Uses less land and water"}},
    {{"item": "Tempeh", "carbon emissions": "1.60", "reason": "Reduced processing emissions"}}
  ]
}}
```
***** EXAMPLE OUTPUT 2 END *****

"""