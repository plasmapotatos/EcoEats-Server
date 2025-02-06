import csv
import json

def load_carbon_data(csv_file="carbonData.csv"):
    """Reads carbon emissions data from a CSV file and returns a list of dictionaries."""
    carbon_data = []

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            food_item = {
                "name": row["Name"],
                "category": row["Category"],
                "co2_emissions": float(row["Total kg CO2-eq/kg"]),
                "agriculture": float(row["Agriculture"]),
                "iLUC": float(row["iLUC"]),
                "food_processing": float(row["Food processing"]),
                "packaging": float(row["Packaging"]),
                "transport": float(row["Transport"]),
                "retail": float(row["Retail"]),
                "energy": float(row["Energy (KJ/100 g)"]),
            }
            carbon_data.append(food_item)

    return carbon_data

# Load data once at startup
CARBON_DATA = load_carbon_data()

ANALYZE_FOOD_PROMPT = f"""
You are an expert in food sustainability. Given an image of a food item, your task is to:

1. **Identify the food item in the image**.
2. **Find its carbon footprint using the dataset below.**
3. **Recommend five lower-impact alternatives** that minimize CO₂ emissions, food processing, transport, and packaging impact.
4. **Consider all environmental factors**, including:
   - Total kg CO₂-eq/kg
   - Agriculture emissions
   - iLUC (Indirect Land Use Change)
   - Food processing impact
   - Packaging emissions
   - Transport emissions
   - Retail emissions
   - Energy content

Here is the **carbon emissions dataset** for reference:

{json.dumps(CARBON_DATA, indent=2)}

---

*****YOU ARE REQUIRED TO USE THE FOLLOWING OUTPUT FORMAT!!*****

```json
{{
  "item": "<identified food>",
  "category": "<food category>",
  "carbon emissions": <kg CO2-eq/kg>,
  "agriculture": <kg CO2-eq/kg>,
  "iLUC": <kg CO2-eq/kg>,
  "food processing": <kg CO2-eq/kg>,
  "packaging": <kg CO2-eq/kg>,
  "transport": <kg CO2-eq/kg>,
  "retail": <kg CO2-eq/kg>,
  "energy": <Energy (KJ/100 g)>,
  "alternatives": [
    {{"item": "<alternative 1>", "carbon emissions": <kg CO2-eq/kg>, "reason": "<why this is better>"}},
    {{"item": "<alternative 2>", "carbon emissions": <kg CO2-eq/kg>, "reason": "<why this is better>"}},
    {{"item": "<alternative 3>", "carbon emissions": <kg CO2-eq/kg>, "reason": "<why this is better>"}},
    {{"item": "<alternative 4>", "carbon emissions": <kg CO2-eq/kg>, "reason": "<why this is better>"}},
    {{"item": "<alternative 5>", "carbon emissions": <kg CO2-eq/kg>, "reason": "<why this is better>"}}
  ]
}}

***** HERE ARE EXAMPLES TO FOLLOW!! *****
***** EXAMPLE 1 START ***** 
{
  "item": "Apple",
  "category": "Fruits",
  "carbon emissions": 0.42,
  "agriculture": 0.20,
  "iLUC": 0.01,
  "food processing": 0.02,
  "packaging": 0.05,
  "transport": 0.10,
  "retail": 0.04,
  "energy": 52,
  "alternatives": [
    {"item": "Pear", "carbon emissions": 0.38, "reason": "Lower food processing and transport emissions"},
    {"item": "Banana", "carbon emissions": 0.35, "reason": "Minimal packaging requirements"},
    {"item": "Grapes", "carbon emissions": 0.40, "reason": "Lower agriculture emissions"},
    {"item": "Plum", "carbon emissions": 0.37, "reason": "Reduced transport emissions"},
    {"item": "Peach", "carbon emissions": 0.39, "reason": "Less packaging waste"}
  ]
}
***** EXAMPLE 1 END *****

***** EXAMPLE 2 START *****
{
  "item": "Chicken",
  "category": "Meat",
  "carbon emissions": 6.90,
  "agriculture": 5.00,
  "iLUC": 0.50,
  "food processing": 0.30,
  "packaging": 0.20,
  "transport": 0.40,
  "retail": 0.50,
  "energy": 143,
  "alternatives": [
    {"item": "Tofu", "carbon emissions": 2.00, "reason": "Dramatically lower agriculture and transport emissions"},
    {"item": "Lentils", "carbon emissions": 0.90, "reason": "Minimal processing and packaging needs"},
    {"item": "Mushrooms", "carbon emissions": 1.50, "reason": "Low agriculture impact"},
    {"item": "Seitan", "carbon emissions": 1.80, "reason": "Uses less land and water"},
    {"item": "Tempeh", "carbon emissions": 1.60, "reason": "Reduced processing emissions"}
  ]
}"""