import os
import base64
from openai import OpenAI

token = ""
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o"

def get_image_data_url(image_file: str, image_format: str) -> str:
    """
    Helper function to converts an image file to a data URL string.

    Args:
        image_file (str): The path to the image file.
        image_format (str): The format of the image file.

    Returns:
        str: The data URL of the image.
    """
    try:
        with open(image_file, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Could not read '{image_file}'.")
        exit()
    return f"data:image/{image_format};base64,{image_data}"


client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that describes images in details.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Extract text from this nutrition facts label using OCR. If the food name is missing, predict it based on the nutritional values. Make sure to use the correct unit value. Replace any missing values with 0g.

                            Output the result as a raw JSON string as follows:
                            food name: [food name], serving size: [value], calories: [value], added sugars: [value], biotin: [value], calcium: [value], chloride: [value], choline: [value], cholesterol: [value], chromium: [value], copper: [value], dietary fiber: [value], fat: [value], folate/folic acid: [value], iodine: [value], iron: [value], magnesium: [value], manganese: [value], molybdenum: [value], niacin: [value], pantothenic acid: [value], phosphorus: [value], potassium: [value], protein: [value], riboflavin: [value], saturated fat: [value], selenium: [value], sodium: [value], thiamin: [value], total carbohydrate: [value], vitamin A: [value], vitamin B6: [value], vitamin B12: [value], vitamin C: [value], vitamin D: [value], vitamin E: [value], vitamin K: [value], zinc: [value]
                            
                            Don't use ```json ``` or any other code block formatting. Just the raw JSON string.""",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": get_image_data_url("static/uploads/unnamed.jpg", "jpg"),
                        "detail": "low"
                    },
                },
            ],
        },
    ],
    model=model_name,
)

print(response.choices[0].message.content)