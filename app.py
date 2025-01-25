from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv  
from openai import OpenAI
import base64

import json

endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o"

# Initialize OpenAI client
client = None

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

def extract(image_path, img_format):
    prompt = """Extract text from this nutrition facts label using OCR. If the food name is missing, predict it based on the nutritional values. Make sure to use the correct unit value. Replace any missing values with 0g.

    Output the result as a raw JSON string as follows:
    food name: [food name], serving size: [value], calories: [value], added sugars: [value], biotin: [value], calcium: [value], chloride: [value], choline: [value], cholesterol: [value], chromium: [value], copper: [value], dietary fiber: [value], fat: [value], folate/folic acid: [value], iodine: [value], iron: [value], magnesium: [value], manganese: [value], molybdenum: [value], niacin: [value], pantothenic acid: [value], phosphorus: [value], potassium: [value], protein: [value], riboflavin: [value], saturated fat: [value], selenium: [value], sodium: [value], thiamin: [value], total carbohydrate: [value], vitamin A: [value], vitamin B6: [value], vitamin B12: [value], vitamin C: [value], vitamin D: [value], vitamin E: [value], vitamin K: [value], zinc: [value]
    
    Don't use ```json ``` or any other code block formatting. Just the raw JSON string."""

    # Generate response
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that describes images and maintains conversation context.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": get_image_data_url(image_path, img_format),
                                "detail": "low"
                            },
                        }
                ],
            },
        ],
        model=model_name
    )
    return response.choices[0].message.content


app = Flask(__name__)

# Load environment variables
load_dotenv()

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')


# Configure the upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST', 'GET'])
def hello_world():
    global client

    if request.method == 'POST':
        key = request.form.get('key')

        if not key:
            flash('key is required!')
        else:
            API_KEY = key
            client = OpenAI(
                base_url=endpoint,
                api_key=API_KEY,
            )
            return render_template('index.html')
        # Check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Process the uploaded image here
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            result = extract(image_path, 'jpg')
            result = json.loads(result)
            return render_template('index.html', filename=filename, result=result)
    return render_template('index.html')

def process_image(filepath):
    # Add your image processing code here
    print(f'Processing image: {filepath}')

if __name__ == '__main__':
    app.run(debug=True)