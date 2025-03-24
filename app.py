from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
from openai import OpenAI
import base64
import json
from pydantic import BaseModel

endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/"
model_name = "gemini-1.5-flash"

# Initialize OpenAI client
client = None

class NutritionInfo(BaseModel):
    food_name: str
    food_uses: str
    recommended_age_group: str
    pros: str
    cons: str
    serving_size: str
    calories: str
    added_sugars: str
    biotin: str
    calcium: str
    chloride: str
    choline: str
    cholesterol: str
    chromium: str
    copper: str 
    dietary_fiber: str
    fat: str
    folate_folic_acid: str 
    iodine: str
    iron: str 
    magnesium: str 
    manganese: str
    molybdenum: str 
    niacin: str
    pantothenic_acid: str 
    phosphorus: str
    potassium: str
    protein: str 
    riboflavin: str
    saturated_fat: str 
    selenium: str 
    sodium: str 
    thiamin: str
    total_carbohydrate: str 
    vitamin_A: str
    vitamin_B6: str
    vitamin_B12: str
    vitamin_C: str
    vitamin_D: str 
    vitamin_E: str 
    vitamin_K: str 
    zinc: str
    

def get_image_data_url(image_file: str, image_format: str) -> str:
    """
    Helper function to convert an image file to a data URL string.

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
    prompt = """Extract text from this nutrition facts label using OCR. If the food name is missing, predict it based on the nutritional values. Give the food uses, its recommended age group, its pros and its cons. Make sure to use the correct unit value. Replace any missing values with 0g or 0mcg or 0mg."""

    # Generate response
    response = client.beta.chat.completions.parse(
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
        response_format=NutritionInfo,
        model=model_name
    )
    return response.choices[0].message.parsed

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

@app.route('/', methods=['GET', 'POST'])
def welcome():
    global client
    if request.method == 'POST':
        key = request.form.get('key')
        if key is None or not key.startswith('AI') or len(key) != 39:
            return render_template('welcome.html', error=True)
        else:
            client = OpenAI(
                base_url=endpoint,
                api_key=key,
            )
            return redirect(url_for('upload'))
    return render_template('welcome.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part!')
            return redirect(request.url)

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            flash('No selected file!')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Redirect to result page with the filename
            return redirect(url_for('result', filename=filename))
    return render_template('upload.html')

@app.route('/result/<filename>')
def result(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result = extract(image_path, 'jpg')
    result_json = json.dumps(result, default=lambda o:o.__dict__)
    result = json.loads(result_json)
    return render_template('result.html', filename=filename, result=result)

if __name__ == '__main__':
    app.run(debug=True)