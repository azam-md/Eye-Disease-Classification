from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
app = Flask(__name__)
model = load_model('eye_disease_model.h5')


# Define the directory where uploaded images will be stored
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Define precautions, solutions, and reasons for each eye disease
disease_info = {
    'Cataract': {
        'precautions': ['Wear sunglasses', 'Protect eyes from UV rays', 'Eat a healthy diet'],
        'solutions': ['Cataract surgery'],
        'reasons': ['Aging', 'Diabetes', 'Exposure to UV radiation']
    },
    'Diabetic Retinopathy': {
        'precautions': ['Control blood sugar levels', 'Regular eye check-ups'],
        'solutions': ['Laser treatment', 'Injection of medication into the eye'],
        'reasons': ['High blood sugar levels', 'Diabetes duration', 'High blood pressure']
    },
    'Glaucoma': {
        'precautions': ['Regular eye check-ups', 'Avoid smoking', 'Protect eyes from injury'],
        'solutions': ['Medication', 'Laser surgery', 'Conventional surgery'],
        'reasons': ['Increased pressure in the eye', 'Family history', 'Age']
    },
    'Normal': {
        'precautions': ['Regular eye check-ups', 'Wear protective eyewear'],
        'solutions': ['Maintain eye health'],
        'reasons': ['N/A']
    }
}

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Resize the image to match the input size expected by your model
    resized_image = image.resize((222, 222))
    # Convert image to numpy array
    img_array = np.array(resized_image)
    # Normalize pixel values
    img_array = img_array / 255.0
    # Expand dimensions to create a batch for the model
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to convert model predictions to class labels
def get_class_label(prediction):
    classes = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
    predicted_class = np.argmax(prediction)
    return classes[predicted_class]

@app.route('/')
def index():
    #return render_template('index3.html')

    return render_template('index2.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Check if a file was uploaded
    if 'image' not in request.files:
       # return render_template('index3.html', error='No file uploaded')
        return render_template('index2.html', error='No file uploaded')

    # Get the uploaded image file
    uploaded_file = request.files.get('image')

    # If the user does not select a file, browser also submit an empty part without filename
    if uploaded_file.filename == '':
#        return render_template('index3.html', error='No selected file')
        return render_template('index2.html', error='No selected file')

    # Read the image file
    img = Image.open(uploaded_file.stream)

    # Preprocess the image
    processed_img = preprocess_image(img)
# Save the uploaded file
    filename = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    uploaded_file.save(filename)

    # Make prediction using the model
    prediction = model.predict(processed_img)

    # Get class label
    result = get_class_label(prediction[0])

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
