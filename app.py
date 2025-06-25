from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

print("🚀 Starting DeepFake Face Detection System...")
print("📡 Loading detection model...")

# Load the trained model
model = load_model('best_model_fast.keras')
print("✅ Model loaded successfully!")

# Prediction function
def predict_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(256, 256))
        img_array = image.img_to_array(img) / 255.0  # Same rescale as training
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0][0]
        print(f"Raw prediction value: {prediction}")  # Debug output
        
        # Convert to percentage and determine classification
        # Higher values mean Real, lower values mean Fake (based on model's output)
        real_confidence = float(prediction) * 100
        fake_confidence = (1 - float(prediction)) * 100
        
        # Use a more conservative threshold to reduce false positives
        if prediction > 0.7:
            result = 'Real'
            confidence = real_confidence
        elif prediction < 0.3:
            result = 'Fake'
            confidence = fake_confidence
        else:
            result = 'Uncertain'
            confidence = max(real_confidence, fake_confidence)
        
        print(f"🔍 Analysis Result: {result} (Confidence: {confidence:.1f}%)")
        
        return {
            'prediction': result,
            'confidence': round(confidence, 1),
            'real_confidence': round(real_confidence, 1),
            'fake_confidence': round(fake_confidence, 1),
            'raw_value': round(float(prediction), 4)
        }
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return {
            'prediction': 'Error',
            'confidence': 0,
            'real_confidence': 0,
            'fake_confidence': 0,
            'raw_value': 0,
            'error': str(e)
        }

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_url = None

    if request.method == 'POST':
        print("📸 New image upload detected...")
        file = request.files['image']
        if file and file.filename:
            filename = file.filename
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            print(f"💾 Image saved: {filename}")
            print("🧠 Running detection analysis...")
            prediction = predict_image(path)
            img_url = path
            print("✅ Analysis complete!")

    return render_template('index.html', prediction=prediction, img_url=img_url)

if __name__ == '__main__':
    print("🌐 Starting DeepFake Face Detection Web Server...")
    print("🔗 Access the application at: http://127.0.0.1:5000")
    print("🛡️  Ready to detect manipulated facial images!")
    app.run(debug=True, host='127.0.0.1', port=5000)
