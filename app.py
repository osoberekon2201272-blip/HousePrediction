from flask import Flask, render_template, request, jsonify
from model import HousePriceModel
import os

app = Flask(__name__)

# Initialize the model
price_model = HousePriceModel()

# Load the trained model (or train if not exists)
if not price_model.load_model():
    print("No trained model found. Training new model...")
    from model import train_and_save_model

    train_and_save_model()
    price_model.load_model()


@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract features
        square_feet = float(data['square_feet'])
        bedrooms = int(data['bedrooms'])
        bathrooms = float(data['bathrooms'])
        age_years = int(data['age_years'])
        garage_spaces = int(data['garage_spaces'])
        location_score = int(data['location_score'])

        # Validate inputs
        if square_feet < 500 or square_feet > 10000:
            return jsonify({
                'success': False,
                'error': 'Square footage must be between 500 and 10,000'
            })

        if bedrooms < 1 or bedrooms > 10:
            return jsonify({
                'success': False,
                'error': 'Bedrooms must be between 1 and 10'
            })

        if bathrooms < 1 or bathrooms > 8:
            return jsonify({
                'success': False,
                'error': 'Bathrooms must be between 1 and 8'
            })

        if age_years < 0 or age_years > 100:
            return jsonify({
                'success': False,
                'error': 'Age must be between 0 and 100 years'
            })

        if garage_spaces < 0 or garage_spaces > 4:
            return jsonify({
                'success': False,
                'error': 'Garage spaces must be between 0 and 4'
            })

        if location_score < 1 or location_score > 10:
            return jsonify({
                'success': False,
                'error': 'Location score must be between 1 and 10'
            })

        # Make prediction
        predicted_price = price_model.predict(
            square_feet=square_feet,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            age_years=age_years,
            garage_spaces=garage_spaces,
            location_score=location_score
        )

        # Return result
        return jsonify({
            'success': True,
            'predicted_price': round(predicted_price, 2),
            'inputs': {
                'square_feet': square_feet,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'age_years': age_years,
                'garage_spaces': garage_spaces,
                'location_score': location_score
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    # For local development
    app.run(debug=True, host='0.0.0.0', port=5000)