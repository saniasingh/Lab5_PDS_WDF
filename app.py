from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extracting input values from the form
        present_price = float(request.form['Present_Price'])
        kms_driven = float(request.form['Kms_Driven'])
        fuel_type = int(request.form['Fuel_Type'])
        seller_type = int(request.form['Seller_Type'])
        transmission = int(request.form['Transmission'])
        owner = int(request.form['Owner'])
        age_of_the_car = int(request.form['Age_of_the_car'])

        # Creating a numpy array of the input features
        features = np.array([[present_price, kms_driven, fuel_type, seller_type, transmission, owner, age_of_the_car]])

        # Make the prediction
        prediction = model.predict(features)

        return render_template('index.html', prediction_text=f"Predicted Selling Price: {prediction[0]:.2f} Lakhs")

if __name__ == "__main__":
    app.run(debug=True)
