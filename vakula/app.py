from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('salary_prediction_model.pkl')

@app.route('/')
def home():
    return 'Salary Prediction Model Deployment with Flask'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Assuming 'work_year', 'job_category', 'experience_level' are present in the JSON data
    features = [
        data['work_year'],
        data['job_category'],
        data['experience_level']
    ]

    # Convert categorical variables to dummy/indicator variables
    features = pd.get_dummies([features], columns=['work_year', 'job_category', 'experience_level'], drop_first=True)

    # Make a prediction using the loaded model
    prediction = model.predict(features)

    # Return the prediction as JSON
    return jsonify({'predicted_salary': prediction[0]})

if __name__ == '__main__':
    app.run(port=5000)
#Use a tool like curl or Postman to send POST requests to http://localhost:5000/predict with JSON data containing the required features ('work_year', 'job_category', 'experience_level').