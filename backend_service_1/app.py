from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import pickle
import os
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the trained ML model
model_path = os.path.join(os.path.dirname(__file__), 'RandomForest.pkl')
model = pickle.load(open(model_path, 'rb'))

# Connect to MongoDB (localhost by default)
client = MongoClient(mongo_uri)  # Replace with your URI if hosted
db = client['strokepredictor']
collection = db['predictions']

# Health check route
@app.route('/', methods=['GET'])
def get_data():
    return jsonify({"message": "API is Running"})

# Prediction + DB insert route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # print("Received data:", data)
        user_id = data.get("userId")
        print("User ID:", user_id)
        # Prepare data for prediction
        rdata = [
            data['gender'], data['age'], data['hypertension'],
            data['heartDisease'], data['everMarried'], data['workType'],
            data['residenceType'], data['avgGlucoseLevel'],
            data['bmi'], data['smokingStatus']
        ]
        query_df = pd.DataFrame([rdata])
        prediction = model.predict(query_df)
        pred_value = int(prediction[0])

        # Store input + prediction in MongoDB
        data_to_store = data.copy()
        data_to_store.pop('userId')  # Remove userId before inserting prediction doc
        data_to_store['prediction'] = pred_value
        pred_result = db.predictions.insert_one(data_to_store)
        prediction_id = pred_result.inserted_id

        print("Inserted with ID:", prediction_id)
        
        # Add prediction reference to user's reports array
        db.users.update_one(
            {'_id': ObjectId(user_id)},
            {'$push': {'reports': prediction_id}}
        )
        print(f"Prediction saved and linked to user {user_id}")
        return jsonify(pred_value)

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

