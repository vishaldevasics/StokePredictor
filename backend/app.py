from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import pickle
import os
from pymongo import MongoClient
from bson import ObjectId

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the trained ML model
model_path = os.path.join(os.path.dirname(__file__), 'RandomForest.pkl')
model = pickle.load(open(model_path, 'rb'))

# Connect to MongoDB (localhost by default)
client = MongoClient("mongodb+srv://vishaldevasics:EKgT3eIxB0Vdgrni@cluster0.ie8do6z.mongodb.net/")  # Replace with your URI if hosted
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


# multiline comment in python
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#         print("Received:", data)

#         # Extract userId for reference
#         user_id = data.get("userId")
#         if not user_id:
#             return jsonify({'error': 'userId is required'}), 400

#         # Prepare data for prediction
#         rdata = [
#             data['gender'], data['age'], data['hypertension'],
#             data['heartDisease'], data['everMarried'], data['workType'],
#             data['residenceType'], data['avgGlucoseLevel'],
#             data['bmi'], data['smokingStatus']
#         ]
#         query_df = pd.DataFrame([rdata])
#         prediction = model.predict(query_df)
#         pred_value = int(prediction[0])

#         # Save prediction + userId
#         data_to_store = {
#             "userId": user_id,
#             "gender": data['gender'],
#             "age": data['age'],
#             "hypertension": data['hypertension'],
#             "heartDisease": data['heartDisease'],
#             "everMarried": data['everMarried'],
#             "workType": data['workType'],
#             "residenceType": data['residenceType'],
#             "avgGlucoseLevel": data['avgGlucoseLevel'],
#             "bmi": data['bmi'],
#             "smokingStatus": data['smokingStatus'],
#             "prediction": pred_value
#         }

#         # Insert prediction
#         result = collection.insert_one(data_to_store)
#         print("Inserted prediction with ID:", result.inserted_id)

#         # Update user's `reports` array with the inserted prediction ID
#         db['users'].update_one(
#             {'_id': user_id},
#             {'$push': {'reports': result.inserted_id}}
#         )

#         return jsonify({"prediction": pred_value, "recordId": str(result.inserted_id)})

#     except Exception as e:
#         return jsonify({'error': str(e)})