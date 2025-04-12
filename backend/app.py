from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS # CORS for handling Cross-Origin Resource Sharing
import pickle 
import os

# Create a Flask application instance
app = Flask(__name__)

# Enable CORS for all routes, allowing requests from any origin
CORS(app,resources={r"/*":{"origins":"*"}})



model_path = os.path.join(os.path.dirname(__file__), 'RandomForest.pkl')
model = pickle.load(open(model_path, 'rb'))


# Define a route for handling HTTP GET requests to the root URL
@app.route('/', methods=['GET'])
def get_data():
    data = {
        "message":"API is Running"
    }
    return jsonify(data)
  
# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()   
        print(data)  
        rdata=[data['gender'],data['age'],data['hypertension'],data['heartDisease'],data['everMarried'],data['workType'],data['residenceType'],data['avgGlucoseLevel'],data['bmi'],data['smokingStatus']]
        query_df = pd.DataFrame([rdata]) 
        # [[data['a'],51.0,0,0,1,2,0,166.29,25.600000,1]]
        prediction = model.predict(query_df) 
        print(prediction[0])    
        # {'Prediction': list(prediction)}
        return jsonify(int(prediction[0]))  
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
