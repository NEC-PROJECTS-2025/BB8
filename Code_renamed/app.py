import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
scaler = StandardScaler()

app = Flask(__name__)

# Load the trained model
model=pickle.load(open('xgb_model.pkl','rb'))
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/test')
def test():
    return render_template('test.html')  # Render the test.html page
@app.route('/login')
def login():
    return render_template('login.html')  # Render the test.html page
@app.route('/reg')
def reg():
    return render_template('reg.html')  # Render the test.html page


@app.route('/predict', methods=['POST'])
def predict():
    # Collect features from form data
    gender=int(request.form['gender'])
    age=int(request.form['age'])
    hypertension=int(request.form['hypertension'])
    heart_disease=int(request.form['heart_disease'])
    ever_married=int(request.form['ever_married'])
    work_type=int(request.form['work_type'])
    Residence_type=int(request.form['Residence_type'])
    avg_glucose_level=float(request.form['avg_glucose_level'])
    bmi=float(request.form['bmi'])
    smoking_status=int(request.form['smoking_status'])
    

    
    
    # feature_array=[[gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,
    #                  avg_glucose_level,bmi,smoking_status]]
    # feature_array = scaler.transform(feature_array)
    # features_array = np.array(feature_array).reshape(1, -1)
    # print(feature_array)
    
    input_data = pd.DataFrame([[gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,
    avg_glucose_level,bmi,smoking_status	]],
                          columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']) # Replace with your actual column names

# Scale the input data using the previously fitted scaler
    # input_data_scaled = scaler.transform(input_data)

# Convert the scaled data back to a DataFrame with original column names
    input_data = pd.DataFrame(input_data, columns=input_data.columns)

# Now you can use column names for selection
    input_data = input_data[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']] # Replace with your actual column names
    prediction = model.predict(input_data)
    print(prediction)

    if prediction == 1:
        return render_template("stroke.html", prediction="Fake Profile...!")  
    else:
        return render_template("nostroke.html", prediction="Real Profile...!")

    
if __name__ == "__main__":
    app.run(debug=True,port=5050)
