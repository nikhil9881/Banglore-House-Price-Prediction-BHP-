from flask import Flask,request,jsonify, render_template
import pickle
import numpy as np
import pandas as pd
app=Flask(__name__)
data=pd.read_csv("D:\Data Science Class\project\Banglore House Price Prediction(BHP)\Cleaned_data.csv")
pipe=pickle.load(open("D:\Data Science Class\project\Banglore House Price Prediction(BHP)\RidgeModel.pkl",'rb'))


@app.route('/')
def index():

    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = int(request.form.get('bhk'))
    bath = int(request.form.get('bath'))
    sqft = float(request.form.get('total_sqft'))


    input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input_data)[0]


    return str(np.round(prediction,2))+"lacks"

if __name__== "__main__":
    print("starting python flask server")
    app.run(debug=True,port=5001)

