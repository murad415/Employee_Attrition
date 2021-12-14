import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import datetime as dt

app = Flask(__name__)
model=joblib.load(open("Employee_attrition.joblib", 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

def age(DOB):
    DOB = pd.to_datetime(DOB)
    today = dt.date.today()
    return today.year - DOB.year - ((today.month,today.day) < (DOB.month,DOB.day))

def vintage(joing_date):
    joing_date = pd.to_datetime(joing_date)
    today = dt.datetime.now()
    return int(((today-joing_date)/np.timedelta64(1,"M")))

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    today = dt.date.today()
    int_features = request.form.to_dict()
    df=pd.DataFrame(int_features,index=[0])
    employee=df['Employee_Name'][0]
    df['Age']=df['Employee_DOB'].apply(age)
    df['week']=pd.to_datetime(df["Employee_Joining_Date"]).dt.week
    df['Employee_Vintage']=df['Employee_Joining_Date'].apply(vintage)
    df.drop(['Employee_Name','Employee_DOB','Employee_Joining_Date'],axis=1)
    output=np.round(model.predict_proba(df)[0][1],2)
    return render_template('index.html', prediction_text=f'{employee} will leave the Organization in next 6 month probability is {output}')


if __name__ == "__main__":
    app.run(debug=True)