import pickle
import pandas as pd
import numpy as np
from flask import Flask,request,jsonify,render_template
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## import the pickle files

ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scalar = pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata", methods = ['GET','POST'])
def predict_data():
    if request.method == 'POST':
        temperature = float(request.form['temperature'])
        rh = float(request.form['rh'])
        ws = float(request.form['ws'])
        rain = float(request.form['rain'])
        ffmc = float(request.form['ffmc'])
        dmc = float(request.form['dmc'])
        isi = float(request.form['isi'])
        classes = float(request.form['classes'])
        region = float(request.form['region'])

        scaled_data = standard_scalar.transform([[temperature,rh,ws,rain,ffmc,dmc,isi,classes,region]])
        result = ridge_model.predict(scaled_data)

        return render_template('home.html', result = result[0])

    else:
        return render_template('home.html')

if __name__== "__main__":
    app.run(host="0.0.0.0")