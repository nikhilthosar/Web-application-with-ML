from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib as mp
import sklearn as sl
import pickle

app = Flask(__name__)
with open('lrmodel.pkl','rb') as pk:
   model = pickle.load(pk)

@app.route('/')
def my_app():    
       return render_template("index.html")

# ==================================login page==========================================
@app.route('/login')
def login():  
       return 'Login'

@app.route('/logout')
def logout():  
       return render_template("index.html")

@app.route('/sign-up')
def signup():  
       return render_template("index.html")
#=========================================================================================
@app.route('/project')
def project():     
       return render_template("project.html")

@app.route('/predict', methods = ['POST'])
def predict():
       int_features = [float(i) for i in request.form.values()]
       features  = [np.array(int_features)]
       prediction = model.predict(features)
       return render_template("project.html",prediction_text = 'Quantitative measure of diabetes progression one year after baseline is {}'.format(prediction))
    
        
        

if __name__ == '__main__':
    app.debug=True
    app.run(host='0.0.0.0',port=8000)
