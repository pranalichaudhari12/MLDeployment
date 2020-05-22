#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math

app = Flask(__name__)
model = pickle.load(open('Log.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features  = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    prediction = str(prediction)[2:-2]
    return render_template('index.html',prediction_text="The Species is {}".format(prediction))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

