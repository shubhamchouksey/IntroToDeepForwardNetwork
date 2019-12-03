#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:32:27 2019

@author: ins
"""

import numpy as np
from flask import Flask, request, render_template
import SimpleDeepForwardNetwork
from SimpleDeepForwardNetwork import SimpleForwardNetwork
import config


app = Flask(__name__)
#with open('model.pkl', 'rb') as f:
#    model = pickle.load(f)
#print(model)
#print(config.w1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_feature = [int(x) for x in request.form.values()]
    final_features = np.array(int_feature)
    final_features = np.insert(1,1,final_features)
    final_features = np.reshape(final_features,(-1,3))
    print(final_features)
    print(config.w1,config.w2)
    obj = SimpleForwardNetwork(config.w1,config.w2)
    prediction = obj.forward(final_features,True)
    
    print(prediction)
    thresh = prediction
    thresh[(thresh > 1e-05)] = 1
    thresh[(thresh > 1e-10) & (thresh < 1e-05)] = 0
    thresh[(thresh < 1) & (thresh>0)] = -1
    
    print(thresh)
    #print(thresh)
    return render_template('index.html', prediction_text = ' {}'.format(int(thresh.item())))

if __name__ == "__main__":
    SimpleDeepForwardNetwork.main()
    app.run(debug=True)


    
    