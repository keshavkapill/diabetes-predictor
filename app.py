import sys
import sklearn.ensemble
import sklearn.tree

# Redirect old internal scikit-learn paths to the new structure
sys.modules['sklearn.ensemble.forest'] = sklearn.ensemble
sys.modules['sklearn.tree.tree'] = sklearn.tree

from flask import Flask, render_template, request
import pickle
import numpy as np
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(BASE_DIR, 'diabetes-prediction-rfc-model.pkl')
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = int(classifier.predict(data)[0])  # extract scalar from numpy array
        
        if my_prediction == 1:
            result_text = "⚠️ Based on the inputs, the person is likely DIABETIC."
        else:
            result_text = "✅ Based on the inputs, the person is likely NOT DIABETIC."
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)