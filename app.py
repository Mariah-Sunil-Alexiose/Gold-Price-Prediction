from flask import Flask, render_template, request
import joblib
from logging import debug
import numpy as np

app = Flask(__name__)

def preprocess(spx, uso, slv, eur_usd):
    test_data = np.array([[spx, uso, slv, eur_usd]])
    trained_model = joblib.load('gold_price')
    prediction = trained_model.predict(test_data)
    return prediction

@app.route('/',methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        spx = request.form.get('spx')
        uso = request.form.get('uso')
        slv = request.form.get('slv')
        eur_usd = request.form.get('eur_usd')
        prediction = preprocess(spx, uso, slv, eur_usd)
        return render_template('prediction.html', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)