from flask import Flask
from flask import render_template
from flask import request
import pandas as pd
import pickle

KEY = 'EEADPN3Z20W94ARM'
app = Flask(__name__)


# def get_data(function='TIME_SERIES_INTRADAY', symbol='IBM', interval='5min'):
#     url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&apikey={KEY}'
#     r = requests.get(url)
#     d = r.json()
#     metadata = d['Meta Data']
#     data = d[f"Time Series ({metadata['4. Interval']})"]
#     return metadata, data

def create_predictor(start, end, company='IBM', predictor='arma'):
    loaded_model = pickle.load(open(f'{predictor}.pkl', 'rb'))
    result=loaded_model.predict(start = start, end=end)
    return result



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        items = request.form.to_dict()
        print(items)
        prediction = create_predictor(start=items['start'],
                                      end=items['end'],
                                      company=items['company'],
                                      predictor=items['predictor'])

        return render_template('result.html', prediction=prediction)


app.run()