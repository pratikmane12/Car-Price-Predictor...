import json
import pickle

import pandas as pd
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin

  app = Flask(__name__, template_folder="../templates")
cors = CORS(app)

model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')


@app.route('/', methods=['GET'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(str(m) for m in car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    return render_template(
        'index.html',
        companies=companies,
        car_models=car_models,
        car_models_json=json.dumps(car_models),  # used by the JS to filter models
        years=years,
        fuel_types=fuel_types,
    )


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    if not company or not car_model:
        return 'Please select a company and a model', 400

    try:
        year = int(year)
        driven = int(driven)
    except (TypeError, ValueError):
        return 'Year and kilometres must be numbers', 400

    input_df = pd.DataFrame(
        [[car_model, company, year, driven, fuel_type]],
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
    )

    prediction = model.predict(input_df)
    print(prediction)

    return str(round(float(prediction[0]), 2))


if __name__ == '__main__':
    app.run(debug=True)
