
# from import PipeCustomOrdinalEncoder, transform_dataset, CustomKNNImputer, return_df
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_restful import Api, Resource

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np

import json

app = Flask(__name__)
api = Api(app)


class PipeCustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    """Convert categorical features (with missing values in it) into ordinal for KNN Imputation

    Args:
        BaseEstimator ([type]): [description]
        TransformerMixin ([type]): [description]
    """

    def __init__(self):
        self.ord_encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=1001)
#         self.feature = feature_name

    def fit(self, X, y=None):
        na_indices = X[X.isna()].index.values
        not_na_rows = X[X.notna()]
        not_na_np = not_na_rows.to_numpy().reshape(-1, 1)
        self.ord_encoder = self.ord_encoder.fit(not_na_np)
        return self

    def transform(self, X, y=None):
        na_indices = X[X.isna()].index.values
        not_na_rows = X[X.notna()]
        not_na_np = not_na_rows.to_numpy().reshape(-1, 1)
        transformed_data = self.ord_encoder.transform(not_na_np)
        not_na_encode = pd.Series(
            transformed_data.flatten(), index=not_na_rows.index.values)
        return pd.DataFrame(pd.concat([not_na_encode, X[X.isna()]]).sort_index())

    def inverse_transform(self, X, y=None):
        col = X.to_numpy().reshape(-1, 1)
        return self.ord_encoder.inverse_transform(col)


class CustomKNNImputer():
    """Impute missing values using KNN algorithm
    """

    def __init__(self, n_neighbors):
        self.imputer = KNNImputer(n_neighbors=n_neighbors)

    def fit(self, X, y=None):
        self.imputer = self.imputer.fit(X)
        return self.imputer

    def transform(self, X, y=None):
        return pd.DataFrame(self.imputer.transform(X), columns=X.columns)

    def fit_transform(self, X, y=None):
        self.imputer = self.imputer.fit(X)
        return self.transform(X, y)


def return_df(x, cols=None):
    return pd.DataFrame(x, columns=cols)


def transform_dataset(data):
    data.loc[:, "LoanAmount"] = np.log(data.LoanAmount)
    data.loc[:, "TotalApplicantIncome"] = np.log(
        data.ApplicantIncome+data.CoapplicantIncome)
    data.loc[:, "ApplicantIncome"] = np.log(data.ApplicantIncome)
    data = data.drop(columns=['CoapplicantIncome',
                     'Loan_ID', 'Loan_Amount_Term'])
    return data


model = joblib.load(r'./models/1_base_model.sav')


class classify_loan_applications(Resource):
    def get(self):
        return "Welcome! Let's see if you have a good chance in getting your loan approved"

    def post(self):
        json_data = request.get_json()
        data = pd.DataFrame(json_data)
        prediction = model.predict(data)
        pred = {}
        pred['predictions'] = prediction.tolist()
        return jsonify(pred)


api.add_resource(classify_loan_applications, '/')

if __name__ == "__main__":
    app.run(port=5000, debug=True)
