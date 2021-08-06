import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
import statsmodels.api as sm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OrdinalEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

from sklearn.compose import ColumnTransformer


from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import FunctionTransformer

from sklearn.preprocessing import LabelEncoder

import joblib

import click
import logging

# from helpers import PipeCustomOrdinalEncoder, transform_dataset, CustomKNNImputer, return_df


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np


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


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):

    df = pd.read_csv(input_filepath)

    cat_columns = ['Self_Employed',
                   'Dependents',
                   'Gender',
                   'Married',
                   'Education',
                   'Property_Area',
                   'Credit_History']

    num_columns = ['LoanAmount', 'ApplicantIncome', 'TotalApplicantIncome']

    # pipeline components from here
    transformations = FunctionTransformer(transform_dataset)

    cat_preprocess = ColumnTransformer(transformers=[
        ('employed', PipeCustomOrdinalEncoder(), 'Self_Employed'),
        ('dependents', PipeCustomOrdinalEncoder(), 'Dependents'),
        #     ('loan_amount',CustomOrdinalEncoder(),'Loan_Amount_Term'),
        ('married', PipeCustomOrdinalEncoder(), 'Married'),
        ('gender', PipeCustomOrdinalEncoder(), 'Gender'),
        ('education', PipeCustomOrdinalEncoder(), 'Education'),
        ('property', PipeCustomOrdinalEncoder(), 'Property_Area'),
        ('credit-history', PipeCustomOrdinalEncoder(), 'Credit_History')
    ])

    categories = Pipeline(steps=[
        ('cat', cat_preprocess),
        ('cat_names', FunctionTransformer(
            return_df, kw_args={"cols": cat_columns}))
    ])

    num_preprocess = ColumnTransformer(
        transformers=[
            ('num_scaling', MinMaxScaler(), num_columns)
        ])

    numerical = Pipeline(steps=[
        ('num', num_preprocess),
        ('num_names', FunctionTransformer(
            return_df, kw_args={"cols": num_columns}))
    ])

    impute_preprocess = Pipeline(steps=[
        ('feature_union', FeatureUnion(transformer_list=[
            ('cat_pipe', categories),
            ('num_pipe', numerical)
        ])),
        ('test', FunctionTransformer(return_df,
         kw_args={"cols": cat_columns+num_columns})),
        ('imputer', CustomKNNImputer(n_neighbors=1,))
    ])

    one_hot_encoding_preprocess = FeatureUnion(transformer_list=[
        ('cat_features', ColumnTransformer(
            transformers=[('categorical', OneHotEncoder(), cat_columns), ]
        )),
        ('num_features', ColumnTransformer(
            transformers=[('numeric', 'passthrough', num_columns), ]
        ))
    ])

    preprocessing = Pipeline(steps=[
        ('impute', impute_preprocess),
        ('one_hot_encoded', one_hot_encoding_preprocess)
    ])

    pipe = Pipeline(steps=[
        ('transformations', transformations),
        ('preprocess', preprocessing),
        ('model', LogisticRegression(random_state=123, fit_intercept=True, max_iter=1000))
    ], verbose=True)

    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Loan_Status']),
                                                        df['Loan_Status'], stratify=df['Loan_Status'], random_state=60, train_size=0.6)

    X_train = X_train.sort_index()
    y_train = y_train.sort_index()
    X_test = X_test.sort_index()
    y_test = y_test.sort_index()

    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    params = {}

    model = GridSearchCV(pipe, param_grid=params)
    model = model.fit(X_train, y_train)

    logging.info(f1_score(y_train, model.predict(X_train)))
    logging.info(f1_score(y_test, model.predict(X_test)))

    joblib.dump(model, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
