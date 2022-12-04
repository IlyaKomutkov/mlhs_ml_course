from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge

import pandas as pd
import pickle

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


def get_models():
    with open('params.pickle', 'rb') as f:
        params = pickle.load(f)
        scaler = StandardScaler()
        encoding = OneHotEncoder()
        ridge = Ridge()
        scaler.mean_ = params['scaler_mean']
        scaler.scale_ = params['scaler_scale']
        scaler.var_ = params['scaler_var']
        encoding.categories_ = params['encoding_categories']
        encoding.drop_idx_ = params['encoding_drop_idx']
        encoding.n_features_in_ = params['encoding_n_features_in']
        encoding.feature_names_in_ = params['encoding_feature_names_in']
        encoding._infrequent_enabled = params['encoding_infrequent_enabled']
        encoding._n_features_outs = params['encoding_n_features_outs']
        ridge.coef_ = params['ridge_coef']
        ridge.intercept_ = params['ridge_intercept']
        ridge.n_iter_ = params['ridge_in_iter']
        ridge.n_features_in_ = params['ridge_n_features_in']
        ridge.feature_names_in_ = params['ridge_feature_names_in']
        ridge.alpha = params['ridge_alpha']
    return scaler, encoding, ridge


scaler_model, encoding_model, ridge_model = get_models()


def parse_torque(df):
    torque = df.torque.str.extract(r"([\.\d]+)\s*([A-Za-z]*)\s*[@at]+ ([\~\d,-]+)(.*)")
    torque[3] = torque[3].replace('(kgm@ rpm)', 'kgm')
    torque[3] = torque[3].replace('+/-500(NM@ rpm)', 'Nm')
    torque.loc[(torque[1] == '') &
               ((torque[3] == 'kgm') |
                (torque[3] == 'Nm')), 1] = torque.loc[(torque[1] == '') &
                                                      ((torque[3] == 'kgm') |
                                                       (torque[3] == 'Nm')), 3]
    torque[2] = torque[2].replace('[\d,]+[\~-]', '', regex=True)
    torque[2] = torque[2].str.replace(',', '').astype(float)
    torque[0] = torque[0].astype(float)
    torque.loc[(torque[1] == 'kgm'), 0] *= 9.80665
    df[['torque', 'max_torque_rpm']] = torque[[0, 2]]
    return df


def preprocess_data(df):
    df.mileage = df.mileage.str.replace(' kmpl', '')
    df.mileage = df.mileage.str.replace(' km/kg', '')
    df.mileage = df.mileage.astype(float)

    df.engine = df.engine.str.replace(' CC', '')
    df.engine = df.engine.astype(float)

    df.max_power = df.max_power.str.replace(' bhp', '')
    df.max_power = df.max_power.str.replace('', '0')
    df.max_power = df.max_power.astype(float)

    df = parse_torque(df)

    df.engine = df.engine.astype(int)
    df.seats = df.seats.astype(int)

    float_features = df[['year', 'km_driven', 'mileage', 'engine',
                         'max_power', 'torque', 'seats', 'max_torque_rpm']]
    float_features = scaler_model.transform(float_features)
    float_features = pd.DataFrame(float_features, columns=['year', 'km_driven', 'mileage',
                                                           'engine', 'max_power', 'torque', 'seats', 'max_torque_rpm'])
    float_features.drop(columns=['seats'], inplace=True)

    cat_features = pd.DataFrame(encoding_model.transform(df[['fuel', 'seller_type',
                                                             'transmission', 'owner', 'seats']]).toarray(),
                                columns=encoding_model.get_feature_names_out(['fuel', 'seller_type',
                                                                              'transmission', 'owner', 'seats']))

    features = pd.concat([float_features, cat_features], axis=1)

    return features


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame([item.dict()])
    features = preprocess_data(df)
    return ridge_model.predict(features)[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    df = pd.DataFrame([item.dict() for item in items])
    features = preprocess_data(df)
    return list(ridge_model.predict(features))
