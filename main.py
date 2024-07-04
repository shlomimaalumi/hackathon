# -*- coding: ISO-8859-8 -*-
from argparse import ArgumentParser
import logging
from datetime import datetime

from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from xgboost import XGBRegressor


def load_data(file_path) -> pd.DataFrame:
    return pd.read_csv(file_path, encoding='ISO-8859-8')


def zone_divide(data):
    coords = data[['latitude', 'longitude']]
    kmeans = KMeans(n_clusters=2000, random_state=0)
    data['zone'] = kmeans.fit_predict(coords)


def remove_columns(data, *columns):
    return data.drop(columns=list(columns))

def calculate_time_difference(data):
    time_format = '%H:%M'
    data["time_in_station"] = df.apply(lambda row: (datetime.strptime(row['door_closing_time'], time_format) -
                                                    datetime.strptime(row['arrival_time'], time_format)).total_seconds() / 60, axis=1)
    return data

def time_difference_in_minutes(time1, time2):
    h1, m1 = map(int, time1.split(':'))
    h2, m2 = map(int, time2.split(':'))
    return (h1 * 60 + m1) - (h2 * 60 + m2)


def calculate_time_difference(data):
    time_format = '%H:%M'
    data["time_in_station"] = data.apply(lambda row: (datetime.strptime(row['door_closing_time'], time_format) -
                                                      datetime.strptime(row['arrival_time'], time_format)).total_seconds() / 60, axis=1)
    return data


def preprocess_data(data):

    # Zone division before transformation
    zone_divide(data)
    
    columns_to_remove = ['trip_id', 'part', 'cluster', 'station_name', 'latitude', 'longitude']
    data = data.drop(columns=columns_to_remove)
    # remove
    # write the new data with the zone column to a new file
    data.to_csv('data_with_zone.csv', index=False)

    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        if column not in label_encoders:
            label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return data


def train_model(X, Y):
    models = [XGBRegressor(), LinearRegression()]
    models[0].fit(X, Y)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    models[1].fit(X_poly, Y)
    return models, poly


def predict(models, test_data, poly, i=0) -> pd.DataFrame:
    if 'passengers_up' in test_data.columns:
        test_data = test_data.drop(columns=['passengers_up'])
    if i == 1:
        test_data = poly.transform(test_data)
    return models[i].predict(test_data)


def save_predictions(predictions, out_path):
    df = pd.DataFrame(predictions, columns=['passengers_up'])
    df.to_csv(out_path, encoding='ISO-8859-8', index=False)


if __name__ == '__main__':
    logging.info("starting...")
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True, help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True, help="path to the test set")
    parser.add_argument('--out', type=str, required=True, help="path of the output file as required in the task description")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logging.info("loading training set...")
    train_data = load_data(args.training_set)

    logging.info("preprocessing training set...")
    processed_train_data = preprocess_data(train_data)
    Y = processed_train_data['passengers_up']
    X = processed_train_data.drop(columns=['passengers_up'])

    logging.info("training model...")
    models, poly = train_model(X, Y)

    logging.info("loading test set...")
    test_data = load_data(args.test_set)

    logging.info("preprocessing test set...")
    processed_test_data = preprocess_data(test_data)

    logging.info("making predictions...")
    prediction = predict(models, processed_test_data, poly)

    logging.info("saving predictions to {}".format(args.out))
    save_predictions(prediction, args.out)

    # Calculate MSE on training data to evaluate model performance
    logging.info("evaluating model on training data...")
    train_predictions = predict(models, X, poly)
    mse_train = mean_squared_error(Y, train_predictions)
    print(f"MSE for model on training data: {mse_train}")
    logging.info(f"MSE for model on training data: {mse_train}")

    logging.info("done.")
