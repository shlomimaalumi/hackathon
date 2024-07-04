# -*- coding: ISO-8859-8 -*-


import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

"""
Given data for a single bus stop, your task is to predict the number of passengers boarding the bus at
that stop.
Input.
A csv where each row holds information of a single bus stop within some bus route ? all columns
except passengers_up.
Output.
A csv file named ?passengers_up_predictions.csv?, including two columns:
trip_id_unique_station and passengers_up. An example of the output is provided in Table 1 and in
y_passengers_up_example.csv.

trip_id_unique_station passengers_up
111a1 0
222b5 12
333c1 3
Table 1: An example of passenger boarding prediction output.
Evaluation.
We will evaluate your predictions according to their mean squared error (MSE) metric.

"""


class ModelPredictPassengersUp:
    """
    the model use Xgboost to predict the number of passengers boarding the bus at that stop.
    also the model use the following features:
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def load_data(self, file_path):
        return pd.read_csv(file_path, encoding='ISO-8859-8')

    def preprocess(self, data):
        # Drop columns that are not needed
        data = data.drop(
            columns=['trip_id', 'part', 'line_id', 'alternative', 'cluster', 'station_name', 'arrival_time',
                     'door_closing_time'])

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        data[['latitude', 'longitude', 'passengers_continue', 'mekadem_nipuach_luz',
              'passengers_continue_menupach']] = imputer.fit_transform(data[['latitude', 'longitude',
                                                                             'passengers_continue',
                                                                             'mekadem_nipuach_luz',
                                                                             'passengers_continue_menupach']])

        # Encode categorical features
        categorical_features = ['trip_id_unique_station', 'trip_id_unique', 'direction', 'arrival_is_estimated']
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                data[feature] = self.label_encoders[feature].fit_transform(data[feature])
            else:
                data[feature] = self.label_encoders[feature].transform(data[feature])

        # Normalize/scale features
        numerical_features = ['latitude', 'longitude', 'passengers_continue', 'mekadem_nipuach_luz',
                              'passengers_continue_menupach']
        data[numerical_features] = self.scaler.fit_transform(data[numerical_features])

        return data

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def predict(self, X):
        return self.model.predict(X)

    def save(self, data, file_path):
        data.to_csv(file_path, encoding='ISO-8859-8')

    def MSE(self, predictions, ground_truth):
        combined = pd.merge(predictions, ground_truth, on='trip_id_unique_station')
        mse_board = mean_squared_error(combined["passengers_up_x"], combined["passengers_up_y"])
        return mse_board


if __name__ == '__main__':
    model = ModelPredictPassengersUp()
    schedule_df = model.load_data("files/passengers/train_data.csv")
    schedule_df = model.preprocess(schedule_df)
    # save the data after preprocessing:
    schedule_df.to_csv("files/passengers/preprocessed_data.csv", encoding='ISO-8859-8')
    X = schedule_df.drop(columns=['passengers_up'])
    y = schedule_df['passengers_up']
    model.model = model.train(X, y)
    # save the model
    model.save(model.model, "files/passengers/model.pkl")
    # predict the test data
    test_data = model.load_data("files/passengers/test_data.csv")
    test_data = model.preprocess(test_data)
    X_test = test_data.drop(columns=['passengers_up'])
    y_test = test_data['passengers_up']
    predictions = model.predict(X_test)
    test_data['passengers_up'] = predictions
    model.save(test_data, "files/passengers/predictions.csv")
    # evaluate the model
    ground_truth = model.load_data("files/passengers/val_data.csv")
    ground_truth = model.preprocess(ground_truth)
    mse = model.MSE(test_data, ground_truth)
    print(f"MSE for boardings: {mse}")
    print("done")
