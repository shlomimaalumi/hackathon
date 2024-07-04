# -*- coding: ISO-8859-8 -*-


import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
    def __init__(self):
        ...

    def load_data(self, file_path):
        return pd.read_csv(file_path, encoding='ISO-8859-8')

    def split_data(self, data):
        ...
        self.train_data, self.test_data = train_test_split(data, test_size=0.25)
        self.test_data, self.val_data = train_test_split(self.test_data, test_size=0.5)
        # save the data
        self.train_data.to_csv("files/passengers/train_data.csv", encoding='ISO-8859-8')
        self.test_data.to_csv("files/passengers/test_data.csv", encoding='ISO-8859-8')
        self.val_data.to_csv("files/passengers/val_data.csv", encoding='ISO-8859-8')

    def preprocess(self, data):
        ...

    def train(self, X, y):
        ...

    def predict(self, X):
        ...

    def save(self, data, file_path):
        ...


if __name__ == '__main__':
    model = ModelPredictPassengersUp()
    schedule_df = model.load_data("X_passengers_up.csv")
    model.split_data(schedule_df)
    # 1. load the training set (args.training_set)
    # 2. preprocess the training set
    # 3. train a model
    # 4. load the test set (args.test_set)
    # 5. preprocess the test set
    # 6. predict the test set using the trained model
    # 7. save the predictions to args.out
    # logging.info("predictions saved to {}".format(args.out))
    # model.save(model.predict(model.test_data), "predictions/trip_duration_predictions.csv")
    print("done")