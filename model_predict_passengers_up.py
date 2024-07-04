import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

"""
Given data for a single bus stop, your task is to predict the number of passengers boarding the bus at
that stop.
Input.
A csv where each row holds information of a single bus stop within some bus route – all columns
except passengers_up.
Output.
A csv file named “passengers_up_predictions.csv”, including two columns:
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
    def __init__(self, training_set_path, test_set_path, out_path):
        ...

    def load_data(self, file_path):
        ...

    def preprocess(self, data):
        ...

    def train(self, X, y):
        ...

    def predict(self, X):
        ...

    def save(self, data, file_path):
        ...
