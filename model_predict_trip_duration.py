# -*- coding: ISO-8859-8 -*-

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split

"""
you want to predict how long is a single bus trip, from its first station to its last station. For this
task, we treat each trip_unique_id as a single sample indicating a complete bus trip (see Figure 1).
Based on all information of the stops in this trip, we want to predict the arrival time to the last stop.
Input.
A csv where each row holds information of a single bus stop within some bus trip. The test set
excludes the arrival_time to all stops within the trip, and only provides that of the first station (i.e.,
the time the bus left its first stop). The predictions should include the trip duration in minutes.
Output.
A csv file named ?trip_duration_predictions.csv?, including two columns: trip_id_unique and
trip_duration_in_minutes: An example of the output is shown in Table 2 and in
y_trip_duration_example.csv.
trip_id_unique trip_duration_in_minutes
111a 78.2
222c 122.0
333b 61.6
Table 2: Trip duration prediction output
3.3 Using your results to improve public transportation 3
Evaluation.
We will evaluate your predictions according to their mean squared error (MSE) metric.

"""


# declare API

class ModelPredictTripDuration:

    def __init__(self, ):
        ...

    def load_data(self, file_path):
        return pd.read_csv(file_path, encoding='ISO-8859-8')

    def split_data(self, data):
        ...
        self.train_data, self.test_data = train_test_split(data, test_size=0.25)
        self.test_data, self.val_data = train_test_split(self.test_data, test_size=0.5)
        # save the data
        self.train_data.to_csv("files/duration/train_data.csv", encoding='ISO-8859-8')
        self.test_data.to_csv("files/duration/test_data.csv", encoding='ISO-8859-8')
        self.val_data.to_csv("files/duration/val_data.csv", encoding='ISO-8859-8')

    def preprocess(self, data):
        ...

    def train(self, X, y):
        ...

    def predict(self, X):
        ...

    def save(self, data, file_path):
        ...


if __name__ == '__main__':
    model = ModelPredictTripDuration()
    schedule_df = model.load_data("X_trip_duration.csv")
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
