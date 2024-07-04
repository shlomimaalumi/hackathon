# -*- coding: ISO-8859-8 -*-

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, PoissonRegressor, Lasso, Ridge, ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor

COLUMNS = ['trip_id', 'part', 'line_id', 'direction', 'alternative', 'cluster', 'station_index',
           'station_id', 'station_name', 'arrival_time', 'door_closing_time', 'arrival_is_estimated',
           'latitude', 'longitude', 'passengers_continue', 'mekadem_nipuach_luz', 'passengers_continue_menupach']


class ModelPredictPassengersUp:
    """
    The model uses XGBoost to predict the number of passengers boarding the bus at that stop.
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = [XGBRegressor(), LinearRegression(), PolynomialFeatures(), PoissonRegressor(), Lasso(), Ridge(),
                       ElasticNet(), DecisionTreeRegressor(), RandomForestRegressor(), AdaBoostRegressor(),
                       GradientBoostingRegressor(), SVR(), KNeighborsRegressor(), MLPRegressor()]

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path, encoding='ISO-8859-8')
        # Split into labels and features
        self.y = self.data['passengers_up']
        self.X = self.data.drop(columns=['passengers_up'])

    def preprocess(self, data):
        """
        Preprocess the data by encoding categorical columns and imputing missing values.
        """

        # Columns to remove
        columns_to_remove = ['trip_id', 'part', 'cluster', 'station_name']
        data = data.drop(columns=columns_to_remove)

        # Encode the categorical columns
        for column in data.select_dtypes(include=['object']).columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
            data[column] = self.label_encoders[column].fit_transform(data[column])

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        data = imputer.fit_transform(data)

        # Scale the data
        data = self.scaler.fit_transform(data)

        return data

    def train(self):
        self.X_preprocessed = self.preprocess(self.X)
        for model in self.models:
            model.fit(self.X_preprocessed, self.y)

    def predict(self, X):
        lst = []
        X_preprocessed = self.preprocess(X)
        for model in self.models:
            lst.append(model.predict(X_preprocessed))
        return lst

    def save(self, predictions, file_path):
        # Save the data in the given path
        # df = pd.DataFrame(predictions, columns=['passengers_up'])
        # df.to_csv(file_path, encoding='ISO-8859-8', index=False)
        for i, prediction in enumerate(predictions):
            df = pd.DataFrame(prediction, columns=['passengers_up'])
            df.to_csv(file_path + f"{i}.csv", encoding='ISO-8859-8', index=False)

    def MSE(self, predictions, ground_truth):
        for i, prediction in enumerate(predictions):
            mse = mean_squared_error(prediction, ground_truth)
            print(f"MSE for model {i}: {mse}")

    def plot_predictions(self, predictions, ground_truth):
        plt.figure(figsize=(14, 7))

        for i, prediction in enumerate(predictions):
            plt.scatter(ground_truth, prediction, label=f'Model {i}', alpha=0.6)

        plt.plot([ground_truth.min(), ground_truth.max()], [ground_truth.min(), ground_truth.max()], 'k--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.legend()
        plt.title('Model Predictions vs True Values')
        plt.show()



if __name__ == '__main__':
    train_path = 'files/passengers/train_data.csv'
    test_path = 'files/passengers/test_data.csv'

    model = ModelPredictPassengersUp()
    model.load_data(train_path)
    model.train()

    # Comparing the predictions to the ground truth
    predictions = model.predict(model.X)
    model.save(predictions, 'predictions/passengers_up_predictions')
    model.MSE(predictions, model.y)
    # print(f"MSE for boardings: {model.MSE(predictions, model.y)}")
