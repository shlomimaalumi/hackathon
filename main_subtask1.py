# -*- coding: ISO-8859-8 -*-
from argparse import ArgumentParser
import logging
from datetime import datetime

from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


def load_data(file_path) -> pd.DataFrame:
    return pd.read_csv(file_path, encoding='ISO-8859-8')


def zone_divide(data):
    coords = data[['latitude', 'longitude']]
    kmeans = KMeans(n_clusters=2000, random_state=0)
    data['zone'] = kmeans.fit_predict(coords)


def remove_columns(data, *columns):
    return data.drop(columns=list(columns))


def remove_rows(data):
    data = data.drop_duplicates(subset='trip_id_unique')
    data = data[data['direction'].isin([1, 2])]
    return data


def calculate_time_difference(data, time_format='%H:%M:%S'):
    # Ensure the times are in the correct format
    data['door_closing_time'] = data['door_closing_time'].apply(
        lambda x: x if pd.isna(x) or len(str(x)) == 8 else str(x) + ':00')
    data['arrival_time'] = data['arrival_time'].apply(
        lambda x: x if pd.isna(x) or len(str(x)) == 8 else str(x) + ':00')

    def safe_time_diff(row):
        try:
            if pd.notna(row['door_closing_time']) and pd.notna(row['arrival_time']) and row[
                'arrival_is_estimated'] == "FALSE":
                return (datetime.strptime(row['door_closing_time'], time_format) -
                        datetime.strptime(row['arrival_time'], time_format)).total_seconds() / 60
            else:
                return None
        except Exception as e:
            return None

    data["time_in_station"] = data.apply(safe_time_diff, axis=1)
    return data


def preprocess_data(data):
    # Zone division before transformation
    zone_divide(data)
    print("done zone")
    calculate_time_difference(data)
    print("done time")

    columns_to_remove = ['trip_id', 'part', 'cluster', 'station_name', 'latitude', 'longitude']
    data = data.drop(columns=columns_to_remove)

    # Write the new data with the zone column to a new file
    data.to_csv('files/passengers/data_with_zone.csv', index=False)

    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        if column not in label_encoders:
            label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # # imputer = SimpleImputer(strategy='mean')
    # # data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    # # 
    # # scaler = StandardScaler()
    # data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    # Write the preprocessed data to a new file
    data.to_csv('files/passengers/preprocessed_data.csv', index=False)

    return data


def train_model(X, Y):
    models = [XGBRegressor(), LinearRegression()]
    models[0].fit(X, Y)
    return models


def predict(models, X, i=0) -> pd.DataFrame:
    return models[i].predict(X)


def save_predictions(predictions, out_path, X_test):
    # i want to save in this format:
    # trip_id_unique_station	passengers_up
    # 310341a1	4
    # 310341a2	22
    # 310341a3	22
    # 310341a4	23
    # 310341a5	20
    # 310341a6	26
    # 310341a7	14
    # 310341a8	11
    # 310341a9	16
    # 310341a10	22
    
    df = pd.DataFrame({'trip_id_unique_station': X_test['trip_id_unique_station'], 'passengers_up': predictions})
    df.to_csv(out_path, index=False)
    


if __name__ == '__main__':
    logging.info("starting...")
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True, help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True, help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logging.info("loading training set...")
    train_data = load_data(args.training_set)
    print(args)
    logging.info("preprocessing training set...")
    processed_train_data = preprocess_data(train_data)
    Y = processed_train_data['passengers_up']
    X = processed_train_data.drop(columns=['passengers_up'])

    logging.info("training model...")
    models = train_model(X, Y)

    logging.info("loading test set...")
    test_data = load_data(args.test_set)

    logging.info("preprocessing test set...")
    processed_test_data = preprocess_data(test_data)

    logging.info("making predictions...")
    X_test = processed_test_data.drop(columns=['passengers_up'])
    test_predictions = predict(models, X_test)

    logging.info("saving predictions to {}".format(args.out))
    save_predictions(test_predictions, args.out, X_test)

    # Calculate MSE on training data to evaluate model performance
    logging.info("evaluating model on training data...")
    train_predictions = predict(models, X)
    mse_train = mean_squared_error(Y, train_predictions)
    print(f"MSE on training data: {mse_train}")
    logging.info("done.")

    # Initialize the model
    model = RandomForestRegressor(random_state=42)

    # Perform k-fold cross-validation
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    cross_val_scores = cross_val_score(model, X, Y, cv=kf, scoring='neg_mean_squared_error')

    # Calculate mean and standard deviation of the cross-validation scores
    mean_cv_mse = -cross_val_scores.mean()
    std_cv_mse = cross_val_scores.std()

    # Train the model on the full training data
    model.fit(X, Y)

    # Final evaluation on the test set
    final_test_predictions = model.predict(X_test)
    test_mse = mean_squared_error(processed_test_data['passengers_up'], final_test_predictions)

    # Plotting the results
    plt.figure(figsize=[14, 6])

    # Plot cross-validation MSE scores
    plt.subplot(1, 2, 1)
    plt.boxplot(-cross_val_scores)
    plt.title('Cross-Validation MSE Scores')
    plt.ylabel('MSE')
    plt.xticks([1], ['Random Forest'])

    # Plot predicted vs actual values on the test set
    plt.subplot(1, 2, 2)
    plt.scatter(processed_test_data['passengers_up'], final_test_predictions, alpha=0.5)
    plt.plot([processed_test_data['passengers_up'].min(), processed_test_data['passengers_up'].max()],
             [processed_test_data['passengers_up'].min(), processed_test_data['passengers_up'].max()], 'r--', lw=2)
    plt.title('Test Set: Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    plt.tight_layout()
    plt.show()

    print(f'Mean Cross-Validation MSE: {mean_cv_mse}')
    print(f'Standard Deviation of Cross-Validation MSE: {std_cv_mse}')
    print(f'Test MSE: {test_mse}')
