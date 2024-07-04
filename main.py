# -*- coding: ISO-8859-8 -*-
from argparse import ArgumentParser
import logging

import pandas as pd
# from scipy.special import df
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler



def load_data(file_path) -> pd.DataFrame:
    return pd.read_csv(file_path, encoding='ISO-8859-8')

def zone_divide(data):
    coords = data[['latitude', 'longitude']].values
    # kmeans = KMeans(n_clusters=2000, random_state=0)
    # df['region'] = kmeans.fit_predict(coords)
    # output_file_path = r'C:\Users\User\Desktop\Programing\IML\Hakaton\coordinates_with_regions.csv'
    # df.to_csv(output_file_path, index=False)

def preprocess_data(data):
    # Implement the logic to preprocess the data
    columns_to_remove = ['trip_id', 'part', 'cluster', 'station_name']
    data = data.drop(columns=columns_to_remove)
    label_encoders = {}
    # Encode the categorical columns
    for column in data.select_dtypes(include=['object']).columns:
        if column not in label_encoders:
            label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    data = imputer.fit_transform(data)

    # Scale the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    zone_divide(data)
    return data


def train_model(training_data):
    # Implement the logic to train the model
    pass


def predict(model, test_data):
    # Implement the logic to make predictions using the trained model
    pass


def save_predictions(predictions, out_path):
    # Implement the logic to save predictions to the specified file
    pass


if __name__ == '__main__':
    logging.info("starting...")
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True, help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True, help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # 1. load the training set
    logging.info("loading training set...")
    print(args)
    print("hello")
    train_data = load_data(args.training_set)

    # 2. preprocess the training set
    logging.info("preprocessing training set...")
    processed_train_data = preprocess_data(train_data)
    # 
    # # 3. train a model
    # logging.info("training model...")
    # # model = train_model(processed_train_data)
    # 
    # # 4. load the test set
    # logging.info("loading test set...")
    # test_data = load_data(args.test_set)
    # 
    # # 5. preprocess the test set
    # logging.info("preprocessing test set...")
    # processed_test_data = preprocess_data(test_data)
    # 
    # # 6. predict the test set using the trained model
    # logging.info("making predictions...")
    # predictions = predict(model, processed_test_data)
    # 
    # # 7. save the predictions to args.out
    # logging.info("saving predictions to {}".format(args.out))
    # save_predictions(predictions, args.out)
    logging.info("done.")
