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


def preprocess_data(X, is_train=True):
    if is_train:
        return prepreccess_duration._preprocess_data(X, is_train=True)
    return prepreccess_duration.preprocess_test(X, is_train=False)


def plot_predictions(y_true, y_pred, mse):
    plt.plot(y_true, color='red', label='Real Values')
    plt.plot(y_pred, color='blue', label='Predicted Values')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title(f"Real Values vs Predicted Values with (MSE = {mse})")
    plt.legend()
    plt.show()

def plot_variance_between_y_true_and_y_pred(y_true, y_pred):
    plt.plot(np.abs(y_true - y_pred), color='red', label='Real Values')
    plt.title(f"Real Values vs Predicted Values")
    plt.legend()
    plt.show()

def train_and_evaluate(X_train, X_valid, y_train, y_valid, model_type, poly=None):
    if model_type == 'base':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge()
    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gb':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_type == 'xgb':
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    elif model_type == 'lgb':
        model = lgb.LGBMRegressor(random_state=42)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    X_train_processed = X_train
    X_valid_processed = X_valid

    logging.info("training...")
    model.fit(X_train_processed, y_train)
    y_pred_on_valid = model.predict(X_valid_processed)
    mse = mean_squared_error(y_pred_on_valid, y_valid)
    score = model.score(X_valid_processed, y_valid)
    mse = round(mse, 3)

    return model, mse, y_pred_on_valid, score


def main():
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True, help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True, help="path to the test set")
    parser.add_argument('--out', type=str, required=True, help="path of the output file as required in the task description")
    parser.add_argument('--train', type=bool, required=True, help="is training or test")
    parser.add_argument('--model_type', type=str, required=True, choices=['base', 'ridge', 'rf', 'gb', 'xgb', 'lgb'], help="type of model to use")
    parser.add_argument('--bootstrap', type=bool, required=True, help="bootstrap")

    args = parser.parse_args()

    is_train = args.train
    seed = 0
    test_size = 0.2

    print("train" if is_train else "test")

    if is_train:
        df = load_data(args.training_set)

        logging.info("preprocessing train...")
        preprocess_x, preprocess_y = preprocess_data(df)

        X_train, X_valid, y_train, y_valid = train_test_split(preprocess_x, preprocess_y, test_size=test_size,
                                                              random_state=seed)
        X_train, X_valid, y_train, y_valid = \
            (np.array(X_train), np.array(X_valid), np.array(y_train), np.array(y_valid))

        model, mse, y_pred_on_valid, score = train_and_evaluate(X_train, X_valid, y_train, y_valid, args.model_type)
        print(f"mse={mse}")
        print(f"score={score}")
        plot_variance_between_y_true_and_y_pred(y_valid, y_pred_on_valid)

        # save the model
        with open(f"model_task1.sav", "wb") as f:
            pickle.dump(model, f)

    else:
        with open("model_task1.sav", "rb") as file:
            model = pickle.load(file)
            df = load_data(args.test_set)
            logging.info("preprocessing test...")
            X_test_processed = preprocess_data(df, is_train=False)

            logging.info("predicting...")
            y_pred = model.predict(X_test_processed)

            logging.info(f"predictions saved to {args.out}")
            predictions = pd.DataFrame(
                {'trip_id_unique_station': X_test_processed['trip_id_unique_station'], 'passenger_up': y_pred})
            predictions.to_csv(args.out, index=False)


# def zone_divide(data):
#     coords = data[['latitude', 'longitude']]
#     kmeans = KMeans(n_clusters=2000, random_state=0)
#     data['zone'] = kmeans.fit_predict(coords)
# 
# 
# def remove_columns(data, *columns):
#     return data.drop(columns=list(columns))
# 
# 
# def remove_rows(data):
#     data = data[data['direction'].isin([1, 2])]
#     return data


# def calculate_time_difference(data, time_format='%H:%M:%S'):
#     # Ensure the times are in the correct format
#     def correct_time_format(time_str):
#         if pd.isna(time_str):
#             return time_str
#         parts = str(time_str).split(':')
#         if len(parts) == 2:
#             return f"{time_str}:00"
#         elif len(parts) == 3 and len(parts[2]) == 1:
#             return f"{time_str}0"
#         return time_str
# 
#     data['door_closing_time'] = data['door_closing_time'].apply(correct_time_format)
#     data['arrival_time'] = data['arrival_time'].apply(correct_time_format)
# 
#     def safe_time_diff(row):
#         try:
#             if pd.notna(row['door_closing_time']) and pd.notna(row['arrival_time']):
#                 return (datetime.strptime(row['door_closing_time'], time_format) -
#                         datetime.strptime(row['arrival_time'], time_format)).total_seconds() / 60
#         except ValueError:
#             return None  # If there's a problem parsing the times, return None
#         return None
# 
#     data["time_in_station"] = data[['door_closing_time', 'arrival_time']].apply(safe_time_diff, axis=1)
#     return data

# def preprocess(df: pd.DataFrame, is_test: bool = False):
#     # remove SCHLE features
#     # df = df.drop(columns=SCHLE)
# 
#     # remove entries with missing values
#     df = df.dropna()
# 
#     # convert time to int
#     df["arrival_time_dt"] = pd.to_datetime(
#         df["arrival_time"], format="%H:%M:%S"
#     ).dt.time
#     df["door_closing_time_dt"] = pd.to_datetime(
#         df["door_closing_time"], format="%H:%M:%S"
#     ).dt.time
# 
#     def convert_time_to_int(t) -> int:
#         return t.hour * 3600 + t.minute * 60 + t.second
# 
#     df["arrival_time"] = df["arrival_time_dt"].apply(convert_time_to_int)
#     df["door_closing_time"] = df["door_closing_time_dt"].apply(convert_time_to_int)
# 
#     df = df.drop(columns=["arrival_time_dt", "door_closing_time_dt"])
# 
#     if not is_test:
#         y_travel_times = df.groupby("trip_id_unique").agg(
#             {
#                 "arrival_time": ["min", "max"],
#             }
#         )
# 
#         y_travel_times.columns = [
#             "start_time",
#             "end_time",
#         ]
#         y_travel_times["trip_duration_in_minutes"] = (
#                                                              y_travel_times["end_time"] - y_travel_times["start_time"]
#                                                      ) / 60
# 
#     # extract travel time
#     travel_times = df.groupby("trip_id_unique").agg(
#         {
#             "trip_id_unique": "first",
#             "latitude": ["first", "last"],
#             "longitude": ["first", "last"],
#             "trip_id": "first",
#             "line_id": "first",
#             "direction": "first",
#             "cluster": "first",
#             "passengers_up": "mean",
#             "passengers_continue": "mean",
#             "mekadem_nipuach_luz": "mean",
#             "passengers_continue_menupach": "mean",
#         }
#     )
#     # print("hello")
#     travel_times.columns = [
#         "trip_id_unique",
#         "start_latitude",
#         "end_latitude",
#         "start_longitude",
#         "end_longitude",
#         "trip_id",
#         "line_id",
#         "direction",
#         "cluster",
#         "passengers_up",
#         "passengers_continue",
#         "mekadem_nipuach_luz",
#         "passengers_continue_menupach",
#     ]

    # print(travel_times)

    # Calculate aerial distance
    # def calculate_distance(row):
    #     start_coords = (row["start_latitude"], row["start_longitude"])
    #     end_coords = (row["end_latitude"], row["end_longitude"])
    #     return geodesic(start_coords, end_coords).kilometers

    # travel_times["aerial_distance"] = travel_times.apply(calculate_distance, axis=1)

    # travel_times = travel_times[['trip_duration_in_minutes', 'aerial_distance']]

    # df = df.merge(travel_times[['trip_id_unique', 'trip_duration_in_minutes', 'aerial_distance']], on='trip_id_unique', how='left')

    # df = pd.get_dummies(df, columns=['trip_id_unique'])
    # print(travel_times.columns)


    # X_travel_times = pd.get_dummies(travel_times, columns=["trip_id_unique", "cluster"])
    # X_travel_times = travel_times.drop(columns=["trip_id_unique", "cluster"])
    # 
    # if not is_test:
    #     return X_travel_times, y_travel_times["trip_duration_in_minutes"]
    # else:
    #     return X_travel_times, travel_times["trip_id_unique"]



# def train_model(X, Y, k=3):
#     models = [XGBRegressor(), LinearRegression()]
#     models[0].fit(X, Y)
#     poly = PolynomialFeatures(degree=k)
#     X_poly = poly.fit_transform(X)
#     models[1].fit(X_poly, Y)
#     return models, poly


# def predict(models, test_data, poly, i=0) -> pd.DataFrame:
#     if 'passengers_up' in test_data.columns:
#         test_data = test_data.drop(columns=['passengers_up'])
#     if i == 1:
#         test_data = poly.transform(test_data)
#     return models[i].predict(test_data)


# def save_predictions(predictions, out_path):
#     df = pd.DataFrame(predictions, columns=['passengers_up'])
#     df.to_csv(out_path, encoding='ISO-8859-8', index=False)


# def union_station(station_1, station_2):
#     ...
#     return station_1


if __name__ == '__main__':
    logging.info("starting...")
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True, help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True, help="path to the test set")
    parser.add_argument('--out', type=str, required=True, help="path of the output file as required in the task description")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    print("loading training set...")
    train_data = load_data(args.training_set)

    print("preprocessing training set...")
    processed_train_data = preprocess(train_data, True)

    Y = processed_train_data['passengers_up']
    X = processed_train_data.drop(columns=['passengers_up'])

    logging.info("training model...")
    models, poly = train_model(X, Y, k=3)

    # logging.info("loading test set...")
    test_data = load_data(args.test_set)

    # logging.info("preprocessing test set...")
    processed_test_data = preprocess_data(test_data, False)

    logging.info("making predictions...")
    prediction = predict(models, processed_test_data, poly)

    logging.info("saving predictions to {}".format(args.out))
    save_predictions(prediction, args.out)

    logging.info("evaluating model on training data...")
    train_predictions = predict(models, X, poly)
    mse_train = mean_squared_error(Y, train_predictions)
    print(f"polynomial degree: 3, mse: {mse_train}")
    logging.info("done.")



# def preprocess_data(data, is_train):
#     # Zone division before transformation
#     # zone_divide(data)
#     print("done zone")
#     calculate_time_difference(data)
#     print("done time")
# 
#     columns_to_remove = ['trip_id', 'part', 'cluster', 'station_name', 'latitude', 'longitude']
#     data = data.drop(columns=columns_to_remove)
# 
#     # Write the new data with the zone column to a new file
#     data.to_csv('files/duration/data_with_zone.csv', index=False)
# 
#     label_encoders = {}
#     for column in data.select_dtypes(include=['object']).columns:
#         if column not in label_encoders:
#             label_encoders[column] = LabelEncoder()
#         data[column] = label_encoders[column].fit_transform(data[column])
# 
#     imputer = SimpleImputer(strategy='mean')
#     data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
# 
#     scaler = StandardScaler()
#     data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
# 
#     # Write the preprocessed data to a new file
#     data.to_csv('files/duration/preprocessed_data.csv', index=False)
# 
#     group_data = data.groupby('trip_id_unique').agg({
#     'trip_id': 'min',
#     'part': 'first',
#     'line_id': 'first',
#     'direction': 'first',
#     'alternative': 'first',
#     'cluster': 'first',
#     'station_index': 'min',
#     'station_id': 'first',
#     'station_name': 'first',
#     'arrival_time': 'first',
#     'door_closing_time': 'first',
#     'arrival_is_estimated': 'first',
#     'latitude': 'first',
#     'longitude': 'first',
#     'passengers_up': 'sum',
#     'passengers_continue': 'sum',
#     'mekadem_nipuach_luz': 'sum',
#     'passengers_continue_menupach': 'sum'
#     })
# 
#     result_rows = []
#     for trip_id, group in grouped_data:
#         result_rows.append(aggregate(group, is_train))
#     result_df = pd.DataFrame(result_rows)
#     result_df.to_csv("duration_after_preprocess.csv", index=False, encoding='ISO-8859-8')
# 
#     return result_df
# 
# 
# def preprocess_data2(data, is_train):
#     # Zone division before transformation
#     # zone_divide(data)
#     print("done zone")
#     calculate_time_difference(data)
#     print("done time")
# 
#     # columns_to_remove = ['trip_id', 'part', 'cluster', 'station_name', 'latitude', 'longitude']
#     # data = data.drop(columns=columns_to_remove)
# 
#     # Write the new data with the zone column to a new file
#     data.to_csv('files/duration/data_with_zone.csv', index=False)
# 
#     label_encoders = {}
#     for column in data.select_dtypes(include=['object']).columns:
#         if column not in label_encoders:
#             label_encoders[column] = LabelEncoder()
#         data[column] = label_encoders[column].fit_transform(data[column])
# 
#     imputer = SimpleImputer(strategy='mean')
#     data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
# 
#     scaler = StandardScaler()
#     data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
# 
#     # Write the preprocessed data to a new file
#     data.to_csv('files/duration/preprocessed_data.csv', index=False)
# 
#     # Group the data and apply the required aggregations
#     # aggregated_data = data.groupby('trip_id_unique').agg(
#     #     trip_id_unique=pd.NamedAgg(column='trip_id_unique', aggfunc='first'),
#     #     line_id=pd.NamedAgg(column='line_id', aggfunc='first'),
#     #     direction=pd.NamedAgg(column='direction', aggfunc='first'),
#     #     # passengers_up=pd.NamedAgg(column='passengers_up', aggfunc='sum'),
#     #     station_count=pd.NamedAgg(column='station_index', aggfunc='count')
#     #     # trip_start_time=pd.NamedAgg(column='arrival_time', aggfunc='first'),
#     #     # hold_time=pd.NamedAgg(column='time_in_station', aggfunc='sum'),
#     #     # trip_duration_in_minute=pd.NamedAgg(column='arrival_time',
#     #     #                                     aggfunc=lambda x: (datetime.strptime(x.iloc[-1], '%H:%M:%S') - datetime.strptime(x.iloc[0], '%H:%M:%S')).total_seconds() / 60)
#     # ).reset_index()
# 
#     aggregated_data = data.groupby('trip_id_unique').agg(
#         trip_id_unique=pd.NamedAgg(column='trip_id_unique', aggfunc='first'),
#         line_id=pd.NamedAgg(column='line_id', aggfunc='first'),
#         direction=pd.NamedAgg(column='direction', aggfunc='first'),
#         station_count=pd.NamedAgg(column='station_index', aggfunc='count')
#     ).reset_index()
# 
#     aggregated_data.to_csv("duration_after_preprocess.csv", index=False, encoding='ISO-8859-8')
# 
#     return aggregated_data







# 
# def aggregate(group, is_train=True):
#     if is_train:
#         return {
#             'trip_id_unique': group['trip_id_unique'].iloc[0],
#             'line_id': group['line_id'].iloc[0],
#             'direction': group['direction'].iloc[0],
#             'passengers_up': group['passengers_up'].sum(),
#             'station_counter': len(group),
#             'trip_start_time': group['arrival_time'].iloc[0],
#             'hold_time': group['time_in_station'].sum(),
#             'trip_duration_in_minute': (datetime.strptime(group['arrival_time'].iloc[-1], '%H:%M:%S') - datetime.strptime(group['arrival_time'].iloc[0], '%H:%M:%S')).total_seconds() / 60
#         }
#     else:
#         return {
#             'trip_id_unique': group['trip_id_unique'].iloc[0],
#             'line_id': group['line_id'].iloc[0],
#             'direction': group['direction'].iloc[0],
#             'passengers_up_sum': group['passengers_up'].sum(),
#             'station_counter': len(group),
#             'trip_start_time': group['arrival_time'].iloc[0],
#             'hold_time': group['time_in_station'].sum(),
#         }
