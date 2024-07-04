# -*- coding: ISO-8859-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

# Ensure the output encoding is set to handle non-ASCII characters
sys.stdout.reconfigure(encoding='utf-8')

# Read the data from the CSV file
filename = 'train_bus_schedule.csv'
df = pd.read_csv(filename, encoding='ISO-8859-8')

# Display the first few rows of the dataframe
print(df.head())

# Function to split the data while keeping the same trip_id_unique together
def split_data(df, train_size=0.75, test_size=0.125, dev_size=0.125):
    # Get unique trip_id_unique values
    unique_trip_ids = df['trip_id_unique'].unique()

    # Split unique_trip_ids into train, test, and dev sets
    train_trip_ids, temp_trip_ids = train_test_split(unique_trip_ids, test_size=(test_size + dev_size))
    test_trip_ids, dev_trip_ids = train_test_split(temp_trip_ids, test_size=dev_size / (test_size + dev_size))

    # Create train, test, and dev dataframes
    train_df = df[df['trip_id_unique'].isin(train_trip_ids)]
    test_df = df[df['trip_id_unique'].isin(test_trip_ids)]
    dev_df = df[df['trip_id_unique'].isin(dev_trip_ids)]

    return train_df, test_df, dev_df

# Split the data
train_df, test_df, dev_df = split_data(df)

# Display the number of rows in each set
print(f'Training set: {len(train_df)}')
print(f'Test set: {len(test_df)}')
print(f'Dev set: {len(dev_df)}')

# Save the sets to new CSV files
train_df.to_csv('files/duration/train_data.csv', index=False, encoding='ISO-8859-8')
test_df.to_csv('files/duration/test_data.csv', index=False, encoding='ISO-8859-8')
dev_df.to_csv('files/duration/dev_data.csv', index=False, encoding='ISO-8859-8')