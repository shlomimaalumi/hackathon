import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
file_path = 'train_bus_schedule.csv'
schedule_df = pd.read_csv(file_path, encoding='latin1')

# Split the data into 75% train, 20% test, and 5% dev
train_data, temp_data = train_test_split(schedule_df, test_size=0.25, random_state=42)
test_data, dev_data = train_test_split(temp_data, test_size=0.2, random_state=42)





# Save the dataframes to csv files
train_data.to_excel('file_train..csv', index=False)
test_data.to_excel('file_test..csv', index=False)
dev_data.to_excel('file_dev..csv', index=False)