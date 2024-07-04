import pandas as pd
from sklearn.model_selection import train_test_split
# Load your data
file_path = 'train_bus_schedule.csv'
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
# file_path = 'train_bus_schedule.csv'

schedule_df = pd.read_csv(file_path, encoding='ISO-8859-8')

# Split the data into 75% train, 20% test, and 5% dev
train_data, temp_data = train_test_split(schedule_df, test_size=0.25, random_state=42)
test_data, dev_data = train_test_split(temp_data, test_size=0.2, random_state=42)
# 

# Save the dataframes to CSV files with the correct encoding
train_data.to_csv('file_train.csv', index=False, encoding='ISO-8859-8')
test_data.to_csv('file_test.csv', index=False, encoding='ISO-8859-8')
dev_data.to_csv('file_dev.csv', index=False, encoding='ISO-8859-8')