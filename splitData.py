import pandas as pd
from sklearn.model_selection import train_test_split
""""
the columns in the data are: 
trip_id,part,trip_id_unique_station,trip_id_unique,line_id,direction,alternative,cluster,station_index,station_id,station_name,arrival_time,door_closing_time,arrival_is_estimated,latitude,longitude,passengers_up,passengers_continue,mekadem_nipuach_luz,passengers_continue_menupach

"""


# Load your data
file_path = 'train_bus_schedule.csv'




# step 1 split the data to train test and validation
def split_data(file_path):
    data = pd.read_csv(file_path)
    train, test = train_test_split(data, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    return train, test, val

# step 2
def save_data(train, test, val):
    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)
    val.to_csv('val.csv', index=False)
    
# remove 