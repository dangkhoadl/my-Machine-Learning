
from pandas import DataFrame, pandas as pd
from sklearn import preprocessing

# creating sample data
sample_data = {
    'name': ['Ray', 'Adam', 'Jason', 'Varun', 'Xiao'],
    'health':['fit', 'slim', 'obese', 'fit', 'slim']}

# storing sample data in the form of a dataframe
data = DataFrame(sample_data, columns = ['name', 'health'])

# Encode
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(data['health'])

# Transform the column to integer data (array([0, 2, 1, 0, 2]))
label_encoder.transform(data['health'])


##### One-hot Encoder #####

pd.get_dummies(data['health'])

# creating OneHotEncoder object
ohe = preprocessing.OneHotEncoder() 

# 
label_encoded_data = label_encoder.fit_transform(data['health'])
ohe.fit_transform(label_encoded_data.reshape(-1,1))
