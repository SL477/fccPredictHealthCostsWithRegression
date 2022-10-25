# src

Here we are storing our code

## get_data

This module provides the links to the data file

### get_date

This returns the Pandas DataFrame of the data in data/insurance.csv

### preprocess_data

This takes a Pandas DataFrame, encodes the categorical data and returns the encoded data

### split_train_test

This takes a Pandas DataFrame, then splits 80% of it out as training data and the rest as test data. This returns the following tuple:

- X_train
- y_train
- X_test
- y_test

## infer

This submodule is used to run a single prediction from the given piece of data

### infer

This function takes in the arguments:

- mdl: trained SciKit Learn model
- age (int): the age of the user
- bmi (float): the BMI of the user
- children (int): the number of children of the user
- smoker (bool): whether or not the user smokes
- gender (str): the gender of the user (Male/Female)
- region (str): the region of the user ('South West'/'South East'/'North West'/'North East')

It then returns the float of the user's predicted healthcare costs
