import pandas as pd
import os
import numpy as np
import sys
def getData():
    #return pd.read_csv('https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv')
    #return pd.read_csv('insurance.csv')
    return pd.read_csv(os.path.join(getFilePath(), 'insurance.csv'))

def getFilePath():
    #return os.path.join(os.path.dirname(__file__),'/insurance.csv')
    #return os.path.join(sys.path[0][0:-7], 'insurance.csv')
    #print('sys.path[0]', sys.path[0])
    return sys.path[0][0:-7]

def getTestAndTrainingData(df):
    # Convert text columns to codes
    # Convert categorical data into numbers
    #df = getData()
    df['smoker'] = df['smoker'] == 'yes'
    df['female'] = df['sex'] == 'female'
    df['male'] = df['sex'] == 'male'
    del df['sex']
    df['southwest'] = df['region'] == 'southwest'
    df['southeast'] = df['region'] == 'southeast'
    df['northwest'] = df['region'] == 'northwest'
    df['northeast'] = df['region'] == 'northeast'
    del df['region']

    df['age'] = df['age'].astype(np.int32)
    df['bmi'] = df['bmi'].astype(np.float32)
    df['children'] = df['children'].astype(np.int32)
    df['smoker'] = df['smoker'].astype(np.int32)
    df['female'] = df['female'].astype(np.int32)
    df['male'] = df['male'].astype(np.int32)
    df['southwest'] = df['southwest'].astype(np.int32)
    df['southeast'] = df['southeast'].astype(np.int32)
    df['northwest'] = df['northwest'].astype(np.int32)
    df['northeast'] = df['northeast'].astype(np.int32)

    #print('df head\n',df.head())
    #print('df info\n',df.info())

    # Split into test and training datasets
    train_dataset = df.sample(frac=0.8, random_state=0)
    test_dataset = df.drop(train_dataset.index)

    return train_dataset, test_dataset