"""This is to provide the link to the data and any pre-processing for it"""
import os
import pandas as pd


def get_data() -> pd.DataFrame:
    """This is to return the pandas data frame at the link

    Returns
    -------
    Pandas Data Frame
        The data from FreeCodeCamp"""
    wd = os.path.dirname(os.path.realpath(__file__))
    return pd.read_csv(os.path.join(wd, '..', 'data', 'insurance.csv'))


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the data for modelling

    Parameters
    ----------
    df: Pandas DataFrame
        The raw data to be encoded

    Returns
    -------
    Pandas DataFrame
        The encoded data"""
    df = df.copy()
    df['smoker'] = df['smoker'] == 'yes'
    df['female'] = df['sex'] == 'female'

    # region
    region_df = pd.get_dummies(df.region, drop_first=True)
    for col in region_df.columns:
        df[col] = region_df[col]
    del region_df

    # drop unneeded columns
    df.drop(['sex', 'region'], axis=1, inplace=True)

    # change data types
    for col in df.columns:
        df[col] = df[col].astype('int')
    return df


def split_train_test(df: pd.DataFrame):
    """Split the data into training and testing sets

    Parameters
    ----------
    df: Pandas DataFrame
        The encoded data

    Returns
        tuple(pd DataFrame, pd Series, pd DataFrame, pd Series)
            X_train, y_train, X_test, y_test"""
    X_train = df.sample(frac=0.8, random_state=0)
    y_train = X_train.expenses
    X_train.drop('expenses', axis=1, inplace=True)
    X_test = df.drop(X_train.index)
    y_test = X_test.expenses
    X_test.drop('expenses', axis=1, inplace=True)
    return X_train, y_train, X_test, y_test
