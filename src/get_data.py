"""This is to provide the link to the data and any pre-processing"""
import pandas as pd


def get_data() -> pd.DataFrame:
    """This is to return the pandas data frame at the link"""
    l = 'https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv'
    return pd.read_csv(l)
