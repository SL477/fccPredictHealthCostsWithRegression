"""This submodule is used to run a single prediction from the given piece of
data"""
import pandas as pd


def infer(mdl, age: int, bmi: float, children: int, smoker: bool, gender: str,
          region: str) -> float:
    """Use the model to infer the expenses for the given patient

    Parameters
    ----------
    mdl: sklearn model
        The trained model

    age: int
        Age of the user

    bmi: float
        The BMI of the user

    children: int
        The number of children of the user

    smoker: bool
        Whether or not the patient smokes

    gender: str
        Male/Female

    region: str
        ['South West', 'South East', 'North West', 'North East']

    Returns
    -------
    float
        The amount of healthcare expenses"""
    df = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'smoker': [int(smoker)],
        'female': [gender == 'Female'],
        'northwest': [region == 'North West'],
        'southeast': [region == 'South East'],
        'southwest': [region == 'South West'],
    })
    return mdl.predict(df)[0]
