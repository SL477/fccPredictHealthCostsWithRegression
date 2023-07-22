"""This is the streamlit app for experimentation"""
import streamlit as st
from joblib import load
import matplotlib.pyplot as plt
from src.get_data import get_data, preprocess_data
from src.infer import infer

# Update title and icon
st.set_page_config(
    page_title="Predict Health Costs With Regression",
    page_icon='https://link477.com/assets/images/link477.png'
)

st.title("Predict Health Costs With Regression")

# load the data
df = preprocess_data(get_data())
y = df.expenses
df.drop('expenses', axis=1, inplace=True)

# load the model
mdl = load('rf.joblib')

# Predict Health costs
age = st.sidebar.slider(
    'Enter your age',
    18.0,
    100.0,
    39.0,
    1.0
)

sex = st.sidebar.radio(
    'Gender:',
    ('Male', 'Female')
)

bmi = st.sidebar.slider(
    'BMI:',
    16.0,
    55.0,
    30.0,
    0.1
)

children = st.sidebar.number_input(
    'Number of Children:',
    0
)

smoker = st.sidebar.checkbox(
    'Smoker?'
)

region = st.sidebar.selectbox(
    'Region:',
    ['South West', 'South East', 'North West', 'North East']
)

# plot the predictions of the main data
fig, ax = plt.subplots()
preds = mdl.predict(df)
ax.scatter(preds, y)
ax.set_xlabel('True Values (expenses)')
ax.set_ylabel('Predictions (expenses)')
ax.set_title('Predicted Expenses vs Actual Expenses')
lims = [0, 5000]
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.axline((0, 0), slope=1)
st.pyplot(fig)

# stats
rmse = (sum((preds - y) ** 2) / len(y)) ** 0.5
st.write('RMSE: ${:.2f}'.format(rmse))
mae = sum(abs(preds - y)) / len(y)
st.write('MAE: ${:.2f}'.format(mae))

# Predict their one
prediction = infer(mdl, age, bmi, children, smoker, sex, region)
st.write("Your predicted health costs are ${:,.2f}.".format(prediction))
