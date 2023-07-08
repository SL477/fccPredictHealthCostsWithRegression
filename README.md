# Predict Health Costs With Regression

I made this model as part of the FreeCodeCamp challenges, but then updated it to use a scikit-learn model and made a Streamlit app.

![Screenshot of app](HealthCostPredictions.jpg)

## Docker

Build with:

```bash
docker image build -t healthcostpredictor .
```

Run with:

```bash
docker run -p 8501:8501 healthcostpredictor
```

## Get the data

Get the data by using:

```bash
make getdata
```

## Running

Run with:

```bash
streamlit run main.py
```
