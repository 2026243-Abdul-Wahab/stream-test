import streamlit as st
import pandas as pd
import numpy as np

st.title("Uber Pickups in NYC")

DATE_COLUMNS="date/time"

DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMNS] = pd.to_datetime(data[DATE_COLUMNS])
    return data

data_load_state = st.text('Loading data...')
load_data(10000)
data_load_state.text('Done!')