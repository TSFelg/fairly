import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from utils import clean_data, render_svg
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import altair as alt
from scipy.special import factorial
from scipy.stats import poisson, norm
import json

with open('params.json') as json_file:
    params = json.load(json_file)

st.set_page_config(page_title="Fairly", layout="wide")

# Show logo
f = open("logo.svg","r")
lines = f.readlines()
line_string=''.join(lines)
render_svg(line_string)

# Menu
col1, col2, col3= st.beta_columns((1,1,1.5))

status = col1.selectbox('What is your working status?', params["status_list"])
job = col1.selectbox('What is your job?', params["job_list"])
work_experience = col1.selectbox('How much work experience do you have?', params["work_experience_list"], 3)
education = col1.selectbox('What is your level of education?', params["education_list"], 5)
employer_type = col1.selectbox('Which of these best represents your employer?', params["employer_type_list"])

english_level = col2.selectbox('What is your english level?', params["english_level_list"], 2)
residence = col2.selectbox('Where in Portugal do you live?', params["residence_list"], 2)
company_country = col2.selectbox('Where is the company you work for located?', params["company_country_list"], 29)
employer_industry =  col2.selectbox('In what industry is your employer in?', params["employer_industry_list"])
employer_size = col2.selectbox('How many employees are in your company?', params["employer_size_list"], 3)

# Create df
df = pd.DataFrame([status, job, work_experience, english_level, residence, education, company_country, employer_industry, employer_type, employer_size], index = params["features"])
df = df.T
df = clean_data(df)

# Load model
with Path("data/model.p").open("rb") as f:
        model = pickle.load(f)

# Predict
y_dists = model.pred_dist(df.values)
mu  = np.round(y_dists.params["loc"])/1000
sigma = np.round(y_dists.params["scale"])/1000

x = np.linspace(0, 80, 1000)
pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))
#pdf = ((mu**x)*np.e**(-mu))/factorial(x)

salary = col3.slider('What is your annual gross salary?', 0, 140, 30)

col3.write("- The average worker with your profile is payed {}€.".format(int(1000*mu[0])))
col3.write("- You are payed more than {}% of the people.".format(round(100*norm.cdf(salary, mu, sigma)[0], 1)))

source = pd.DataFrame({'x': x, 'f(x)': pdf, 'salary':salary})
brush = alt.selection(type='interval')
chart = alt.Chart(source).mark_line(color='orange').encode(x='x',y='f(x)').add_selection(brush)
rule = alt.Chart(source).mark_rule(color='#5DAB24', size=2).encode(x='mean(salary)')
chart = (chart + rule)
col3.altair_chart(chart, use_container_width=True)