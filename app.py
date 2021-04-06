import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from utils import process_data, render_svg
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import altair as alt
from scipy.special import factorial
from scipy.stats import poisson, norm
import json
from scipy.stats import lognorm
import os
from sqlalchemy import create_engine
import datetime

import time
start_time = time.time()

with open('params.json') as json_file:
    params = json.load(json_file)

engine = create_engine(os.getenv("DATABASE_URL"))

icon = Image.open("resources/coin.png")
st.set_page_config(page_title="Fairly", layout="wide", page_icon=icon)

# Show logo
f = open("resources/logo.svg","r")
lines = f.readlines()
line_string=''.join(lines)
render_svg(line_string)

# Intro
st.write("Fairly is a tool to help tech workers living in Portugal know if they're being paid fairly. For more \
information on how the data was collected and the modelling formulation visit the [official repo](https://github.com/TSFelg/fairly).")

# Menu
col1, col2, col3= st.beta_columns((1,1,1.5))

job = col1.selectbox('What is your job?', params["job_list"])
work_experience = col1.selectbox('How much work experience do you have?', params["work_experience_list"], 3)
education = col1.selectbox('What is your level of education?', params["education_list"], 5)
english_level = col1.selectbox('What is your english level?', params["english_level_list"], 2)
residence = col2.selectbox('Where in Portugal do you live?', params["residence_list"], 2)
status = col2.selectbox('What is your working status?', params["status_list"])
employer_type = col2.selectbox('Which of these best represents your employer?', params["employer_type_list"])
company_country = col2.selectbox('Where is the company you work for located?', params["company_country_list"], 6)
salary = col1.slider('What is your annual gross salary?', 0, 200, 30)

# Create df
df = pd.DataFrame([status, job, work_experience, english_level, residence, education, company_country, employer_type], index = params["features"])
df = df.T
df_processed = process_data(df)
df["time"] =  datetime.datetime.now()
df["Avg_Salary"] = salary

@st.cache(allow_output_mutation=True)
def load_model():
    with Path("modelling/model.p").open("rb") as f:
        model = pickle.load(f)
    return model 

# Load model
model = load_model()

# Predict
y_dists = model.pred_dist(df_processed.values)
mu  = np.log(y_dists.params["scale"]/1000)
s  = y_dists.params["s"]
avg = np.exp(mu + s**2/2)

x = np.linspace(0, 500, 1000)

pdf = lognorm.pdf(x, s,scale=np.exp(mu))
cdf = lognorm.cdf(x, s,scale=np.exp(mu))
treshold = np.round(100*lognorm.cdf(salary, s,scale=np.exp(mu))[0],1)

x_max_arg = np.argwhere(cdf > 0.992)[0][0]
x_max = np.max([x[x_max_arg],salary])
x_ax_max = 100.0 if x_max<100 else x_max
y_ax_max = 0.08 if max(pdf)<0.07 else max(pdf)

source = pd.DataFrame({'salary': x, 'pdf': pdf, 'avg_salary':salary, 'avg':avg[0]})
chart = alt.Chart(source).mark_line(color='orange', clip=True).encode(x = alt.X('salary', scale=alt.Scale(domain=[0, x_ax_max])), 
                                                           y = alt.Y('pdf', scale=alt.Scale(domain=[0, y_ax_max]))
                                                           ).properties(height=350)

rule = alt.Chart(source).mark_rule(color='#5DAB24', size=2).encode(x=alt.X('avg_salary', axis=alt.Axis(title="annual gross salary (k€)")))
rule2 = alt.Chart(source).mark_rule(color='indianred', size=2, strokeDash=[5,5]).encode(x='mean(avg)')


#col3.markdown("<font style='color:yellowgreen'> **Results** </font>", unsafe_allow_html=True)
col3.markdown("\n")
chart = (chart + rule + rule2)
col3.altair_chart(chart, use_container_width=True)

col3.markdown("&nbsp; The average worker with your profile is paid <font style='color:darkred'>{}€</font>  ".format(int(avg[0]*1000)) + 
"\n &nbsp; You are paid more than <font style='color:darkorange'>{}%</font>".format(treshold) + " of the people", 
unsafe_allow_html=True)

if col2.button("Please consider uploading your data to continue improving the model"):
    df.to_sql('params', engine, if_exists='append', index=False)
