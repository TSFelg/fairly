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
st.set_page_config(page_title="Fairly", layout="wide", page_icon=icon, initial_sidebar_state="expanded")

def _max_width_():
    max_width_str = f"max-width: 1200px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )
_max_width_()

# Show logo
f = open("resources/logo.svg","r")
lines = f.readlines()
line_string=''.join(lines)
render_svg(line_string)

# Intro
st.sidebar.write("Fairly is a tool to help tech workers living in Portugal know if they're being paid fairly. For more \
information on how the data was collected and the modelling formulation visit the [official repo](https://github.com/TSFelg/fairly).")

# Menu
col1 = col2 =  col3= st.sidebar

salary = col3.slider('What is your annual gross salary? This should include bonuses, meal and medical allowance, etc.', 0, 160, 30)
job = col1.selectbox('What is your job?', params["job_list"])
work_experience = col1.selectbox('How much work experience do you have?', params["work_experience_list"], 3)
education = col1.selectbox('What is your level of education?', params["education_list"], 5)
english_level = col1.selectbox('What is your english level?', params["english_level_list"], 2)
residence = col2.selectbox('Where in Portugal do you live?', params["residence_list"], 2)
status = col2.selectbox('What is your working status?', params["status_list"])
employer_type = col2.selectbox('Which of these best represents your employer?', params["employer_type_list"])
company_country = col2.selectbox('Where is the company you work for located?', params["company_country_list"], 6)


col4, col6= st.beta_columns((2,1.5))

col7, col8, col9, col10 = st.beta_columns((0.5,0.5,0.5,1.5))
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

x_max_arg = np.argwhere(cdf > 0.99)[0][0]
x_max = np.max([x[x_max_arg],salary])
x_ax_max = 100.0 if x_max<100 else x_max
y_ax_max = 0.08 if max(pdf)<0.07 else max(pdf)


source = pd.DataFrame({'salary': x, 'pdf': pdf, 'avg_salary':salary, 'avg':avg[0]})

chart = alt.Chart(source).transform_fold(['Conditional Distribution']).mark_line(color='orange', clip=True).encode(x = alt.X('salary', scale=alt.Scale(domain=[0, x_ax_max])), 
                                        y = alt.Y('pdf', scale=alt.Scale(domain=[0, y_ax_max])),color=alt.Color('key:N', scale=alt.Scale(range=['orange','indianred','#5DAB24',]),
                                        legend=alt.Legend(title=None, labelFontSize=15,symbolStrokeWidth=10, orient="top-right"))).properties(height=450)

rule = alt.Chart(source).transform_fold(['Your salary']).mark_rule(color='#5DAB24', size=2).encode(x=alt.X('avg_salary', axis=alt.Axis(title="annual gross salary (k€)")),color=alt.Color('key:N', 
                                        scale=alt.Scale(range=['indianred','#5DAB24',]),legend=alt.Legend(title=None)))

rule2 = alt.Chart(source).transform_fold(['Conditional Mean']).mark_rule(color='indianred', size=2, strokeDash=[5,5]).encode(x='mean(avg)', color=alt.Color('key:N', 
                                        scale=alt.Scale(range=['indianred']),legend=alt.Legend(title=None)))


st.header("Estimated Conditional Distribution")
col9, col10, col11 = st.beta_columns((1,1, 0.3))

col9.markdown("- The average worker with your profile is paid <font style='color:darkred'>{}€</font>.    ".format(int(avg[0]*1000)) + 
"\n - You are paid more than <font style='color:darkorange'>{}%</font>".format(treshold) + " of the population with your profile.", 
unsafe_allow_html=True)
#col3.markdown("<font style='color:yellowgreen'> **Results** </font>", unsafe_allow_html=True)
col3.markdown("\n")
chart = (chart + rule + rule2)
st.altair_chart(chart, use_container_width=True)


col10.write("- Consider uploading your data to continue improving the model. Please use your real salary to avoid polluting the dataset :)")
if col11.button("Upload"):
    df.to_sql('params', engine, if_exists='append', index=False)


