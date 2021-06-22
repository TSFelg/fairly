import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from utils import process_data, render_svg, set_max_width
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

import streamlit.components.v1 as components  # Import Streamlit

# Configs
with open('params.json') as json_file:
    params = json.load(json_file)

engine = create_engine(os.getenv("DATABASE_URL"))
icon = Image.open("resources/coin.png")
st.set_page_config(page_title="Fairly", layout="wide", page_icon=icon, initial_sidebar_state="expanded")
set_max_width()

# Show logo
f = open("resources/logo.svg","r")
lines = f.readlines()
line_string=''.join(lines)
render_svg(line_string)

with st.sidebar:
    st.write("[![Star](https://img.shields.io/github/stars/tsfelg/fairly.svg?logo=github&style=social)](https://gitHub.com/tsfelg/fairly) [![Follow](https://img.shields.io/twitter/follow/TSFelg?style=social)](https://www.twitter.com/TSFelg)")

# Intro
st.sidebar.write("Fairly is a tool to help tech workers living in Portugal know if they're being paid fairly.")

# Menu
salary = st.sidebar.slider('What is your annual gross salary? This should include bonuses, meal and medical allowance, etc.', 0, 160, 30)
status = st.sidebar.selectbox("What is your working status? If you're a contractor/freelancer adjust the salary to the equivalent of 1600 hours/year.", params["status_list"])
job = st.sidebar.selectbox('What is your job?', params["job_list"])
work_experience = st.sidebar.selectbox('How much work experience do you have?', params["work_experience_list"], 3)
education = st.sidebar.selectbox('What is your level of education?', params["education_list"], 5)
english_level = st.sidebar.selectbox('What is your english level?', params["english_level_list"], 2)
residence = st.sidebar.selectbox('Where in Portugal do you live?', params["residence_list"], 2)
employer_type = st.sidebar.selectbox('Which of these best represents your employer?', params["employer_type_list"])
company_country = st.sidebar.selectbox('Where is the company you work for located?', params["company_country_list"], 6)

# Create df
df = pd.DataFrame([status, job, work_experience, english_level, residence, education, company_country, employer_type], index = params["features"])
df = df.T
df_processed = process_data(df)
df["time"] =  datetime.datetime.now()
df["Avg_Salary"] = salary

# Load model
@st.cache(allow_output_mutation=True)
def load_model():
    with Path("modelling/model.p").open("rb") as f:
        model = pickle.load(f)
    return model 
model = load_model()

# Predict
y_dists = model.pred_dist(df_processed.values)
mu  = np.log(y_dists.params["scale"]/1000)
s  = y_dists.params["s"]
conditional_mean = np.exp(mu + s**2/2)[0]

# Calculate pdf, cdf and treshold
x = np.linspace(0, 500, 1000)
pdf = lognorm.pdf(x, s,scale=np.exp(mu))
cdf = lognorm.cdf(x, s,scale=np.exp(mu))
treshold = np.round(100*lognorm.cdf(salary, s,scale=np.exp(mu))[0],1)

# Axes tricks
x_max_arg = np.argwhere(cdf > 0.99)[0][0]
x_max = np.max([x[x_max_arg],salary])
x_ax_max = 100.0 if x_max<100 else x_max
y_ax_max = 0.08 if max(pdf)<0.07 else max(pdf)

#Results
st.header("Estimated Conditional Distribution")

col1, col2, col3 = st.beta_columns((1,1, 0.3))

col1.markdown("- The average worker with your profile is paid <font style='color:darkred'>{}€</font>.    ".format(int(conditional_mean*1000)) + 
"\n - You are paid more than <font style='color:darkorange'>{}%</font>".format(treshold) + " of the population with your profile.", 
unsafe_allow_html=True)


col2.write("- Consider uploading your data to continue improving the model. Please use your real salary to avoid polluting the dataset :)")
if col3.button("Upload"):
    df.to_sql('params', engine, if_exists='append', index=False)
    st.warning("Thanks for sharing your anonymous data! Consider leaving your feedback [here](https://forms.gle/M8oBHAASBbvEaTq57)")

# Plotting
source = pd.DataFrame({'salary': x, 'pdf': pdf, 'user_salary':salary, 'conditional_mean':conditional_mean})

chart = alt.Chart(source).transform_fold(['Conditional Distribution']).mark_line(color='orange', clip=True).encode(x = alt.X('salary', scale=alt.Scale(domain=[0, x_ax_max])), 
                                        y = alt.Y('pdf', scale=alt.Scale(domain=[0, y_ax_max])),color=alt.Color('key:N', scale=alt.Scale(range=['orange','indianred','#5DAB24',]),
                                        legend=alt.Legend(title=None, labelFontSize=15,symbolStrokeWidth=10, orient="top-right"))).properties(height=450)

rule = alt.Chart(source).transform_fold(['Your salary']).mark_rule(color='#5DAB24', size=2).encode(x=alt.X('user_salary', axis=alt.Axis(title="annual gross salary (k€)")),color=alt.Color('key:N', 
                                        scale=alt.Scale(range=['indianred','#5DAB24',]),legend=alt.Legend(title=None)))

rule2 = alt.Chart(source).transform_fold(['Conditional Mean']).mark_rule(color='indianred', size=2, strokeDash=[5,5]).encode(x='mean(conditional_mean)', color=alt.Color('key:N', 
                                        scale=alt.Scale(range=['indianred']),legend=alt.Legend(title=None)))

chart = (chart + rule + rule2)
st.altair_chart(chart, use_container_width=True)