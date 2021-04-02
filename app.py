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
from scipy.stats import lognorm
from annotated_text import annotated_text

with open('params.json') as json_file:
    params = json.load(json_file)

st.set_page_config(page_title="Fairly", layout="wide")

# Show logo
f = open("logo.svg","r")
lines = f.readlines()
line_string=''.join(lines)
render_svg(line_string)

# Intro
st.write("Fairly is a tool to help tech workers living in Portugal know if they're being paid fairly. For more \
information on how the data was collected and the modelling formulation visit the [official repo](https://github.com/TSFelg/fairly).")

# Menu
col1, col2, col3= st.beta_columns((1,1,1.5))

status = col1.selectbox('What is your working status?', params["status_list"])
job = col1.selectbox('What is your job?', params["job_list"])
work_experience = col1.selectbox('How much work experience do you have?', params["work_experience_list"], 3)
education = col1.selectbox('What is your level of education?', params["education_list"], 5)
employer_type = col1.selectbox('Which of these best represents your employer?', params["employer_type_list"])

english_level = col2.selectbox('What is your english level?', params["english_level_list"], 2)
residence = col2.selectbox('Where in Portugal do you live?', params["residence_list"], 2)
company_country = col2.selectbox('Where is the company you work for located?', params["company_country_list"], 6)
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
mu  = np.log(y_dists.params["scale"]/1000)
s  = y_dists.params["s"]
avg = np.exp(mu + s**2/2)

x = np.linspace(0, 120, 1000)

salary = col3.slider('What is your annual gross salary?', 0, 160, 30)
pdf = lognorm.pdf(x, s,scale=np.exp(mu))
cdf = np.round(100*lognorm.cdf(salary, s,scale=np.exp(mu))[0],1)
#col3.write("- The average worker with your profile is payed {}€.".format(int(avg[0])))
#col3.write("- You are payed more than {}% of the people.".format(cdf))
col3.markdown("- The average worker with your profile is payed <font style='color:darkred'>{}€</font>".format(int(avg[0]*1000)) + 
"\n - You are payed more than <font style='color:darkorange'>{}%</font>".format(cdf) + " of the people", 
unsafe_allow_html=True)
#col3.markdown("- You are payed more than <font style='color:orange'>{}%</font>".format(cdf) + " of the people", unsafe_allow_html=True)

source = pd.DataFrame({'salary': x, 'pdf': pdf, 'avg_salary':salary, 'avg':avg[0]})
chart = alt.Chart(source).mark_line(color='orange').encode(x='salary',y='pdf')
rule = alt.Chart(source).mark_rule(color='#5DAB24', size=2).encode(x=alt.X('avg_salary', axis=alt.Axis(title="annual gross salary (k€)")))
rule2 = alt.Chart(source).mark_rule(color='indianred', size=2, strokeDash=[5,5]).encode(x='mean(avg)')

# Create a selection that chooses the nearest point & selects based on x-value
nearest = alt.selection(type='single', nearest=True, on='mouseover',
                        fields=['salary'], empty='none')

# Transparent selectors across the chart. This is what tells us
# the x-value of the cursor
selectors = alt.Chart(source).mark_point().encode(
    x='salary',
    opacity=alt.value(0),
).add_selection(
    nearest
)

# Draw points on the line, and highlight based on selection
points = chart.mark_point(color="green").encode(
    opacity=alt.condition(nearest, alt.value(1), alt.value(0))
)

# Draw text labels near the points, and highlight based on selection
text = chart.mark_text(align='left', dx=5, dy=-5).encode(
    text=alt.condition(nearest, 'pdf:Q', alt.value(' '))
)

chart = (chart + rule + rule2)
col3.altair_chart(chart, use_container_width=True)