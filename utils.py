import base64
import pandas as pd
from pathlib import Path
import pickle
import streamlit as st

def process_data(df):
    df = df[['Employment_Status', 'Residence_District_Aggregated',
    'Work_Company_Country', 'Job_Role_Original', 'Employer_Org_Type', 'English_Level',
    'Education_Level', 'Working_Experience']]

    with Path("modelling/ordinal_encoder.p").open("rb") as f:
        ordinal_encoder = pickle.load(f)

    ordinal_features = ["English_Level", "Working_Experience"]
    df[ordinal_features] = ordinal_encoder.transform(df[ordinal_features])

    with Path("modelling/onehot_encoder.p").open("rb") as f:
        onehot_encoder = pickle.load(f)

    onehot_features = ["Employment_Status",
                        "Residence_District_Aggregated",
                        "Work_Company_Country",
                        "Job_Role_Original",         
                        "Employer_Org_Type",
                        "Education_Level"]

    transformed = onehot_encoder.transform(df[onehot_features])
    df_transformed = pd.DataFrame(transformed, columns=onehot_encoder.get_feature_names())
    df = df.drop(columns = onehot_features)
    df = pd.concat([df,df_transformed], axis=1)

    return df

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.sidebar.write(html, unsafe_allow_html=True)

def set_max_width():
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