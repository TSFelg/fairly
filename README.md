
<img src="logo.svg">

Fairly is a tool to help tech workers living in Portugal know if they're being paid fairly. Fairly models the probabilistic distribution of the annual gross salary conditioned on several input features such as the job, years of experience and the company's country location. The app then allows new users to specify their charateristics and understand how they are positioned within the conditional distribution. 

The data used to train the model comes from the [Tech Careers Report 2021](https://wp.landing.jobs/techcareersreport2021/?utm_source=taikai&utm_medium=event-platform&utm_term=102909&utm_content=tech-careers-report-taikai&utm_campaign=tech-careers-report-2021) which gathered more than 3000 answers from tech workers in Portugal. Fairly is being developed in the context of the [Landing.Jobs Data Challenge](https://taikai.network/en/landingjobs/challenges/datachallenge) which aims 
to generate knowledge based on the [Tech Careers Report 2021](https://wp.landing.jobs/techcareersreport2021/?utm_source=taikai&utm_medium=event-platform&utm_term=102909&utm_content=tech-careers-report-taikai&utm_campaign=tech-careers-report-2021).

- **ML Stack:** Data processing and transformers are implemented using `numpy`, `pandas`, and `scikit-learn`. The models are developed using `ngboost` and `scipy`. Visualizations are built with `altair`.
- **Ops Stack:** The web app is developed using `streamlit` and the data collection uses `postgresql`. Both are hosted on `heroku`.
