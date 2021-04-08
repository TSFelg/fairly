
<img src="resources/logo.svg">

Fairly is a tool to help tech workers living in Portugal know if they're being paid fairly. Fairly models the probabilistic distribution of the annual gross salary conditioned on several input features such as the job, years of experience and the company's country location. The app then allows new users to specify their charateristics and understand how they are positioned within the conditional distribution. Fairly is being developed in the context of the [Landing.Jobs Data Challenge](https://taikai.network/en/landingjobs/challenges/datachallenge) which aims 
to generate knowledge based on the [Tech Careers Report 2021](https://wp.landing.jobs/techcareersreport2021/?utm_source=taikai&utm_medium=event-platform&utm_term=102909&utm_content=tech-careers-report-taikai&utm_campaign=tech-careers-report-2021).

# Stack
- **ML Stack:** Data processing and transformers are implemented using `numpy`, `pandas`, and `scikit-learn`. The models are developed using `ngboost` and `scipy`. Model analysis is done using `shap` and the visualizations are built with `altair`.
- **Ops Stack:** The web app is developed using `streamlit` and the data collection uses `postgresql`. Both are hosted on `heroku`.

# Data

The data used in this project comes from the [Tech Careers Report 2021](https://wp.landing.jobs/techcareersreport2021/?utm_source=taikai&utm_medium=event-platform&utm_term=102909&utm_content=tech-careers-report-taikai&utm_campaign=tech-careers-report-2021) which gathered more than 3000 answers from tech workers in Portugal. The report collected more than 100 variables but not all of these are relevant for answering the question: "Are you paid fairly?". Adding to this the fact that when the app is deployed the objective is for it to be easy for users to get their answer, this makes it important to find the right balance between data collection exaustivness and model performance. Given this, there are three main reasons why features were discarded:

- **Ethical:** Gender, age and citizenship were disconsidered since in my definition of fairness payment should be equal regardless of these.
- **Non-causality:** These are features that may or may not improve model performance but have no causal relation to determine if the user is paid fairly. Examples include the feature: "Are you thinking about changing jobs in the next 12 months?". It may be that this feature helps predict the salary, as in underpaid workers may be more keen to change jobs or perhaps overpaid workers are more ambitious. But here the causal relationship is opposite, it's the salary that is affecting the feature, and so it doesn't help us answer if the worker is paid fairly or not. Other examples include questions like what most motivates workers and the importance they give to specific job perks.
- **Non-relevance:** Opposite to the previous, certain features may have a plausible causal relation with our question but the data shows they are not relevant. Examples include the size of the company and the languages and frameworks the worker knows. These features may not be relevant because the information they provide is already contained in other features. For example, the size of the company may also be explained already by the organisation type (start-up, scale-up, corporate, etc.) while the languages and frameworks should be strongly correlated with the job type (Data Scientist, Backend Dev, etc.).

After this selection the final features used to train the model are:
`Working Experience`, `English Level`, `Residence District`, `Education Level`, `Company Country`, `Company Type`, `Employment Status`, `Job Role`.
The first two were ordinally encoded and the latter were one-hot encoded.

# Modelling
Fairly models the probabilistic distribution of the gross annual salary conditioned on the input features. With a deterministic model we could only say if the user is paid more or less than the average of the population with its profile. But given the natural variance in the data this wouldn't be very informative, you may be paid more or less because of some variables we're not conditioning on or simply due to the aleatoric uncertainty in the data. But by modelling the full conditional distribution we enable questions such as: "Is my salary lower than 20% of the population with the same profile?". These are empowering questions that can help users know if they're paid fairly.

To model the conditional distribution we used [ngboost](https://stanfordmlgroup.github.io/projects/ngboost/). This model takes the high performance of gradient boosting algorithms coupled with natural gradients to learn multi-parameter distributions. The chosen distributions to model were the `LogNormal`, `Normal`, and `Laplace`. For each of these distributions the models were trained using grid search with cross-validation to explore the hyper-parameter space of the ngboost models. The results of the best model according to the negative log likelihood (NLL) were chosen and tested for the mean absolute error (MAE) and the root mean squared error (RMSE) as well. As can be seen on the table below the LogNormal distribution obtained the best results not only in terms of the NLL, which was expected given that it is the distribution that best fits the target distrbution, but also on the deterministic scores. This highlights the idea that calibrated probabilistic models can also be of interest for deterministic tasks. 


|      | LogNormal | Normal | Laplace |
|------|-----------|--------|---------|
| NLL  | **10.66**     | 10.88  | 10.76   |
| MAE  | **9474**     | 9644  | 9509   |
| RMSE | **13735**     | 13844  | 14043   |
