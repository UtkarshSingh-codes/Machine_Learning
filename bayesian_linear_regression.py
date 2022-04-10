
%matplotlib inline
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from scipy import stats

import aesara
import pymc3 as pm

plt.style.use("seaborn-darkgrid")
print(f"Running on PyMC3 v{pm.__version__}")


np.random.seed(16)

# Parameters
size = 1000
beta_0, beta_1 = 1, 2

# Simulate outcome variable
x = np.random.random(size)
y = beta_0 + beta_1 * x + np.random.randn(size)
data = pd.DataFrame(dict(x=x, y=y))

X = sm.add_constant(data['x'])
Y = data['y']
reg = sm.OLS(Y, X)

model_freq = reg.fit()
print(model_freq.summary())

beta_0_ols, beta_1_ols = model_freq.params
print(f'Intercept: {beta_0_ols}')
print(f'Slope: {beta_1_ols}')

# Given your data, how your coefficients is going to be distributed
with pm.Model() as model:
    intercept = pm.Normal('Intercept', mu=0, sd=20)
    slope = pm.HalfCauchy('x', beta=10, testval=1.0)
    sigma = pm.HalfCauchy('sigma', beta=10, testval=1.0)
    
    # Estimate of mean
    mu = intercept + slope * x
    likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)
    
    # Sampler
    step = pm.NUTS()

    # Posterior distribution
    trace = pm.sample(10000, step)
pm.plot_posterior(trace, figsize = (12, 5));


no_data = 500
with pm.Model() as model:
    intercept = pm.Normal('Intercept', mu=0, sd=20)
    slope = pm.HalfCauchy('x', beta=10, testval=1.0)
    sigma = pm.HalfCauchy('sigma', beta=10, testval=1.0)
    
    # Estimate of mean
    np.random.seed(16)
    pos_random = np.random.choice(np.arange(len(x)), size=500)
    x_sample = x[pos_random]
    y_sample = y[pos_random]
    mu = intercept + slope * x_sample
    likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y_sample)
    
    # Sampler
    step = pm.NUTS()
pm.plot_posterior(trace_500, figsize = (12, 5));


x_pred = 0.5
bayes_prediction_500 = trace_500['Intercept'] + trace_500['x'] * x_pred
bayes_prediction = trace['Intercept'] + trace['x'] * x_pred

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.kdeplot(bayes_prediction_500, label = 'Bayes Posterior Prediction', ax=axs[0])
axs[0].vlines(1+2*x_pred, ymin = 0, ymax = 8.5, label = 'OLS Prediction', colors = 'red', linestyles='--')
axs[0].set_xlim(1.8, 2.2)
axs[0].set_ylim(0, 13)
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].legend
sns.kdeplot(bayes_prediction, label = 'Bayes Posterior Prediction', ax=axs[1])
axs[1].vlines(1+2*x_pred, ymin = 0, ymax = 9.5, label = 'OLS Prediction', colors = 'red', linestyles='--')
axs[1].set_xlim(1.8, 2.2)
axs[1].set_ylim(0, 13)
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].legend

# Initialize random number generator
np.random.seed(16)

# True parameter values
beta0_true = 5
beta1_true = 7
beta2_true = 13

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.5

# Simulate outcome variable
Y = beta0_true + beta1_true * X1 + beta2_true * X2 + np.random.randn(size)


basic_model = pm.Model()
with basic_model:
    # Priors for unknown model parameters
    beta0 = pm.Normal("beta0", mu=0, sigma=1)
    beta1 = pm.Normal("beta1", mu=12, sigma=1)
    beta2 = pm.Normal("beta2", mu=18, sigma=1)

    # Expected value of outcome
    mu = beta0 + beta1 * X1 + beta2 * X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=1, observed=Y)

    # draw 1000 posterior samples
    trace = pm.sample(1000)

    
def from_posterior(param, samples):
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)

    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    return pm.distributions.Interpolated(param, x, y)
    
traces = [trace]
for _ in range(10):
    # generate more data
    X1 = np.random.randn(size)
    X2 = np.random.randn(size) * 0.2
    Y = beta0_true + beta1_true * X1 + beta2_true * X2 + np.random.randn(size)

    model = pm.Model()
    with model:
        # Priors are posteriors from previous iteration
        beta0 = from_posterior("beta0", trace["beta0"])
        beta1 = from_posterior("beta1", trace["beta1"])
        beta2 = from_posterior("beta2", trace["beta2"])

        mu = beta0 + beta1 * X1 + beta2 * X2
        Y_obs = pm.Normal("Y_obs", mu=mu, sigma=1, observed=Y)

        trace = pm.sample(1000)
        traces.append(trace)