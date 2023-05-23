#!/usr/bin/env python3
from datetime import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse

import sklearn
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import modelling as mod

class Stat:
    cnt = { 'mean': .0, "max": .0, "min": .0 }
    cas = { 'mean': .0, "max": .0, "min": .0 }
    reg = { 'mean': .0, "max": .0, "min": .0 }

    def set_cnt(self, mean, max, min):
        self.cnt['mean'] = mean
        self.cnt['max'] = max
        self.cnt['min'] = min

parser = argparse.ArgumentParser(
                    prog='analyze',
                    description='Analyze binking data')

parser.add_argument('-p', '--plot', dest='plot_flag', action='store_true')
parser.add_argument('-s', '--plot-seasonal-data', dest='seasonal_flag', action='store_true')
parser.add_argument('-d', '--describe', dest='describe_flag', action='store_true')
args = parser.parse_args()

output_dir = '../output/'

def plot_seasonal_data(data_hour):
    print('\thourly_customers.png...', end='')
    cols = ['casual_mean','registered_mean']
    stats_hourly = pd.DataFrame(columns=cols)
    stats_hourly_spring = pd.DataFrame(columns=cols)
    stats_hourly_summer = pd.DataFrame(columns=cols)
    stats_hourly_autumn = pd.DataFrame(columns=cols)
    stats_hourly_winter = pd.DataFrame(columns=cols)

    stats_hourly_seasons = [stats_hourly_spring, stats_hourly_summer,
                            stats_hourly_autumn, stats_hourly_winter]

    for i in range(0,24):
        hour = data_hour.where(data_hour['hr']==i)
        cas = hour.mean()['casual']
        reg = hour.mean()['registered']

        df = pd.DataFrame([[cas,reg]], columns=cols, index=[i])
        stats_hourly = stats_hourly.append(df)

        spring = hour.where(hour['season']==1)
        summer = hour.where(hour['season']==2)
        autumn = hour.where(hour['season']==3)
        winter = hour.where(hour['season']==4)

        for j,s in enumerate([spring,summer,autumn,winter]):
            cas = s.mean()['casual']
            reg = s.mean()['registered']
            df = pd.DataFrame([[cas,reg]], columns=cols, index=[i])
            stats_hourly_seasons[j] = stats_hourly_seasons[j].append(df)


            hourly_plot = stats_hourly.plot(kind='bar',stacked=True)
            hourly_plot.set_ylim(0,600)
            plt.savefig(output_dir + 'hourly_customers.png')
            plt.clf()
            plt.close()

    print('OK')

    print('\thourly_customers_season_i.png...', end='')
    for i,df in enumerate(stats_hourly_seasons):
        plt.clf()
        plot = df.plot(kind='bar',stacked=True)
        plot.set_ylim(0,600)
        plot.get_figure().savefig(output_dir + 'hourly_customers_season' +
                                  str(i+1) + '.png')

    print('OK')



data_day = pd.read_csv("../day.csv")
data_hour = pd.read_csv("../hour.csv")

data_day.drop("instant", axis = 1, inplace = True)
data_day.select_dtypes('number')

if args.describe_flag:
    print(data_day[['temp','hum','casual','registered']].describe())
    print('\nNull values:')
    nulls = data_day.isnull().sum().sort_values(ascending=False)/data_day.shape[0]
    print(nulls)
    print(data_hour[['temp','hum','casual','registered']].describe())
    print('\nNull values:')
    nulls = data_hour.isnull().sum().sort_values(ascending=False)/data_hour.shape[0]
    print(nulls)
    print()

data_spring = data_day.where(data_day["season"]==1)
data_summer = data_day.where(data_day["season"]==2)
data_autumn = data_day.where(data_day["season"]==3)
data_winter = data_day.where(data_day["season"]==4)


cols = ['cnt_min','cnt_mean','cnt_max',
        'casual_min','casual_mean','casual_max',
        'registered_min','registered_mean','registered_max']

stats_daily = pd.DataFrame(columns=cols)

for name,dt in [('spring',data_spring), ('summer',data_summer), ('autumn',data_autumn), ('winter',data_winter)]:
    measures = []
    for s in ['cnt','casual','registered']:
        measures.append(dt.min()[s])
        measures.append(dt.mean()[s])
        measures.append(dt.max()[s])

    df = pd.DataFrame([measures], columns=cols,
                     index=[name])

    stats_daily = stats_daily.append(df)

if args.plot_flag:
    print('Plotting...')
    sns.set_theme(style="whitegrid")

    means = stats_daily.loc[:,['casual_mean','registered_mean']]
    day_conditions = data_day.loc[:,['temp','atemp','hum','windspeed']]
    customers = data_day.loc[:,['casual','registered','cnt']]

    print('\tday_conditions.png...', end='')
    day_cond = sns.boxplot(data=day_conditions) # hum outlier at 0 interesting
    plt.savefig(output_dir + 'day_conditions.png')
    plt.clf()
    print('OK')

    print('\tcustomers.png...', end='')
    cust_plot = sns.boxplot(data=customers)
    plt.savefig(output_dir + 'customers.png')
    plt.clf()
    print('OK')

    print('\tstacked_means.png...', end='')
    stacked_means = means.plot(kind='bar',stacked=True)
    plt.tight_layout()
    plt.savefig(output_dir + 'stacked_means.png')
    plt.clf()
    print('OK')

    if args.seasonal_flag:
        plot_seasonal_data(data_hour)

    print('\tcnt_temp_scatter.png...', end='')
    plt.clf()
    splot = sns.scatterplot(data=data_day, x='cnt', y='temp', hue='weathersit')
    plt.savefig(output_dir + 'cnt_temp_scatter.png')
    print('OK')

    print('\tcnt_hum_scatter.png...', end='')
    plt.clf()
    splot = sns.scatterplot(data=data_day, x='cnt', y='hum', hue='weathersit')
    plt.savefig(output_dir + 'cnt_hum_scatter.png')
    print('OK')

    print('\tcnt_hum_scatter.png...', end='')
    plt.clf()
    splot = sns.scatterplot(data=data_day, x='cnt', y='windspeed', hue='weathersit')
    plt.savefig(output_dir + 'cnt_windspeed_scatter.png')
    print('OK')

    print('\theatmap.png...', end='')
    plt.clf()
    heat = sns.heatmap(data_day.corr(), linewidth=.5)
    plt.tight_layout()
    plt.savefig(output_dir + 'heatmap.png')
    plt.clf()
    print('OK')

    print('\tlmplot-...-.png...', end='')
    sns.lmplot(data=data_day, x='temp', y='cnt')
    plt.savefig(output_dir + 'lmplot-temp-cnt.png')
    plt.clf()

    sns.lmplot(data=data_day, x='hum', y='cnt')
    plt.savefig(output_dir + 'lmplot-hum-cnt.png')
    plt.clf()

    sns.lmplot(data=data_day, x='windspeed', y='cnt')
    plt.savefig(output_dir + 'lmplot-wind-cnt.png')
    plt.clf()

    sns.lmplot(data=data_day, x='weathersit', y='cnt', x_estimator=np.mean)
    plt.savefig(output_dir + 'lmplot-weathersit-cnt.png')
    plt.clf()
    print('OK')

    print('\tpairplot.png...', end='')
    sns.pairplot(data=data_day[['hum','temp','windspeed','cnt']], kind='kde')
    plt.savefig(output_dir + 'pairplot.png')
    plt.clf()
    print('OK')


    plt.close()
    print('Plotting completed\n')

### end plot if
print('Regression models:')
print('Daily results')
y_test, predictions = mod.train_test_models(data_day)

mod.plot_linearity_assumption(y_test, predictions['lm'], 'daily-lm', output_dir)
mod.plot_linearity_assumption(y_test, predictions['l'], 'daily-l', output_dir)
mod.plot_linearity_assumption(y_test, predictions['r'], 'daily-r', output_dir)
mod.plot_linearity_assumption(y_test, predictions['en'], 'daily-en', output_dir)
mod.plot_linearity_assumption(y_test, predictions['rf'], 'daily-rf', output_dir)

# plt.scatter(X_test['temp'], y_test, color="black")
# plt.plot(X_test['temp'], lm_fit.coef_, color = "blue", linewidth=.5)
# plt.plot(X_test['temp'], rf_predict, color = "red", linewidth=.5)
# plt.savefig(output_dir + 'results.png')
# plt.clf()

print('Optimized')
y_test, predictions = mod.train_test_models_optimized(data_day)

mod.plot_linearity_assumption(y_test, predictions['lm'], 'daily-lm-opt', output_dir)
mod.plot_linearity_assumption(y_test, predictions['rf'], 'daily-rf-opt', output_dir)
mod.plot_dist_predition(y_test, predictions['lm'], 'daily-lm-dist', output_dir)
mod.plot_dist_predition(y_test, predictions['rf'], 'daily-rf-dist', output_dir)

print('Hourly results')
y_test, predictions = mod.train_test_models(data_hour)

mod.plot_linearity_assumption(y_test, predictions['lm'], 'hourly-lm', output_dir)
mod.plot_linearity_assumption(y_test, predictions['l'], 'hourly-l', output_dir)
mod.plot_linearity_assumption(y_test, predictions['r'], 'hourly-r', output_dir)
mod.plot_linearity_assumption(y_test, predictions['en'], 'hourly-en', output_dir)
mod.plot_linearity_assumption(y_test, predictions['rf'], 'hourly-rf', output_dir)

# plt.scatter(X_test['temp'], y_test, color="black")
# plt.plot(X_test['temp'], lm_fit.coef_, color = "blue", linewidth=.5)
# plt.plot(X_test['temp'], rf_predict, color = "red", linewidth=.5)
# plt.savefig(output_dir + 'results.png')
# plt.clf()

print('Optimized')
y_test, predictions = mod.train_test_models_optimized(data_hour)

mod.plot_linearity_assumption(y_test, predictions['lm'], 'hourly-lm-opt', output_dir)
mod.plot_linearity_assumption(y_test, predictions['rf'], 'hourly-rf-opt', output_dir)
mod.plot_dist_predition(y_test, predictions['lm'], 'hourly-lm-dist', output_dir)
mod.plot_dist_predition(y_test, predictions['rf'], 'hourly-rf-dist', output_dir)
