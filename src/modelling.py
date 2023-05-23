from datetime import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse

import sklearn
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def train_test_models(dt):
    y = dt['cnt']
    X = dt.drop(['cnt','yr','season','registered','casual'], axis = 1)
    X['day'] = X['dteday'].apply(
        lambda time: datetime.strptime(time, '%Y-%m-%d').day)

    X = X.drop('dteday', axis = 1)
    y_test,predictions = training(X,y)
    print_results(y_test, predictions)
    return y_test,predictions

def train_test_models_optimized(dt):
    y = dt['cnt']
    X = dt.drop(['cnt','yr','season','registered','casual','atemp'], axis = 1)
    X['day'] = X['dteday'].apply(
        lambda time: datetime.strptime(time, '%Y-%m-%d').day)

    X = X.drop('dteday', axis = 1)

    weathersit = pd.get_dummies(X.weathersit, prefix='weathersit')
    X = pd.concat([X, weathersit], axis=1)
    X.drop(columns='weathersit', inplace=True)

    # weekday = pd.get_dummies(X.weekday, prefix='weekday')
    # X = pd.concat([X, weekday], axis=1)
    # X.drop(columns='weekday', inplace=True)

    # sc = StandardScaler()
    # numeric = ['day']
    # X[numeric] = sc.fit_transform(X[numeric])
    # print(X)

    y_test,predictions = training(X,y)
    print_results(y_test, predictions)
    return y_test,predictions

def training(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=111)

    lm = LinearRegression()
    lm_fit = lm.fit(X_train, y_train)
    lm_predict = lm_fit.predict(X_test)

    ridge = Ridge()
    ridge_fit = ridge.fit(X_train,y_train)
    r_predict = ridge_fit.predict(X_test)

    lasso = Lasso(max_iter=1000)
    lasso_fit = lasso.fit(X_train,y_train)
    l_predict = lasso_fit.predict(X_test)

    ENet = ElasticNet(l1_ratio=.8)
    ENet_fit = ENet.fit(X_train,y_train)
    en_predict = ENet_fit.predict(X_test)

    rand = RandomForestRegressor()
    rand_fit = rand.fit(X_train,y_train)
    rf_predict = rand_fit.predict(X_test)

    predictions = {
        'lm': lm_predict,
        'r': r_predict,
        'l': l_predict,
        'en': en_predict,
        'rf': rf_predict
    }
    return y_test,predictions


def print_results(y_test, predictions):
    print('\nMSE:')
    print('\tLinear Model:\t %d' % mean_squared_error(y_test,predictions['lm']))
    print('\tRidge Model:\t %d' % mean_squared_error(y_test,predictions['r']))
    print('\tLasso Model:\t %d' % mean_squared_error(y_test,predictions['l']))
    print('\tElastic Net:\t %d' % mean_squared_error(y_test,predictions['en']))
    print('\tRandom Forest:\t %d' % mean_squared_error(y_test,predictions['rf']))
    print('Coefficient of determination:')
    print('\tLinear Model:\t %.2f' % r2_score(y_test,predictions['lm']))
    print('\tRidge Model:\t %.2f' % r2_score(y_test,predictions['r']))
    print('\tLasso Model:\t %.2f' % r2_score(y_test,predictions['l']))
    print('\tElastic Net:\t %.2f' % r2_score(y_test,predictions['en']))
    print('\tRandom Forest:\t %.2f' % r2_score(y_test,predictions['rf']))


def plot_linearity_assumption(y_test, prediction, name, output_dir):
    sns.set_theme(style="whitegrid")
    df_lm = pd.DataFrame()
    df_lm['cnt_actual'] = y_test
    df_lm['cnt_predicted'] = prediction
    sns.lmplot(data=df_lm, x='cnt_actual', y='cnt_predicted', fit_reg=False)
    # Plotting the diagonal line
    line_coords = np.arange(df_lm[['cnt_actual', 'cnt_predicted']].min().min()-10,
                            df_lm[['cnt_actual', 'cnt_predicted']].max().max()+10)
    plt.plot(line_coords, line_coords,  # X and y points
             color='darkorange', linestyle='--')
    plt.savefig(output_dir + 'linearity_assumption-' + name + '.png')
    plt.clf()

def plot_dist_predition(y_test, prediction, name, output_dir):
    sns.set_theme(style="whitegrid")
    df = pd.DataFrame()
    df['cnt_actual'] = y_test
    df['cnt_predicted'] = prediction
    sns.displot(df, kind='kde')
    plt.savefig(output_dir + 'distribution-' + name + '.png')
    plt.clf()
