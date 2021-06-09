import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from datetime import date, timedelta, datetime
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint
import numpy as np

mpl.rcParams['pdf.fonttype'] = 42
today = datetime.today()

datapath = '../data/'


def plot_number_tested_timeseries(adf) :

    cols = ['num_delivered', 'num_tested']
    df = adf.groupby('delivery_date')[cols].agg(np.sum).reset_index()
    df = df.sort_values(by='delivery_date')
    df['excess_delivered'] = df['num_delivered'] - df['num_tested']

    sns.set_style('whitegrid', {'axes.linewidth' : 0.5})
    fig = plt.figure('Fig 1: deliveries tested and not tested', figsize=(8,2))
    ax = fig.gca()
    formatter = mdates.DateFormatter("%m-%d")

    ax.bar(df['delivery_date'], df['num_tested'], color='#808080',
           align='center', linewidth=0, label='tested for SARS-CoV-2')
    ax.bar(df['delivery_date'], df['excess_delivered'], bottom=df['num_tested'], color='k',
           align='center', linewidth=0, label='not tested for SARS-CoV-2')
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.legend()
    ax.set_ylabel('number of individuals')
    ax.set_xlabel('date of delivery')


def plot_tpr(adf) :

    cols = ['num_positive', 'num_tested']
    df = adf[adf['covid_region'] == 11].groupby('delivery_date')[cols].agg(np.sum).reset_index()
    df = df.sort_values(by='delivery_date')
    df = df[df['num_tested'] > 0]
    df['fraction_positive'] = df['num_positive']/df['num_tested']

    sns.set_style('whitegrid', {'axes.linewidth' : 0.5})
    palette = sns.color_palette('Set1')
    fig = plt.figure('Fig 2: TPR and CLI timeseries', figsize=(8,4))
    ax = fig.add_subplot(2,1,1)
    formatter = mdates.DateFormatter("%m-%d")

    df['moving_ave_frac'] = df['fraction_positive'].rolling(window=7, center=True).mean()
    ax.plot(df['delivery_date'], df['moving_ave_frac'], '-k', label='L&D TPR')

    df['moving_ave_test'] = df['num_tested'].rolling(window=7, center=True).sum()
    df['moving_ave_pos'] = df['num_positive'].rolling(window=7, center=True).sum()
    lows, highs = [], []
    for r, row in df.iterrows() :
        low, high = proportion_confint(row['moving_ave_pos'], row['moving_ave_test'])
        lows.append(low)
        highs.append(high)

    ax.fill_between(df['delivery_date'].values, lows, highs, color='k', linewidth=0, alpha=0.3)

    ax.set_ylabel('L&D percent positive')
    ax.set_title('Chicago')

    adf = pd.read_csv(os.path.join(datapath, '210413_region11_tpr.csv'))
    adf['date'] = pd.to_datetime(adf['date'])

    adf['moving_ave_frac'] = adf['tpr'].rolling(window=7, center=True).mean()
    ax.plot(adf['date'], adf['moving_ave_frac'], '--k', label='general population TPR')
    adf['moving_ave_test'] = adf['total_specs'].rolling(window=7, center=True).sum()
    adf['moving_ave_pos'] = adf['positive_specs'].rolling(window=7, center=True).sum()

    ax.set_ylim(0, 0.18)
    ax.set_xlim(date(2020, 6, 10), date(2021, 1, 15))
    ax.set_ylabel('test positivity rate')
    ax.legend()
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(mdates.MonthLocator())

    ax = fig.add_subplot(2,1,2)
    cdf = pd.read_csv(os.path.join(datapath, 'CLI_admissions.csv'))
    cdf = cdf[cdf['region'] == 'Chicago']
    cdf['date'] = pd.to_datetime(cdf['date'])
    cdf = cdf[(cdf['date'] >= np.min(df['delivery_date']) - timedelta(days=3)) & (cdf['date'] <= np.max(df['delivery_date'] + timedelta(days=3)))]
    cdf = cdf.groupby('date')['inpatient'].agg(np.sum).reset_index()
    cdf = cdf.sort_values(by='date')
    cdf['moving_ave_cli'] = cdf['inpatient'].rolling(window=7, center=True).mean()
    ax.plot(cdf['date'], cdf['moving_ave_cli'], '-', color=palette[1])
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.set_ylabel('CLI admissions')
    ax.set_xlabel('date')
    ax.set_ylim(0,)
    ax.set_xlim(date(2020, 6, 10), date(2021, 1, 15))


def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))


def plot_lag(adf) :

    cols = ['num_positive', 'num_tested']
    df = adf[adf['covid_region'] == 11].groupby('delivery_date')[cols].agg(np.sum).reset_index()
    df = df.sort_values(by='delivery_date')
    df = df[df['num_tested'] > 0]
    df['fraction_positive'] = df['num_positive']/df['num_tested']

    sns.set_style('whitegrid', {'axes.linewidth' : 0.5})
    df['moving_ave_frac'] = df['fraction_positive'].rolling(window=7, center=True).mean()

    cdf = pd.read_csv(os.path.join(datapath, 'CLI_admissions.csv'))
    cdf = cdf[cdf['region'] == 'Chicago']
    cdf['date'] = pd.to_datetime(cdf['date'])
    cdf = cdf[(cdf['date'] >= np.min(df['delivery_date']) - timedelta(days=3)) & (cdf['date'] <= np.max(df['delivery_date'] + timedelta(days=3)))]
    cdf = cdf.groupby('date')['inpatient'].agg(np.sum).reset_index()
    cdf = cdf.sort_values(by='date')
    cdf['moving_ave_cli'] = cdf['inpatient'].rolling(window=7, center=True).mean()

    df = pd.merge(left=df[['delivery_date', 'moving_ave_frac']],
                  right=cdf[['date', 'moving_ave_cli']],
                  left_on='delivery_date', right_on='date', how='inner')

    cdf = pd.read_csv(os.path.join(datapath, '210413_region11_tpr.csv'))
    cdf['date'] = pd.to_datetime(cdf['date'])
    cdf = cdf[(cdf['date'] >= np.min(df['delivery_date'])) & (cdf['date'] <= np.max(df['delivery_date']))]
    cdf = cdf.sort_values(by='date')
    cdf = cdf.rename(columns={'moving_ave_frac' : 'moving_ave_tpr'})

    df = pd.merge(left=df[['delivery_date', 'moving_ave_frac', 'moving_ave_cli']],
                  right=cdf[['date', 'moving_ave_tpr']],
                  left_on='delivery_date', right_on='date', how='inner')

    lags = range(-30,20)

    fig = plt.figure('Fig 3: cross correlations', figsize=(9,3))
    fig.subplots_adjust(bottom=0.15, wspace=0.3, left=0.1, right=0.97)
    ax = fig.add_subplot(1,2,1)

    rs = [crosscorr(df['moving_ave_cli'], df['moving_ave_frac'], lag) for lag in lags]
    ax.plot(lags, rs)
    ax.set_ylim(0.35, 0.8)
    ax.set_xlabel('Number of days TPR at labor and delivery leads CLI admissions')
    ax.set_ylabel('cross correlation (Pearson)')

    ax = fig.add_subplot(1,2,2)
    rs = [crosscorr(df['moving_ave_tpr'], df['moving_ave_frac'], lag) for lag in lags]
    ax.plot(lags, rs)
    ax.set_ylim(0.35, 0.8)
    ax.set_xlabel('Number of days TPR at labor and delivery leads general TPR')
    ax.set_ylabel('cross correlation (Pearson)')


def plot_fraction_asymptomatic(adf) :

    cols = ['num_positive', 'num_asymptomatic']
    df = adf.groupby('delivery_date')[cols].agg(np.sum).reset_index()
    df = df.sort_values(by='delivery_date')
    df = df[df['num_positive'] > 0]
    df['fraction_asymptomatic'] = df['num_asymptomatic']/df['num_positive']

    sns.set_style('whitegrid', {'axes.linewidth' : 0.5})
    fig = plt.figure('Extra: fraction asymptomatic', figsize=(8,2))
    ax = fig.gca()
    palette = sns.color_palette('Set1')
    formatter = mdates.DateFormatter("%m-%d")

    df['moving_ave_frac'] = df['fraction_asymptomatic'].rolling(window=7, center=True).mean()
    ax.plot(df['delivery_date'], df['moving_ave_frac'], '-', color=palette[1])

    df['moving_ave_asymp'] = df['num_asymptomatic'].rolling(window=7, center=True).sum()
    df['moving_ave_pos'] = df['num_positive'].rolling(window=7, center=True).sum()
    lows, highs = [], []
    for r, row in df.iterrows() :
        low, high = proportion_confint(row['moving_ave_asymp'], row['moving_ave_pos'])
        lows.append(low)
        highs.append(high)

    ax.fill_between(df['delivery_date'].values, lows, highs, color=palette[1], linewidth=0, alpha=0.3)

    ax.set_ylabel('percent of positives who were asymptomatic at time of test')
    ax.set_xlabel('date of delivery')
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(mdates.MonthLocator())


if __name__ == '__main__' :

    adf = pd.read_csv(os.path.join(datapath, '210324_labor_delivery_clean.csv'))
    adf['delivery_date'] = pd.to_datetime(adf['delivery_date'])

    plot_number_tested_timeseries(adf)
    plot_tpr(adf)
    plot_lag(adf)
    plot_fraction_asymptomatic(adf)
    plt.show()
