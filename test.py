import argparse
from sys import argv
from typing import NamedTuple, List, Type, Iterable
import logging

import time

import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime
from dateutil import tz
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

default_figure_size = plt.rcParams.get('figure.figsize')

try:
    import psycopg2
    from psycopg2.extensions import quote_ident
except ImportError:
    raise ImportError("To use this command you must install the psycopg2-binary (or psycopg2) package")


args = None


import numpy as np
import time


plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    #plt.show()
    return axs

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def SVD(X , num_components):

    #X_meaned=(X-X.mean())/X.std()

    X_meaned = X - np.mean(X , axis = 0)

    cov_mat = np.cov(X_meaned , rowvar = False)

    U, S, Vt = np.linalg.svd(cov_mat, full_matrices=False)
    V = Vt.T
    # Sort the PCs by descending order of the singular values (i.e. by the proportion of total variance they explain)
    ind = np.argsort(S)[::-1]
    U, S, V = U[:, ind], S[ind], V[:, ind]

    eigenvector_subset = U[:,0:num_components]

    # flip the first component upside down
    eigenvector_subset[:,0] = -1*eigenvector_subset[:,0]

    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()

    return X_reduced

#https://stackoverflow.com/questions/51347398/need-to-save-pandas-correlation-highlighted-table-cmap-matplotlib-as-png-image
def heatmap(images,df):
    fig, ax = plt.subplots(figsize=(15, 15))
    fig.set_tight_layout(True)
    sns.heatmap(df.corr(method='pearson'), annot=True, fmt='.4f', 
                cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    images.append(base64.b64encode(figfile.getvalue()).decode('utf8'))
    plt.subplots(figsize=default_figure_size)
    plt.clf()

sqlToday = "SELECT * FROM device_data_logs1 where timestamp >= date_trunc('day', now() AT TIME ZONE 'PST') AT TIME ZONE 'PST' order by timestamp desc;"
sqlYesturday = "SELECT * FROM device_data_logs1 where timestamp >= date_trunc('day', (now() - interval '24h') AT TIME ZONE 'PST') AT TIME ZONE 'PST' AND timestamp <= date_trunc('day', now() AT TIME ZONE 'PST') AT TIME ZONE 'PST' order by timestamp desc;"

def graphsvd():
    global args
    with psycopg2.connect(args.database_url) as conn:
        sql = sqlToday
        df = pd.read_sql_query(sql, conn)
        df = df.drop('inverter_output_current', 1)
        df = df.drop('inverter_charge_current', 1)
        df = df.drop('inverter_buy_current', 1)
        df = df.drop('inverter_sell_current', 1)
        df = df.drop('inverter_operating_mode', 1)
        df = df.drop('aux_output_state', 1)
        df = df.drop('minimum_ac_input_voltage', 1)
        df = df.drop('maximum_ac_input_voltage', 1)
        df = df.drop('sell_status', 1)
        df = df.drop('output_kw', 1)
        df = df.drop('buy_kw', 1)
        df = df.drop('sell_kw', 1)
        df = df.drop('charge_kw', 1)
        df = df.drop('ac_couple_kw', 1)
        df = df.drop('cc1_charger_state', 1)
        df = df.drop('cc2_charger_state', 1)
        midnight=(datetime
             .now(tz.gettz('America/Tijuana'))
             .replace(hour=0, minute=0, second=0, microsecond=0)
             .astimezone(tz.tzutc()))
        dayPercentComplete = df.iloc[:,0].apply(lambda x: (x.to_pydatetime() - midnight).total_seconds() / 60 / 60 / 24)
        df['dayPercentComplete'] = dayPercentComplete
        df = df.drop('timestamp', 1)
        
        cc_watts = pd.DataFrame(df['cc1_watts'] + df['cc2_watts'], columns = ['cc_watts']) 
        #target == cc1_watts
        
        # current leaving looks like a loss so I will inverse it
        df["shunt_c_current"] = df["shunt_c_current"]*-1
        df["shunt_c_accumulated_kwh"] = df["shunt_c_accumulated_kwh"]*-1
        df["shunt_c_accumulated_ah"] = df["shunt_c_accumulated_ah"]*-1
        df = df.loc[:, (df != 0).any(axis=0)]


        df_scaled=(df-df.mean())/df.std()
        
        pca = PCA()
        X_pca = pca.fit_transform(df_scaled)
        component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        X_pca = pd.DataFrame(X_pca, columns=component_names)

        print(X_pca.head())
        plot_variance(pca)
        mi_scores = make_mi_scores(X_pca, cc_watts, discrete_features=False)
        print(mi_scores)
        idx = X_pca["PC3"].sort_values(ascending=False).index
        print(df.loc[idx])

def graphsvddata():
    global args
    with psycopg2.connect(args.database_url) as conn:
        sql = sqlToday
        df = pd.read_sql_query(sql, conn)
        df = df.drop('inverter_output_current', 1)
        df = df.drop('inverter_charge_current', 1)
        df = df.drop('inverter_buy_current', 1)
        df = df.drop('inverter_sell_current', 1)
        df = df.drop('inverter_operating_mode', 1)
        df = df.drop('aux_output_state', 1)
        df = df.drop('minimum_ac_input_voltage', 1)
        df = df.drop('maximum_ac_input_voltage', 1)
        df = df.drop('sell_status', 1)
        df = df.drop('output_kw', 1)
        df = df.drop('buy_kw', 1)
        df = df.drop('sell_kw', 1)
        df = df.drop('charge_kw', 1)
        df = df.drop('ac_couple_kw', 1)
        df = df.drop('cc1_charger_state', 1)
        df = df.drop('cc2_charger_state', 1)
        midnight=(datetime
             .now(tz.gettz('America/Tijuana'))
             .replace(hour=0, minute=0, second=0, microsecond=0)
             .astimezone(tz.tzutc()))
        hourOfDay = df.iloc[:,0].apply(lambda x: (x.to_pydatetime() - midnight))
        dayPercentComplete = df.iloc[:,0].apply(lambda x: (x.to_pydatetime() - midnight).total_seconds() / 60 / 60 / 24 )
        df['dayPercentComplete'] = dayPercentComplete
        df = df.drop('timestamp', 1)
        
        #target == cc1_watts + cc2_watts
        cc_watts = pd.DataFrame(df['cc1_watts'] + df['cc2_watts'], columns = ['cc_watts']) 

        # current leaving looks like a loss so I will inverse it
        df["shunt_c_current"] = df["shunt_c_current"]*-1
        df["shunt_c_accumulated_kwh"] = df["shunt_c_accumulated_kwh"]*-1
        df["shunt_c_accumulated_ah"] = df["shunt_c_accumulated_ah"]*-1
        df = df.loc[:, (df != 0).any(axis=0)]

        #Applying it to SVD function
        mat_reduced = SVD(df , 3)
        
        #Creating a Pandas DataFrame of reduced Dataset
        principal_df = pd.DataFrame(mat_reduced , columns = ['PC1','PC2','PC3'])
        
        #Concat it with cc_watts variable to create a complete Dataset
        principal_df = pd.concat([principal_df , pd.DataFrame(cc_watts)] , axis = 1)
        principal_df = pd.concat([principal_df , pd.DataFrame(hourOfDay)] , axis = 1)

def about():
    global args
    with psycopg2.connect(args.database_url) as conn:
        sql = sqlToday
        df = pd.read_sql_query(sql, conn)
        df = df.drop('inverter_output_current', 1)
        df = df.drop('inverter_charge_current', 1)
        df = df.drop('inverter_buy_current', 1)
        df = df.drop('inverter_sell_current', 1)
        df = df.drop('inverter_operating_mode', 1)
        df = df.drop('aux_output_state', 1)
        df = df.drop('minimum_ac_input_voltage', 1)
        df = df.drop('maximum_ac_input_voltage', 1)
        df = df.drop('sell_status', 1)
        df = df.drop('output_kw', 1)
        df = df.drop('buy_kw', 1)
        df = df.drop('sell_kw', 1)
        df = df.drop('charge_kw', 1)
        df = df.drop('ac_couple_kw', 1)
        df = df.drop('cc1_charger_state', 1)
        df = df.drop('cc2_charger_state', 1)
        midnight=(datetime
             .now(tz.gettz('America/Tijuana'))
             .replace(hour=0, minute=0, second=0, microsecond=0)
             .astimezone(tz.tzutc()))
        dayPercentComplete = df.iloc[:,0].apply(lambda x: (x.to_pydatetime() - midnight).total_seconds() / 60 / 60 / 24)
        #prepare the data
        df['dayPercentComplete'] = dayPercentComplete
        # move column to 2nd place
        col = df.pop("dayPercentComplete")
        df.insert(1, col.name, col)

        df = df.loc[:, (df != 0).any(axis=0)]

parser = argparse.ArgumentParser(description="Read all available data from the Mate3 controller")

parser.add_argument(
    "--database-url",
    dest="database_url",
    help="Postgres database URL",
    default="postgres://postgres@localhost/postgres",
)
args = parser.parse_args(argv[1:])
graphsvd()
