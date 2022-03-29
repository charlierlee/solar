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

default_figure_size = plt.rcParams.get('figure.figsize')

try:
    import psycopg2
    from psycopg2.extensions import quote_ident
except ImportError:
    raise ImportError("To use this command you must install the psycopg2-binary (or psycopg2) package")


args = None


import numpy as np
import time

def PCA(X , num_components):
     
    #Step-1
    X_meaned = X - np.mean(X , axis = 0)
     
    #Step-2
    cov_mat = np.cov(X_meaned , rowvar = False)
     
    #Step-3
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    #Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    #Step-5
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
    
    #Step-6
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
    
    return X_reduced

def SVD(X , num_components):

    #mu=(X-X.mean())/X.std()
    X = X.to_numpy()
    mu = np.mean(X , axis = 0)
    ma_X = X - mu

    cov_mat = np.cov(ma_X , rowvar = False)

    U, S, Vt = np.linalg.svd(cov_mat, full_matrices=False)
    V = Vt.T
    # Sort the PCs by descending order of the singular values (i.e. by the proportion of total variance they explain)
    ind = np.argsort(S)[::-1]
    U, S, V = U[:, ind], S[ind], V[:, ind]

    eigenvector_subset = V[:,0:num_components]

    # if you multiply the first component by -1, it will closely match the PCA algorithm above
    #https://stackoverflow.com/questions/27781872/eigenvectors-computed-with-numpys-eigh-and-svd-do-not-match
    eigenvector_subset[:,0] = -1*eigenvector_subset[:,0]

    eigenvalue_subset = U[:,0:num_components]

    X_reduced = np.dot(eigenvector_subset.T , ma_X.T ).T
    X_reduced2 = np.dot(eigenvalue_subset.T , ma_X.T ).T
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


        #Applying it to SVD function
        mat_reduced, reconstructed = SVD(df , 3)

        #Creating a Pandas DataFrame of reduced Dataset
        principal_df = pd.DataFrame(mat_reduced , columns = ['PC1','PC2','PC3'])
        
        principal_df=(principal_df-principal_df.mean())/principal_df.std()
        
        #Concat it with target variable to create a complete Dataset
        principal_df = pd.concat([principal_df , df['dayPercentComplete']] , axis = 1)
        

        #colors = list()
        #palette = {0: "red", 64: "green", 10: "blue", 26: "orange", 16: "yellow", 80: "black"}

        #for c in target: 
        #    colors.append(palette[int(c)])
        images = []
        plt.scatter(principal_df['dayPercentComplete'],principal_df['PC1'], cmap='YlOrRd', c=cc_watts["cc_watts"], s=1)
        plt.xlabel('time of day')
        plt.ylabel('PC1')

        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)  # rewind to beginning of file
        
        figdata_png = base64.b64encode(figfile.getvalue())
        images.append(figdata_png.decode('utf8'))
        plt.clf()

        plt.scatter(principal_df['dayPercentComplete'],principal_df['PC2'], cmap='YlOrRd', c=df["battery_voltage"], s=1)
        plt.xlabel('time of day')
        plt.ylabel('PC2')

        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)  # rewind to beginning of file
        
        figdata_png = base64.b64encode(figfile.getvalue())
        images.append(figdata_png.decode('utf8'))
        plt.clf()
        
        plt.scatter(principal_df['dayPercentComplete'],principal_df['PC3'], cmap='YlOrRd', c=df["battery_voltage"], s=1)
        plt.xlabel('time of day')
        plt.ylabel('PC3')

        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)  # rewind to beginning of file
        
        figdata_png = base64.b64encode(figfile.getvalue())
        images.append(figdata_png.decode('utf8'))
        plt.clf()

        heatmap(images,df)
        


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
