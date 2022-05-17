import argparse
from sys import argv
from typing import NamedTuple, List, Type, Iterable
import logging

from pymodbus.constants import Defaults
from pymodbus.exceptions import ModbusIOException, ConnectionException

from mate3 import mate3_connection
import time
from mate3.api import AnyBlock, Device
from mate3.base_structures import get_parser

from flask import Flask, render_template, request, url_for, redirect                                                      
import threading

import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime
from dateutil import tz
import seaborn as sns
import random
from sklearn.decomposition import PCA

default_figure_size = plt.rcParams.get('figure.figsize')
logger = logging.getLogger('mate3.mate3_pg')


try:
    from yaml import load, FullLoader
except ImportError:
    raise ImportError("To use this command you must install the pyyaml package")

try:
    import psycopg2
    from psycopg2.extensions import quote_ident
except ImportError:
    raise ImportError("To use this command you must install the psycopg2-binary (or psycopg2) package")

host = '192.168.0.123'
port = 502
args = None

class Table(NamedTuple):
    name: str
    definitions: List["Definition"]


class Definition(NamedTuple):
    port: int
    device: Device
    field: str
    db_column: str


def read_definitions(f) -> List[Table]:
    logger.info(f"Reading field definitions from {f.name}")
    in_yaml = load(f, Loader=FullLoader)
    tables = []

    for table_name, in_definitions in in_yaml.get('tables', {}).items():
        definitions = []
        for in_definition in in_definitions:
            # Get the block class
            definitions.append(Definition(
                port=int(in_definition['port']),
                device=Device[in_definition['device']],
                field=in_definition['field'],
                db_column=in_definition['db_column'],
            ))

        tables.append(
            Table(table_name, definitions)
        )

    logger.debug(f"Found definitions: {tables}")
    return tables


def create_table(conn, table: Table, hypertables: bool):
    with conn.cursor() as curs:
        # Create the table in case it does not already exist
        sql = (
            f"CREATE TABLE IF NOT EXISTS {quote_ident(table.name, curs)} (\n"
            f"    timestamp TIMESTAMPTZ NOT NULL\n"
            f")"
        )
        logger.info(f"Executing: {sql}")
        curs.execute(sql)

        # Get existing columns
        sql = (
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' "
            "AND table_name = %s"
        )
        curs.execute(sql, [table.name])
        column_names = {row[0] for row in curs.fetchall()}

        for definition in table.definitions:
            if definition.db_column not in column_names:
                parser = get_parser(definition.device)
                field = getattr(parser, definition.field)
                column_type = 'VARCHAR(100)' if field.type == str else 'INTEGER'
                sql = f'ALTER TABLE {quote_ident(table.name, curs)} ADD COLUMN {quote_ident(definition.db_column, curs)} {column_type} NULL'
                logger.debug(f"Executing: {sql}")
                curs.execute(sql)

        if hypertables:
            try:
                sql = f"SELECT create_hypertable('{table.name}', 'timestamp')"
                logger.debug(f"Executing: {sql}")
                curs.execute(sql, [table.name])
            except psycopg2.DatabaseError as e:
                if 'already a hypertable' in str(e):
                    logger.debug("Table is already a hypertable")
                else:
                    raise

def create_tables(conn, tables: List[Table], hypertables: bool):
    logger.info("Creating tables (if needed)")
    for table in tables:
        create_table(conn, table, hypertables)


def _get_value(blocks: List[AnyBlock], definition: Definition) -> ...:
    for block in blocks:
        if not hasattr(block, 'port_number'):
            continue

        if block.port_number == definition.port and block.device == definition.device:
            return getattr(block, definition.field)


def insert(conn, tables: List[Table], blocks: List[AnyBlock]):

    with conn.cursor() as curs:
        for table in tables:
            insert_kv = {}
            for definition in table.definitions:
                value = _get_value(blocks, definition)
                if value is not None:
                    insert_kv[definition.db_column] = value

            column_names = [quote_ident(c, curs) for c in insert_kv.keys()]
            placeholders = ['%s'] * len(insert_kv)
            sql = (
                f"INSERT INTO {quote_ident(table.name, curs)} "
                f"(timestamp, {', '.join(column_names)}) "
                f"VALUES (NOW(), {', '.join(placeholders)})"
            )
            values = list(insert_kv.values())
            logger.debug(f"Executing: {sql}; With values {values}")
            curs.execute(sql, values)


def main():
    global args
    parser = argparse.ArgumentParser(description="Read all available data from the Mate3 controller")

    parser.add_argument(
        "--host", "-H",
        dest="host",
        help="The host name or IP address of the Mate3",
        required=True,
    )
    parser.add_argument(
        "--port", "-p",
        dest="port",
        default=Defaults.Port,
        help="The port number address of the Mate3",
    )
    parser.add_argument(
        "--interval", "-i",
        dest="interval",
        default=5,
        help="Polling interval in seconds",
        type=int,
    )
    parser.add_argument(
        "--database-url",
        dest="database_url",
        help="Postgres database URL",
        default="postgres://postgres@localhost/postgres",
    )
    parser.add_argument(
        "--definitions",
        dest="definitions",
        default='text',
        help="YAML definition file",
        type=argparse.FileType('r'),
        required=True,
    )
    parser.add_argument(
        "--hypertables",
        dest="hypertables",
        help="Should we create tables as hypertables? Use only if you are using TimescaleDB",
        action='store_true',
    )
    parser.add_argument(
        "--quiet", "-q",
        dest="quiet",
        help="Hide status output. Only errors will be shown",
        action='store_true',
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        help="Show debug logging",
        action='store_true',
    )

    args = parser.parse_args(argv[1:])

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.ERROR)
    root_logger = logging.getLogger()
    mate3_logger = logging.getLogger('mate3')

    if args.debug:
        root_logger.setLevel(logging.DEBUG)
    elif args.quiet:
        mate3_logger.setLevel(logging.ERROR)
    else:
        mate3_logger.setLevel(logging.INFO)

    tables = read_definitions(args.definitions)

    logger.info(f"Connecting to Postgres at {args.database_url}")
    with psycopg2.connect(args.database_url) as conn:
        conn.autocommit = True
        logger.debug(f"Connected to Postgres")
        create_tables(conn, tables, hypertables=args.hypertables)
        conn.commit()
        while True:  # Reconnect loop
            try:
                while True:  # Block fetching loop
                    logger.debug(f"Connecting to mate3 at {args.host}:{args.port}")
                    start = time.time()

                    # Read data from mate3s
                    # We keep the connection open for the minimum time possible
                    # as the mate3s cannot only sustain one modbus connection at a once.
                    with mate3_connection(args.host, args.port) as client:
                        blocks = list(client.all_blocks())

                    # Insert into postgres
                    insert(conn, tables, blocks)
                    conn.commit()
                    # Sleep until the end of this interval
                    total = time.time() - start
                    sleep_time = args.interval - total
                    if sleep_time > 0:
                        time.sleep(args.interval - total)

            except (ModbusIOException, ConnectionException) as e:
                logger.error(f"Communication error: {e}. Will try to reconnect in {args.interval} seconds")
                time.sleep(args.interval)

app = Flask(__name__)

import numpy as np
import time


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

sqlToday = "SELECT * FROM device_data_logs1 where timestamp >= date_trunc('day', now() AT TIME ZONE 'PST') AT TIME ZONE 'PST' order by timestamp desc "
sqlYesturday = "SELECT * FROM device_data_logs1 where timestamp >= date_trunc('day', (now() - interval '24h') AT TIME ZONE 'PST') AT TIME ZONE 'PST' AND timestamp <= date_trunc('day', now() AT TIME ZONE 'PST') AT TIME ZONE 'PST' order by timestamp desc "
        
@app.route("/")
def index():
    return render_template('index.html')

@app.route('/graphsvd')
def graphsvd():
    global args
    with psycopg2.connect(args.database_url) as conn:
        sql = sqlToday
        if 'yesturday' in request.args:
            sql = sqlYesturday
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

        images = []
        plt.scatter(df['dayPercentComplete'],cc_watts['cc_watts'], cmap='YlOrRd', c=cc_watts["cc_watts"], s=1)
        plt.xlabel('time of day')
        plt.ylabel('watts')

        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)  # rewind to beginning of file
        
        figdata_png = base64.b64encode(figfile.getvalue())
        images.append(figdata_png.decode('utf8'))
        plt.clf()

        plt.scatter(df['dayPercentComplete'],df['shunt_c_accumulated_kwh'], cmap='YlOrRd', c=df["shunt_c_accumulated_kwh"], s=1)
        plt.xlabel('time of day')
        plt.ylabel('accumulated_kwh')

        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)  # rewind to beginning of file
        
        figdata_png = base64.b64encode(figfile.getvalue())
        images.append(figdata_png.decode('utf8'))
        plt.clf()

        plt.scatter(df['dayPercentComplete'],df['battery_voltage'], cmap='YlOrRd', c=df["battery_voltage"], s=1)
        plt.xlabel('time of day')
        plt.ylabel('battery_voltage')

        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)  # rewind to beginning of file
        
        figdata_png = base64.b64encode(figfile.getvalue())
        images.append(figdata_png.decode('utf8'))
        plt.clf()
        
        plt.scatter(df['dayPercentComplete'],df['shunt_c_current'], cmap='YlOrRd', c=df["shunt_c_current"], s=1)
        plt.xlabel('time of day')
        plt.ylabel('shunt_c_current')

        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)  # rewind to beginning of file
        
        figdata_png = base64.b64encode(figfile.getvalue())
        images.append(figdata_png.decode('utf8'))
        plt.clf()
        
        df = df.loc[:, (df != 0).any(axis=0)]
        heatmap(images,df)
        
        return render_template('svd.html', images=images)


@app.route('/graphsvddata')
def graphsvddata():
    global args
    with psycopg2.connect(args.database_url) as conn:
        sql = sqlToday
        if 'yesturday' in request.args:
            sql = sqlYesturday
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

        df_scaled=(df-df.mean())/df.std()
        pca = PCA()
        X_pca = pca.fit_transform(df_scaled)
        component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        X_pca = pd.DataFrame(X_pca, columns=component_names)

        loadings = pd.DataFrame(
            pca.components_.T,  # transpose the matrix of loadings
            columns=component_names,  # so the columns are the principal components
            #index=df.columns,  # and the rows are the original features
        )
        loadings.insert(0, 'columns', df.columns)
        return render_template('graphsvddata.html', samples=loadings.values.tolist(), column_names=loadings.head())

@app.route('/about')
def about():
    global args
    with psycopg2.connect(args.database_url) as conn:
        sql = sqlToday + " LIMIT 25"
        if 'yesturday' in request.args:
            sql = sqlYesturday+ " LIMIT 25"
        df = pd.read_sql_query(sql, conn)
        if df.empty:
            raise ValueError('There are no results when there should be')
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

        #conn.close()
        return render_template('about.html', samples=df.values.tolist(), column_names=df.head())

if __name__ == '__main__':
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8511, debug=True, use_reloader=False)).start()
    main()
