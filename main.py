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

    # if you multiply the first component by -1, it will closely match the SVD algorithm below
    #https://stackoverflow.com/questions/27781872/eigenvectors-computed-with-numpys-eigh-and-svd-do-not-match
    eigenvector_subset[:,0] = -1*eigenvector_subset[:,0]
    
    #Step-6
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
    
    return X_reduced

def SVD(X , num_components):

    #X_meaned=(X-X.mean())/X.std()

    X_meaned = X - np.mean(X , axis = 0)

    cov_mat = np.cov(X_meaned , rowvar = False)

    U, S, Vt = np.linalg.svd(cov_mat, full_matrices=False)
    V = Vt.T
    # Sort the PCs by descending order of the singular values (i.e. by the proportion of total variance they explain)
    ind = np.argsort(S)[::-1]
    U, S, V = U[:, ind], S[ind], V[:, ind]

    eigenvector_subset = V[:,0:num_components]

    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()

    return X_reduced

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/graphsvd')
def graphsvd():
    global args
    with psycopg2.connect(args.database_url) as conn:

        df = pd.read_sql_query("SELECT * FROM device_data_logs1 where timestamp >= date_trunc('day', now() AT TIME ZONE 'PST') AT TIME ZONE 'PST' order by timestamp desc;", conn)
        midnight=(datetime
             .now(tz.gettz('America/Tijuana'))
             .replace(hour=0, minute=0, second=0, microsecond=0)
             .astimezone(tz.tzutc()))
        dayPercentComplete = df.iloc[:,0].apply(lambda x: (x.to_pydatetime() - midnight).total_seconds() / 60 / 60 / 24)
        df = df.loc[:, list(df.columns[1:23]) + list(df.columns[25:50])]
        df['dayPercentComplete'] = dayPercentComplete
        df = df.loc[:, (df != 0).any(axis=0)]
        #target == cc1_watts
        
        target = df.iloc[:,18]


        #Applying it to SVD function
        mat_reduced = SVD(df , 2)

        #Creating a Pandas DataFrame of reduced Dataset
        principal_df = pd.DataFrame(mat_reduced , columns = ['PC1','PC2'])
        
        #Concat it with target variable to create a complete Dataset
        principal_df = pd.concat([principal_df , pd.DataFrame(target)] , axis = 1)
        

        #colors = list()
        #palette = {0: "red", 64: "green", 10: "blue", 26: "orange", 16: "yellow", 80: "black"}

        #for c in target: 
        #    colors.append(palette[int(c)])


        plt.scatter(principal_df['PC1'], principal_df['PC2'], cmap='Greens', c=dayPercentComplete, s=1)
        plt.xlabel('PC1')
        plt.ylabel('PC2')

        
        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)  # rewind to beginning of file
        
        #figdata_png = base64.b64encode(figfile.read())
        figdata_png = base64.b64encode(figfile.getvalue())
        plt.clf()
        return render_template('svd.html', figdata_png=figdata_png.decode('utf8'))


@app.route('/graphsvddata')
def graphsvddata():
    global args
    with psycopg2.connect(args.database_url) as conn:

        df = pd.read_sql_query("SELECT * FROM device_data_logs1 where timestamp >= date_trunc('day', now() AT TIME ZONE 'PST') AT TIME ZONE 'PST' order by timestamp desc;", conn)
        midnight=(datetime
             .now(tz.gettz('America/Tijuana'))
             .replace(hour=0, minute=0, second=0, microsecond=0)
             .astimezone(tz.tzutc()))
        hourOfDay = df.iloc[:,0].apply(lambda x: (x.to_pydatetime() - midnight))
        dayPercentComplete = df.iloc[:,0].apply(lambda x: (x.to_pydatetime() - midnight).total_seconds() / 60 / 60 / 24 )
        df = df.loc[:, list(df.columns[1:23]) + list(df.columns[25:50])]
        df['dayPercentComplete'] = dayPercentComplete
        df = df.loc[:, (df != 0).any(axis=0)]
        #target == cc1_watts + cc2_watts
        cc_watts = pd.DataFrame(df['cc1_watts'] + df['cc2_watts'], columns = ['cc_watts']) 
        #Applying it to SVD function
        mat_reduced = SVD(df , 2)
        
        #Creating a Pandas DataFrame of reduced Dataset
        principal_df = pd.DataFrame(mat_reduced , columns = ['PC1','PC2'])
        
        #Concat it with cc_watts variable to create a complete Dataset
        principal_df = pd.concat([principal_df , pd.DataFrame(cc_watts)] , axis = 1)
        principal_df = pd.concat([principal_df , pd.DataFrame(hourOfDay)] , axis = 1)

        return render_template('graphsvddata.html', samples=principal_df.values.tolist(), column_names=principal_df.head())

@app.route('/about')
def about():
    global args
    with psycopg2.connect(args.database_url) as conn:
        df = pd.read_sql_query("SELECT * FROM device_data_logs1 where timestamp >= date_trunc('day', now() AT TIME ZONE 'PST') AT TIME ZONE 'PST' order by timestamp desc;", conn)
        midnight=(datetime
             .now(tz.gettz('America/Tijuana'))
             .replace(hour=0, minute=0, second=0, microsecond=0)
             .astimezone(tz.tzutc()))
        dayPercentComplete = df.iloc[:,0].apply(lambda x: (x.to_pydatetime() - midnight).total_seconds() / 60 / 60 / 24)
        #prepare the data
        df = df.loc[:, list(df.columns[0:23]) + list(df.columns[25:50])]
        df['dayPercentComplete'] = dayPercentComplete
        df = df.loc[:, (df != 0).any(axis=0)]
        # move column to 2nd place
        col = df.pop("dayPercentComplete")
        df.insert(1, col.name, col)

        df = df.loc[:, (df != 0).any(axis=0)]

        #conn.close()
        return render_template('about.html', samples=df.values.tolist(), column_names=df.head())

if __name__ == '__main__':
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8511, debug=True, use_reloader=False)).start()
    main()
