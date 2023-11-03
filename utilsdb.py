import logging
import pandas as pd

from sqlalchemy import create_engine, text


import numpy as np
import time

# follows django database settings format, replace with your own settings
DATABASES = {
'production':{
'NAME': 'ravi',
'USER': 'black',
'PASSWORD': 'scholes',
'HOST':  '172.18.240.1',  # '127.0.0.1' , # 127.0.0.1 is local, 10.1.1.96 is the windows machine
'PORT': 5432,
    },
}
# choose the database to use
db = DATABASES['production']
# construct an engine connection string
engine_string = "postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}".format(
user = db['USER'],
password = db['PASSWORD'],
host = db['HOST'],
port = db['PORT'],
database = db['NAME'],
)
def save_pd_db(df, tablename, if_exists):
    engine = create_engine(engine_string)
    df.to_sql(tablename, index = False, if_exists=if_exists, con=engine)
    #print(if_exists, tablename)
    engine.dispose() # this application is ok because few reads and
    return

def read_pd_db(tablename):
    engine = create_engine(engine_string)
    df = pd.read_sql("select * from %s"%(tablename), engine)
    engine.dispose()
    return df

# create sqlalchemy engine
def get_all_data(query=None, retdf = False, columns = None ,
                 tablename=None, stripnan=True, if_exists='replace'):
    # read from query or tablename. if both or there save the query into tablename
    assert query or tablename
    engine = create_engine(engine_string)
    # read a table from database into pandas dataframe, replace "tablename" with your table name
    if query:
        for i in range(3):
            #print(query)
            #df = pd.read_sql(query,engine) deprecated use the cpnn
            with engine.begin() as conn:
                df = pd.read_sql_query(sql=text(query), con=conn)
            if df.size > 0:
                break
            else:
                #print(query)
                logging.info(f'{query}')
                time.sleep(30)
        if tablename:
            save_pd_db (df, tablename, if_exists)
    else:
        df = read_pd_db(tablename)
        #print("reading", tablename )
        logging.info(f'reading {tablename}')
    engine.dispose()
    if columns :
        df = df[columns]

    if retdf: # return a data frame instead
        return df
    b = (df).to_numpy()
    #b = b[numpy.logical_not(numpy.isnan(b))]
    if b.shape[0] > 0 and stripnan:
        if(np.isnan (b).sum()):
            #print(b.shape, "stripped",np.isnan (b).sum())
            logging.info(f'{b.shape} stripped {np.isnan (b).sum()}')
        b = b[~np.isnan(b).any(axis=1)]


    return b