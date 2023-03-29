import talib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from dbutils import *
from matplotlib import pyplot
import math, pydotplus
from sklearn.preprocessing import scale
from sklearn.model_selection import *
from treeinterpreter import treeinterpreter as ti
from sklearn.linear_model import LogisticRegression
# example of making a single class prediction

from configs import *
from sklearn import linear_model
import os
from sklearn.tree import export_graphviz
import six
import pydot
from sklearn import tree

def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m, X_train, y_train, X_valid, y_valid):
    print("predict training rms , valid rmse, score train, score valid ")
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),"       ",
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)

get_data_query = spxrawbasic  #  # override whats in configs select t_date,  high , low, bid, ask, open , close  from spxraw

input_lensa = [30]
fsizes = [30]  # no downsampling
loadtable = 'technicals'

"""
# 30 min samples only one len needed
traingen, testgen, validgen, numfeatures , Xseq, Yseq, encoded_classes = make_data_gen(input_lensa,  reserve_for_test=1024, get_data_query = q )
#Xa, Ya, encoded_classes = make_Xa_Ya(input_lens, reserve_for_test)
x,y1  = next(traingen)
x1 = np.array(x[0]) # traingen returns a list get rid of the list
# x1 has shape 16 batches X 30 time periods X 7 features.
# cause our data mangling made a pd into a np
"""

# single time period cycles
def get_technicals( ta_lib_inputs, timeperiod):
    # note high water and low water are different . they are the closing price, not intro time period
    timestampx = ta_lib_inputs['timestampx']
    hour_min = ta_lib_inputs['hour_min']
    highest = ta_lib_inputs['highest']
    lowest = ta_lib_inputs['lowest']
    close = ta_lib_inputs['close']
    #close = ta_lib_inputs['mid']

    techfeaturenames = ['slowk', 'lowlowbool',
                    'highhighbool', 'parabolicsar', 'sma', 'ema', 'macd', 'macdhist', 'high_ema_spread', 'rsi']

    # macd, ema sma
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=timeperiod*12, slowperiod=9*timeperiod,
                                                   signalperiod=4)
    sma = talib.SMA(close, timeperiod=timeperiod)
    ema = talib.EMA(close, timeperiod = timeperiod*9 )
    # PARABOLIC SAR
    highw = talib.MAX(close, timeperiod=timeperiod)
    loww = talib.MIN(close, timeperiod=timeperiod)
    parabolicsar = talib.SAR(highw, loww, acceleration=0.02, maximum=0.2)
    slowk, slowd = talib.STOCH(highw, loww, close, fastk_period=10,
                           slowk_period=8, slowk_matype=0, slowd_period=10, slowd_matype=8)
    rsi = talib.RSI(close, timeperiod=timeperiod)
    high_low_spread = highw - loww
    high_close_spread = highw - close
    high_ema_spread = highw - ema
    # if slowk and slowd are almost same what is the red stochastic ?

    # high high is take the high array. Then is each element greater than previous element ?. To do this shorten array by 1 and do the subtraction. add a nan in th ebeinngin
    highhigh = highw[1:] - highw[:-1]
    highhigh = np.insert(highhigh, 0, np.nan)
    highhighbool = np.not_equal(highhigh, 0.)

    # lowlow is take the low array. Then is each element greater than previous element ?. To do this shorten array by 1 and do the subtraction. add a nan in bngin
    lowlow =   loww[1:] - loww[:-1]
    lowlow = np.insert(lowlow, 0, np.nan)
    lowlowbool = np.not_equal(lowlow, 0.)
    return techfeaturenames, slowk, lowlowbool, highhighbool, parabolicsar, sma, ema, macd, macdhist, high_ema_spread , rsi

# wrapper to make technicals with two time periods
def concat_technicals(input_lens, reserve_for_test, get_data_query, columns = None):
    # data = get_all_data(query , retdf = False )
    #input_lensx, reserve_for_test, get_data_query, columns, tablename = tablename
    X1, Y1 = make_Xa_Ya (get_data_query=get_data_query, columns = None, tablename=None,
                            stripnan=False)# loadtable)
                         # ,columns = ['rnk', 'hour_min', 'high' , 'low', 'bid', 'ask',
                         # 'open', 'close', 'delta_future']
    # cut paste  Xa = fulldata[-lastnrows:, 1:-1]  # all rows except last column whic is Y null keep the timestamp
    #filterout = np.isnan(Xa[:, :-1]).any(axis=1)
    #b = b[~np.isnan(b).any(axis=1)]
    Xa = X1[~np.isnan(X1[:, :-1]).any(axis=1)]
    Ya = Y1[~np.isnan(X1[:, :-1]).any(axis=1)]

    ta_lib_inputs = {
        'timestampx': Xa[:, 0],
        'month': Xa[:, 1],
        'day': Xa[:, 2],
        'dow': Xa[:, 3],
        'hour_min': Xa[:, 4],  # all batches, all rows, column0  in sql the order is timestamp, high, low...
        'highest': Xa[:, 5],
        'lowest': Xa[:, 6],
        #'bid': Xa[:, 7],
        #'ask': Xa[:, 8],
        #'open': Xa[:, 9],
        'close': Xa[:, 7],
    }
    feat_names_base = [   'timestampx',
        'hour_min',
        'month',
        'day',
        'dow',
        'close']
    timestampx = ta_lib_inputs['timestampx']
    month = ta_lib_inputs['month']
    day = ta_lib_inputs['day']
    hour_min = ta_lib_inputs['hour_min']
    dow = ta_lib_inputs['dow']
    hour_min = ta_lib_inputs['hour_min']
    close = ta_lib_inputs['close']
    #first rainbow stack
    tperiod = 2

    techfeaturenames, slowk, lowlowbool, highhighbool, parabolicsar, \
              sma, ema, macd, macdhist, high_ema_spread, rsi = get_technicals( ta_lib_inputs, timeperiod = tperiod)
    fnames = feat_names_base + ["%s_%d"%(x, tperiod) for x in techfeaturenames]
    #base and 30 minute features
    allfeat30 = np.vstack([timestampx, # this timestamp will overfit for sure strip before use
                      # hour_min used to filter  this help the overnight problem ?
                      hour_min,
                      month, day, dow,
                      close, slowk, lowlowbool,
                      highhighbool, parabolicsar, sma, ema, macd, macdhist, high_ema_spread, rsi]).T

    # second rainbow stack
    tperiod = 10

    techfeaturenames, slowk, lowlowbool, highhighbool, parabolicsar,\
            sma, ema, macd, macdhist, high_ema_spread, rsi = get_technicals( ta_lib_inputs,
                                                                                    timeperiod =tperiod)
    feat_names = fnames + ["%s_%d" % (x, tperiod) for x in techfeaturenames]
    allfeat10 = np.vstack([ slowk, lowlowbool,
                       highhighbool, parabolicsar,
                       sma, ema, macd, macdhist, high_ema_spread, rsi]).T

    tperiod = 2
    # Need  multiple of 3 features to exactly to match the  autencoder conv and deconv

    # use only some of the features
    techfeaturenames, slowk, lowlowbool, highhighbool, parabolicsar, \
          sma, ema, macd, macdhist, high_ema_spread , rsi = get_technicals(ta_lib_inputs, timeperiod=tperiod)
    feat_names = feat_names + [ "slowk_%s"%tperiod, "ema_%s"%tperiod,
                                "high_ema_spread_%s"%tperiod, "rsi_%s"%tperiod]
    allfeat2 = np.vstack([ slowk, ema, high_ema_spread, rsi]).T
    b_all_features = np.concatenate([allfeat30 , allfeat10, allfeat2 ], axis = 1 )
    # b_all_features = scale(b_all_features)
    # Ya = scale(Ya)
    #Ya = np.floor(Ya );
    Ya = np.clip(Ya, -50, 50)
    return b_all_features, feat_names, Ya


# first batch only hack tbd fix this laer
#??i = 3
#b_all_features, feat_names, Ya  = concat_technicals(input_lens, reserve_for_test, get_data_query, columns = None)
def do_something():
    print(" TIMESTAMP WILL OVERFIT STRIP IT HERE !!! ")
    m = RandomForestRegressor(n_estimators=1024,
                              max_features= "sqrt",
                              #max_depth = 16,
                              n_jobs=-1,
                              oob_score=True)
    #m = RandomForestClassifier(n_estimators=128, min_samples_leaf=.05, max_features="sqrt", max_depth = 140, n_jobs=-1,
    #            oob_score=True)
    #model = LogisticRegression()

    # to make it logistic Ya = np.round((  Ya /(np.abs(Ya)+0.001) /2   ) + 0.5)
    #m = linear_model.LinearRegression()
    #m.fit(b[2000:50000], Ya[2000:50000]);
    b = b_all_features
    X_train = b[5000:80000]
    Y_train = Ya[5000:80000]

    X_valid = b[80001:85000]
    y_valid = Ya[80001:85000]

    X_samp = b[85001:]
    y_samp = Ya[85001:]*1.0
    m.fit(X_train, Y_train)


    """fig = plt.figure()
    plt.plot(data)
    fig.suptitle('test title', fontsize=20)
    plt.xlabel('xlabel', fontsize=18)
    plt.ylabel('ylabel', fontsize=16)
    fig.savefig('test.jpg')"""

    y_samp_pred = m.predict(X_samp)
    y_valid_predict = m.predict(X_valid)
    y_train_predict = m.predict(X_train)

    print("sample shape", y_samp.shape, y_samp_pred.shape)
    pyplot.scatter(y_samp, y_samp_pred)
    pyplot.ylabel('y_samp_pred', fontsize=16)
    pyplot.show()

    pyplot.scatter(y_valid, y_valid_predict)
    pyplot.ylabel('y_valid_pred', fontsize=16)
    pyplot.show()
    print_score(m, X_train, Y_train, X_valid, y_valid)

    print("valid check", np.concatenate([X_valid[10:50],
                                         np.expand_dims(y_valid[10:50], axis = 1) ,
                                         np.expand_dims(y_valid_predict[10:50], axis = 1)
                                         ], axis = 1 ))


    #for k, v in zip(featnames, m.coef_) : print(k,v)
    importances = m.feature_importances_
    #importances = m.coef_[-1]

    print("feature_imp", importances)

    class_name = ['low', 'neutral', 'high']
    disp = zip (feat_names, importances)

    print("-----")
    for k,v in disp :
        print(k, (int(v*1000)))
    estimator = m.estimators_[10]
    """ to fix bug
    dot_data = export_graphviz(estimator, out_file='tree.dot',
                     feature_names = featnames,
                     class_names = class_name,
                    rounded = True, proportion = False,
                  precision = 2, filled = True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('tree.png')
    #os.system('dot -Tpng tree.dot -o tree.png')
    """
    #pyplot.plot(close, timeperiod=8)
    #pyplot.plot(macd)
    #pyplot.show()
    #macd is ndarray of shape timeperiod=30,
    # sklearn library has its own graping different from graphviz
    treedraw = False
    if treedraw:
        tree.plot_tree(m.estimators_[0],
                    feature_names = feat_names,
                    class_names= ["down", 'neut', 'up'],
                   filled = True);
    plt.show()

def create_technicals(get_data_query, savetable ):
    b_all_features, feat_names, Ya  = concat_technicals(input_lens, reserve_for_test,
                                                get_data_query, columns = None)
    tmparr = np.concatenate((b_all_features, np.expand_dims(Ya, axis=1)), axis=1)
    columnsx = feat_names + ["delta"]
    ptmp = pd.DataFrame(tmparr, columns=columnsx)
    print("beginning of save create_technicals ...")
    save_pd_db(ptmp, savetable, 'replace')
    print("end of save")

def historical():
    dq  = get_data_query % ({'tablename': 'spxmin'})
    #({'tablename': 'spxraw'})
    create_technicals( dq, 'technicals')

def live():
    # transform_ibkr takes ibkr_spx and converts into a table called transformed_live
    gdq = transform_ibkr%({'tablename': 'ibkr_spx'}) + spxrawbasic%({'tablename': 'transformed_live'})
    create_technicals( gdq, 'technicals_live')

if __name__ == "__main__" :
   #live() for testing and get_market_Data
   historical()