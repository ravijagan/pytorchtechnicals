#import boto3
from dateutil.parser import parse
#anamoly_input_len = 128 moved to config
anamoly_input_len = None
if not anamoly_input_len :
    anamoly_input_len = 16
def sendm(phonelist, err = None):
    if err < 0.6:
        tstamp = time.strftime("%h%d_%H_%M_%S", time.localtime())
        message = "%s SPX could change by $20+ in next 30-45 mins : score %s "%(tstamp, 1-err)
        print(message)
    else:
        return
    client = boto3.client(
        "sns",
        aws_access_key_id="AKIAJGR5JXS4LLXZZPLQ",
        aws_secret_access_key="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",
        region_name="us-east-1",
    )
    for phone in phonelist:
        client.publish(
            PhoneNumber=phone,
            Message= message + "do not reply to this "
        )
        print(phone, message)

class ecdlive:
    modelm, xmean , xstd = None, None, None
    def __init__(self,modelm, xmean, xstd, anamoly_input_len):
        self.modelm = modelm
        self.xmean  = xmean
        self.xstd   = xstd
        self.anamoly_input_len = anamoly_input_len
        self.save_df = None
        self.tobesaved=0
    def saveprediction(self, prediction, predicted_for, savetable, final=False):
        nowx = datetime.datetime.now ()
        predicted_for = int(predicted_for)
        ptmp =  pd.DataFrame ([[nowx, prediction, predicted_for]],
                             columns=['predictiontime', 'prediction', 'predicted_for'])
        if not self.tobesaved:
            self.save_df = ptmp
        else:
            self.save_df = self.save_df.append(ptmp)
        self.tobesaved += 1
        if final or self.tobesaved%100==0:
            save_pd_db (self.save_df, savetable, 'append')
            self.tobesaved =0
            self.save_df = None

    def testonce(self,dataquery, savetable=None):
        Xlive, extras = make_Xa_live ([self.anamoly_input_len],
                             dataquery,
                              self.xmean, self.xstd, None, tablename=None)
        try:
            if(Xlive.shape)[1] >= self.anamoly_input_len:
                pass
                #print (Xlive.shape, "makeXaLive")
            # print ("Xlive is of shape", Xlive.shape)
            np.set_printoptions (threshold=self.anamoly_input_len)
            tmse = tf.keras.losses.MeanSquaredError ()
            sess = K.get_session ()
            with sess.as_default ():
                xdecoded = self.modelm.predict (Xlive, verbose=0)
                err = tmse (Xlive [0], xdecoded [0])
                sx = err.eval ()  # this is the error across the whole batch
                print (" %s PREDICTED sx: %s \t" % (time.strftime ("%h%d_%H_%M_%S",
                                                                   time.localtime ()), sx))
            return sx
        except:
            traceback.print_exc()
            return None

    def testforever(self):
        while(1):
            nowx = datetime.datetime.now ()
            startstr = (nowx - datetime.timedelta (days=14)).strftime ("%Y%m%d%H%M")
            endt = nowx + datetime.timedelta (hours=2)
            endstr = int(endt.strftime ("%Y%m%d%H%M"))  # tbd convert to chicago
            dataquery = ecd_base_query % ({'tablename': 'technicals_live',
                                           'starttime': startstr,
                                           'endtime': endstr})
            try:
                score = self.testonce(dataquery)
                self.saveprediction (score, endstr, 'predictions')
                sendm (['+14085487284'], score)
            except:
                traceback.print_exc()
                print(dataquery)
                time.sleep(10)
                pass
            time.sleep(30)
    # NOT LIVE USUALLY
    def test_n_steps(self,starttestat, n_steps):
        startx = parse(starttestat)
        steps = 0
        endt = startx # where input end ends
        while steps < n_steps:
            steps += 1
            if steps == n_steps:
                final = True
            else:
                final = False
            endt     = endt + datetime.timedelta(minutes=10)
            startt   = endt - datetime.timedelta(days=14) # not too large holidays where input begins
            if endt.hour < 6 or endt.hour > 13 :
                endt = endt + datetime.timedelta (minutes=60) #tbd be smarter than 60
                continue # until it hits the hour
            endtstr  = endt.strftime ("%Y%m%d%H%M")
            startstr  = startt.strftime("%Y%m%d%H%M")
            # edit the table to go to live if needed
            dataquery = ecd_base_query % ({'tablename': 'technicals',
                                           'starttime': startstr,
                                       'endtime': endtstr})
            try:
                prediction = self.testonce(dataquery)
                #tbd create a big dataframe and then save
                self.saveprediction (prediction, endtstr ,'predictions', final)
            except:
                traceback.print_exc()
                time.sleep(1) # in case database being updated
                self.saveprediction (prediction, endtstr, 'predictions', final)
                pass
            print("--tested --- ", endtstr)
            #time.sleep(1)
from matplotlib import pyplot
def plotres(start, end)  :
    df = (get_all_data (
        query="""select prediction, delta from predictions join technicals on  
              timestampx::text = predicted_for::text 
              where timestampx >= %s and timestampx <= %s 
              order by 1 desc"""%(start, end), retdf=False, columns=None,
        tablename=None, stripnan=True))
    pyplot.scatter (df [:, 0], df [:, 1])
    pyplot.show ()


import dbutils as d
from queries2 import *

import sranodec as anom
def gen_score(test_signal, amp_window_size= 12,series_window_size=4 ,score_window_size=16):
    # less than period amp_window_size= 12
    # (maybe) as same as period series_window_size=4
    # score_window_size=16a number enough larger than period
    spec = anom.Silency(amp_window_size, series_window_size, score_window_size)
    score = spec.generate_anomaly_score(test_signal)
    pyplot.plot(score)
    return score


def makejsondataoffline( query, filename = 'jsonout.json'):
    dt = d.get_all_data(query=query, retdf=True)
    dt.to_json("C:\\Users\\ravi\\Downloads\\"+filename, orient="index")
    #dt.to_json('spx_es_tlt_split', orient='split')
    return dt

from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
def get_residuals(data, period=16):
    result = seasonal_decompose(data, model='additive', period=period)
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid
    bstrip =residual[~np.isnan(residual).any()]
    return trend, seasonal, residual

def foo(colname='close'):
    dt = makejsondataoffline(q1min2022, filename = 'spx22_indexoriented')
    b = dt[colname][100:500].to_numpy()
    b = b[~np.isnan(b).any()][0]
    trend, seasonal, residual= get_residuals(b)
    offset = 100 # first few will be nans
    score = gen_score(test_signal=residual[offset:offset+200], amp_window_size=12, series_window_size=4, score_window_size=16)
    idx_changes = np.where(score > np.percentile(score, 95))[0]
    print(idx_changes+offset)
    list(idx_changes)
    for i in idx_changes+offset :
        print ('index', i, b[i-8:i+8])
        pyplot.plot(b[i-24:i+24])
        pyplot.title(f"{i}")
        pyplot.show()

    pass
    #np.where(score > score + 2*score.std())
    #np.where(b > 4465.49 )


foo('close')
makejsondataoffline( q1min2022, filename = 'spx22_indexoriented')
