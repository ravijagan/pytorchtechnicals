from matplotlib import pyplot

import sranodec as anom
def gen_score(test_signal, amp_window_size= 12,series_window_size=4 ,score_window_size=16):
    # less than period amp_window_size= 12
    # (maybe) as same as period series_window_size=4
    # score_window_size=16a number enough larger than period
    spec = anom.Silency(amp_window_size, series_window_size, score_window_size)
    score = spec.generate_anomaly_score(test_signal)
    #pyplot.plot(score)
    return score

def plotres(start, end)  :
    df = (get_all_data (
        query="""select prediction, delta from predictions join technicals on  
              timestampx::text = predicted_for::text 
              where timestampx >= %s and timestampx <= %s 
              order by 1 desc"""%(start, end), retdf=False, columns=None,
        tablename=None, stripnan=True))
    pyplot.scatter (df [:, 0], df [:, 1])
    pyplot.show ()
import queries2 as q
import dbutils as d
from testandutils import *
from json import loads, dumps
def makejsondataoffline(q = query, filename = 'jsonout.json'):
    query = q.multiplesymbols
    dt = d.get_all_data(query=query, retdf=True)
    dt.to_json(filename, orient="index")
    #dt.to_json('spx_es_tlt_split', orient='split')
    return dt
from statsmodels.tsa.seasonal import seasonal_decompose
def get_residuals(data, period=16):
    result = seasonal_decompose(data, model='additive', period=period)
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid
    bstrip =residual[~np.isnan(residual).any()]
    residual.shape
    bstrip.shape

    print(score[0:100])
    return trend, seasonal, residual, score

def foo():
    dt = makejsondataoffline()
    b = dt['spx'][100:500].to_numpy()
    b = b[~np.isnan(b).any()][0]
    trend, seasonal, residual, score = get_residuals(b)
    print(list(score))
    pass
    #np.where(score > score + 2*score.std())
    #np.where(b > 4465.49 )


makejsondataoffline(q = q1min2022, filename = 'spx22_indexoriented')