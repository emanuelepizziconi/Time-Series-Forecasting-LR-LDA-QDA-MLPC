#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd 
import datetime as dt
import sklearn
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import yfinance as yf
from pandas.tseries.offsets import BDay


# In[5]:


def retrievedata (ticker, start_dt,end_dt, nlags):
    ydata = [ticker]
    mdata = yf.download(ydata, start=start_dt, end=end_dt)
    df = pd.DataFrame(data=mdata)
    dflag = pd.DataFrame(index=df.index)
    dflag["Today"] = df["Adj Close"]
    dflag["Volume"] = df["Volume"]
    dfret = pd.DataFrame(index=dflag.index)
    dfret["Volume"] = dflag['Volume']
    dfret["Adj Close"] = df["Adj Close"]
    dfret["Adj Close t-1"] = df["Adj Close"].shift(+1)
    dfret["Today"] = dflag["Today"].pct_change()*100.0
    print("Non-Directional Days: ",len(dfret.loc[dfret['Today'].abs() <= 0.0001]))
    dfret.loc[dfret['Today'].abs() <= 0.00001, 'Today'] = dfret['Today'].shift(1).loc[dfret['Today'].abs() <= 0.00001]
    for i in range(0, nlags):
            dflag[f"Lag {i+1}"] = df["Adj Close"].shift(i+1)
            dfret[f"Lag {i+1}"] = dflag[f"Lag {i+1}"].pct_change()*100.0
    dfret["Move"] = np.sign(dfret["Today"])
    dfret = dfret[dfret.index >= start_dt]
    dfret = dfret.dropna(axis='rows', how='any')
    return dfret


# In[7]:


def fit_model(name, model, X_train, y_train, X_test, X_forecast, pred):
    if name == "MLPC":
        model = MLPC(max_iter=5000).fit(X_train, y_train)
    else:
        try:
            model.fit(X_train, y_train)
        except:
            model = MLPC(max_iter=5000).fit(X_train, y_train)
            print("Error: MLPC executed instead of ", name)
    pred[name] = model.predict(X_test)
    prediction = model.predict(X_forecast)
    if prediction ==1:
        print(name, ": Buy T+1")
    else:
        print(name, ": Sell T+1")
    pred["%s_Trade" % name] = pred[name]*pred["Adj Close"] - pred[name]*pred["Adj Close t-1"]
    pred["%s_Correct" % name] = (1.0+pred[name]*pred["Actual"])/2.0
    hit_rate = np.mean(pred["%s_Correct" % name])
    print( "%s: %.3f" % (name, hit_rate))
    return hit_rate


# In[9]:


def fit_data(ticker, start_T):
    print(ticker)
    today = pd.Timestamp.today()
    hist_data = retrievedata(ticker,"2000-01-01",today, 10)
    Xclose = hist_data[["Adj Close","Adj Close t-1"]]
    X = hist_data[["Lag 1","Lag 2","Lag 3","Lag 4","Lag 5", "Lag 6", "Lag 7", "Lag 8", "Lag 9", "Lag 10"]]
    y = hist_data["Move"]
    start_test = start_T
    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]
    X_forecast = hist_data[["Today","Lag 1","Lag 2","Lag 3","Lag 4","Lag 5", "Lag 6", "Lag 7", "Lag 8", "Lag 9"]]
    X_forecast.columns = ['Lag 1', 'Lag 2', 'Lag 3', 'Lag 4', 'Lag 5', 'Lag 6', 'Lag 7', 'Lag 8', 'Lag 9', 'Lag 10']
    X_forecast = X_forecast[X_forecast.index >= X.index.max()]
    ind = []
    ind.append(X.index.max()+pd.Timedelta(days=1))
    X_forecast = X_forecast.set_index(pd.Index(ind))
    pred = pd.DataFrame(index=y_test.index)
    pred[["Adj Close","Adj Close t-1"]] = Xclose[Xclose.index >= start_test]
    pred["Actual"] = y_test
    print("Hit Rates:")
    models = (("LR", LogisticRegression()), ("LDA", LDA()), ("QDA", QDA()), ("MLPC", MLPC()))
    rate = []
    for m in models:
        rate.append(fit_model(m[0], m[1], X_train, y_train, X_test, X_forecast, pred))
    pred["AVG"] = pred["LR"]+ pred["LDA"]+ pred["QDA"] + 2*pred["MLPC"]
    pred.loc[pred['AVG'] <= 0, 'AVG'] = -1
    pred.loc[pred['AVG'] > 0, 'AVG'] = 1
    pred["AVG_Trade"] = pred["AVG"]*pred["Adj Close"] - pred["AVG"]*pred["Adj Close t-1"]
    pred["AVG_Correct"] = (1.0+pred['AVG']*pred["Actual"])/2.0
    hit_rate = np.mean(pred["AVG_Correct"])
    print("AVG: ","%.3f" % (hit_rate))
    rate.append(hit_rate)
    print("LR:","%.3f" %(np.sum(pred["LR_Trade"])/pred["Adj Close t-1"][0]*100),"%")
    print("LDA:","%.3f" %(np.sum(pred["LDA_Trade"])/pred["Adj Close t-1"][0]*100),"%")
    print("QDA:","%.3f" %(np.sum(pred["QDA_Trade"])/pred["Adj Close t-1"][0]*100),"%")
    print("MLPC:","%.3f" %(np.sum(pred["MLPC_Trade"])/pred["Adj Close t-1"][0]*100),"%")
    print("AVG:","%.3f" %(np.sum(pred["AVG_Trade"])/pred["Adj Close t-1"][0]*100),"%")
    return rate


# In[34]:


rates = []
#tickers = pd.read_csv('S&P500.csv')
#print(tickers)
#c = 0
#while c < len(tickers["Ticker"]):
#    try:
#         rates.append(fit_data(tickers["Ticker"][c],"2024-06-01"))
#    except:
#        pass
#    c = c+1
rates.append(fit_data("NVDA","2024-06-01"))
rates_arr = np.array(rates, dtype=float)
rates1 = rates_arr.mean(axis = 0)
print(rates1)

