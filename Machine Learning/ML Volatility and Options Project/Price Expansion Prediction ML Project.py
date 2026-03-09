
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf

from sklearn.model_selection import train_test_split as tts
from sklearn import tree, ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, fbeta_score, confusion_matrix, classification_report, average_precision_score

from xgboost import XGBClassifier

import seaborn as sn
from scipy.stats import norm

# ==== Importing price data from yfinance ====

ticker = 'AAPL'              #Input ticker symbol
date_start = '2024-09-27'   #Input the start date in the form YYYY-MM-DD
timeframe = '1D'            

stock = yf.Ticker(ticker)
asset = stock.history(period='5y', interval=timeframe.lower(), auto_adjust=True)


# __________________________
# ==== Input Parameters ====
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

# These parameters can be adjusted to control certain aspects of the model

# ML Model Controls
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
upper_percentile = 70                           # Upper percentile for probabilistic prediction thresholds. Anything above this will be counted as a positive prediction (high volatility). 
data_split = np.array([0.6, 0.2, 0.2])          # Data split percentages [train, validation, test]

target_threshold = 'percentile'                 # Choose 'percentile' or 'fixed'. Decides the threshold for binary distinction of target variable as 'high future return' or not. A fixed threshold will define any future return above x% (e.g. 5%) as a high return (value 1). A rolling percentile will define the top x% of returns as high future return.
retrain = True                                  # Whether the model retrains using combined train and validation sets after the best model has been selected from the validation performance.
selection_metric = "Sharpe Ratio"               # The primary metric for model selection.
selection_metric_secondary = "ROC-AUC"          # The secondary metric for model selection, e.g. if all Sharpe ratios are negative on the validation set, it will use this metric instead.
override_model_selection = True                # Can override automatic model selection to use a specific ML model on the test set.
trading_model = "XGBoost"                       # Choose from: 'Random Forest', 'AdaBoost', 'XGBoost'. This is the model that will be used on the test set if automatic selection is overriden. 
# Trading Controls
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
expiry = 30                                     # Expiry length of options
TP = 30                                         # Take profit percentage
SL = 12                                         # Stop loss percentage
timeout = 15                                    # Exits trades after this length of time if TP or SL haven't been hit. Avoids carrying options to expiry.
risk_free_rate = 0.04                           # Risk free rate. Used for calculating Sharpe ratio.
sharpe_window = 63                              # Number of trading periods for the rolling Sharpe ratio window.
annualisation = 252                             # Annualisation factor for Sharpe ratio. 252 for daily data, 252*6.5 for hourly, 252*390 for minute.

# Results Controls
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾      
trade_to_csv = False                            # Choose whether to export the trade logs from the test set as a .csv file. This can be used for strategy validation.

# ____________________________________
# ==== Engineering input features ====
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

# Volatility compression
#       ATR - Long                  DONE
#       ATR compression = ATR / ATR.rolling(100).mean() - After ATR     DONE
#       Bollinger Band width        DONE
#       vol_5/vol_20 = vol_ratio    DONE
#       vol_20                      DONE

# trend strength
#       ADX - Long                  DONE
#       ADX_slope - After ADX       DONE
#       Distance from 50 MA         DONE

# Momentum
#   ROC_5 (rate-of-change)          DONE
#   ROC_10                          DONE
#   RSI - Long                      DONE

# Range expansion
#   range_20 (rolling_high_20 - rolling_low_20)     DONE
#  


# Distance from 50 MA
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
asset['50_MA'] = asset['Close'].rolling(window=50).mean().shift()
asset['50_MA_distance'] = (asset['Close'] - asset['50_MA']) / asset['50_MA']

# Volatility
# ‾‾‾‾‾‾‾‾‾‾
asset['log_return_1d'] = np.log(asset['Close'] / asset['Close'].shift(1))

asset['5_day_volatility'] = asset['log_return_1d'].rolling(5).std() 
asset['20_day_volatility'] = asset['log_return_1d'].rolling(20).std() 
asset['volatility_ratio'] = asset['5_day_volatility'] / asset['20_day_volatility']

# Bollinger Band Width
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
asset['20_MA'] = asset['Close'].rolling(window=20).mean().shift()
asset['20_STD'] = asset['Close'].rolling(window=20).std().shift()

upper_band = asset['20_MA'] + asset['20_STD'] * 2
lower_band = asset['20_MA'] - asset['20_STD'] * 2

asset['bollinger_width'] = (upper_band - lower_band) / asset['Close']

# Range
# ‾‾‾‾‾
asset['range_20'] = (asset['High'].rolling(20).max() - asset['Low'].rolling(20).min()) / asset['Close']

# Rate of Change
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾
asset['roc_5'] = asset['Close'].pct_change(5)
asset['roc_10'] = asset['Close'].pct_change(10)

# Defining Average True Range
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
def get_ATR(df, period):
    # Default period is 14
    highs_array = df['High'].values
    lows_array = df['Low'].values
    close_array = df['Close'].values

    TR_array = np.zeros(len(df))

    for i in range(1,len(df)):
        high = highs_array[i]
        low = lows_array[i]
        prev_close = close_array[i-1]

        TR_array[i] = max( (high-low),(high-prev_close),(low-prev_close) )
    
    df["True Range"] = TR_array
    df["ATR"] = df["True Range"].rolling(period).mean() #Normal ATR should not be shifted, i.e. ATR on row 14 uses data from rows 1-14.
    df["ATR_feature"] = df["ATR"].shift(1) #However, when used as a predictor in ML, it should be shifted once to avoid look-ahead bias.

    return df

# Defining Average True Range Compression
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
def get_ATR_compression(df, period):
    # Default period is 100
    df['ATR_compression'] = df['ATR'] / df['ATR'].rolling(period).mean()    
    df['ATR_compression'] = df['ATR_compression'].shift(1) # Shifting by 1 to avoid look-ahead bias.

    return df

# ==== Defining ADX ====
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

def get_ADX(df,period):
    # Default period is 14
    highs_array = df['High'].values
    lows_array = df['Low'].values
    close_array = df['Close'].values

    DM_plus = np.zeros(len(df))
    DM_minus = np.zeros(len(df))
    
    TR_array = df['True Range'].values    #True range from ATR is period 1
    TR_smoothed_array = np.zeros(len(df)) #Creating a smoothed true range of period 14

    # Calculating DM+ and DM- for entire data set (1 day period)
    for i in range(1,len(df)):
        up_move = highs_array[i]-highs_array[i-1]
        down_move = lows_array[i-1]-lows_array[i]

        if (up_move>down_move) and up_move>0:
            DM_plus[i],DM_minus[i] = up_move, 0
        elif (down_move>up_move) and down_move>0:
            DM_plus[i], DM_minus[i] = 0, down_move

    # Adding a smoothed version of DM+, DM- and TR (14 day period)
    DM_plus_14_smoothed = np.zeros(len(df))
    DM_minus_14_smoothed = np.zeros(len(df)) 

    DM_plus_14_smoothed[period] = np.sum(DM_plus[1:period+1])   # Calculating the first DM smoothed values with a simple sum then applying smoothing after
    DM_minus_14_smoothed[period] = np.sum(DM_minus[1:period+1]) # This should be the 15th value which is the sum of DM values 2 to 15 (13 values before it then this is the 14th)
    TR_smoothed_array[period] = np.sum(TR_array[1:period+1])    # Same for TR.

    #Applying Wilder smoothing
    for i in range(period+1,len(df)):
        DM_plus_14_smoothed[i] = DM_plus_14_smoothed[i-1] - DM_plus_14_smoothed[i-1]/period + DM_plus[i]   
        DM_minus_14_smoothed[i] = DM_minus_14_smoothed[i-1] - DM_minus_14_smoothed[i-1]/period + DM_minus[i]    
        TR_smoothed_array[i] = TR_smoothed_array[i-1] - TR_smoothed_array[i-1]/period + TR_array[i]

    #Calulating DI+ and DI-
    DI_plus = 100 * DM_plus_14_smoothed/ np.where(TR_smoothed_array==0, np.nan, TR_smoothed_array)
    DI_minus = 100 * DM_minus_14_smoothed/ np.where(TR_smoothed_array==0, np.nan, TR_smoothed_array)
    
    DI_plus = np.nan_to_num(DI_plus)
    DI_minus = np.nan_to_num(DI_minus)

    #Calculating Directional Index (DX)
    DX_array = 100 * abs(DI_plus - DI_minus) / np.where((DI_plus + DI_minus)==0, np.nan, (DI_plus + DI_minus))

    #Calculating ADX
    ADX_array = np.zeros(len(df))
    ADX_array[2*period - 1] = np.sum(DX_array[period:2*period])/period
    for i in range(2*period,len(df)):
        ADX_array[i] = ADX_array[i-1] + (DX_array[i]-ADX_array[i-1])/period
    
    df['-DM'] = DM_minus
    df['+DM'] = DM_plus
    df['TR14'] = TR_smoothed_array
    df['-DM14'] = DM_minus_14_smoothed
    df['+DM14'] = DM_plus_14_smoothed
    df['-DI14'] = DI_minus #DM_smoothed / TR_smoothed
    df['+DI14'] = DI_plus
    df['DX'] = DX_array
    df['ADX'] = ADX_array
    df['ADX_feature'] = df['ADX'].shift(1)

    return(df)

asset = get_ATR(asset, 14)
asset = get_ATR_compression(asset, 50)
asset = get_ADX(asset, 14)


# ADX Slope
# ‾‾‾‾‾‾‾‾‾
asset['ADX_slope'] = asset['ADX_feature'].diff(5) # ADX change over 5 days


# Defining RSI
# ‾‾‾‾‾‾‾‾‾‾‾‾
def get_RSI(asset, rsi_period):
    close_array = asset['Close'].values
    prev_closes = close_array[0:rsi_period]
    rsi_gains = 0
    rsi_losses = 0
    
    #Calculating starting RSI value
    for i in range(1,rsi_period):
        delta = prev_closes[i] - prev_closes[i-1]
        if delta>0: rsi_gains+=delta
        elif delta<0: rsi_losses+=abs(delta)

    rsi_avg_gain = rsi_gains/rsi_period
    rsi_avg_loss = rsi_losses/rsi_period

    rs_start = rsi_avg_gain/rsi_avg_loss
    rsi_start = 100 - (100/(1+rs_start))

    rsi_array = np.zeros(len(asset))
    rsi_array[rsi_period] = rsi_start

    #Calculating rolling RSI values

    for i in range(rsi_period+1, len(asset)):
        close = close_array[i]
        prev_close = close_array[i-1]
        
        delta = close - prev_close
        
        if delta > 0:
            rsi_avg_gain = ((rsi_avg_gain * (rsi_period - 1)) + delta) / rsi_period
            rsi_avg_loss = ((rsi_avg_loss * (rsi_period - 1)) + 0) / rsi_period
        elif delta < 0:
            rsi_avg_gain = ((rsi_avg_gain * (rsi_period - 1)) + 0) / rsi_period
            rsi_avg_loss = ((rsi_avg_loss * (rsi_period - 1)) + abs(delta)) / rsi_period
        elif delta == 0:
            rsi_avg_gain = ((rsi_avg_gain * (rsi_period - 1)) + delta) / rsi_period
            rsi_avg_loss = ((rsi_avg_loss * (rsi_period - 1)) + delta) / rsi_period

        RS = rsi_avg_gain/rsi_avg_loss
        rsi_local = 100 - (100/(1+RS))

        rsi_array[i] = rsi_local
    
    asset['RSI'] = rsi_array
    asset['RSI_feature'] = asset['RSI'].shift(1)

    return asset

asset = get_RSI(asset, 14)



# ______________________________________
# ==== Initialising Target Variable ====
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

# ==== Creating Target Variable ====
asset['future_return'] = asset['Close'].pct_change(timeout).shift(-timeout)
asset['abs_future_return'] = asset['future_return'].abs()

min_periods = 100

asset['rolling_target_threshold'] = asset['abs_future_return'].expanding(min_periods=min_periods).quantile(0.7)
asset = asset.iloc[min_periods:] #Dropping the first 100 (min_periods) data points so the expanding threshold window can come into effect.

if target_threshold == 'percentile':    
    asset['Target'] = (asset['abs_future_return'] >= asset['rolling_target_threshold']).astype(int)
elif target_threshold == 'fixed':
    asset['Target'] = (asset['abs_future_return'] >= 0.05).astype(int) # This will class any future return over 5% as 'high future return' (value 1).

asset_features = asset[['50_MA_distance','5_day_volatility','20_day_volatility','volatility_ratio',
                        'bollinger_width','range_20',
                        'roc_5','roc_10','RSI_feature',
                        'ATR_feature','ATR_compression',
                        'ADX_feature','ADX_slope']]

# ____________________________________________________________________
# ==== Splitting data into training, validation, and testing sets ====
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

# Defines a function for splitting the dataset
def split_dataframe(df, data_split):
    training_end = int(len(df)*data_split[0])
    val_end = int(training_end + len(df)*data_split[1])
    
    training = df[:training_end]
    validation = df[training_end:val_end]
    test = df[val_end:]

    return training, validation, test

training, validation, test = split_dataframe(asset, data_split)
test = test.copy()

# ________________________________________________________________________________
# ==== Converting to numpy arrays and splitting features from target variable ====
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

def convert_to_numpy(df):
    X = df[['5_day_volatility','20_day_volatility','volatility_ratio','ATR_feature','ADX']].to_numpy()
    y = df['Target'].to_numpy()

    return X,y

X_train,y_train = convert_to_numpy(training)
X_val,y_val = convert_to_numpy(validation)
X_test,y_test = convert_to_numpy(test)

# _______________________________________
# ==== Defining decision tree models ====
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

# Random Forest
# ‾‾‾‾‾‾‾‾‾‾‾‾‾
def random_forest(X_train, y_train, X_test, upper_percentile):

    rf_clf = ensemble.RandomForestClassifier(max_features='sqrt', n_estimators=500, max_depth=4, min_samples_leaf=50, bootstrap=True,
                                             class_weight='balanced', n_jobs=-1, random_state=42)
    model = rf_clf.fit(X_train, y_train)
    
    y_test_prob = model.predict_proba(X_test)[:,1]
    prob_threshold = np.percentile(y_test_prob, upper_percentile)
    y_test_pred_class = (y_test_prob>=prob_threshold).astype(int)

    return y_test_prob, y_test_pred_class, model

# AdaBoost
# ‾‾‾‾‾‾‾‾
def boosting(X_train, y_train, X_test, upper_percentile):

    estimator = DecisionTreeClassifier(max_depth=1,min_samples_leaf=50)
    ab_clf = ensemble.AdaBoostClassifier(estimator=estimator, n_estimators=500, learning_rate=0.03, random_state=42)
    model = ab_clf.fit(X_train, y_train)

    y_test_prob = model.predict_proba(X_test)[:,1]
    prob_threshold = np.percentile(y_test_prob, upper_percentile)
    y_test_pred_class = (y_test_prob>=prob_threshold).astype(int)

    return y_test_prob, y_test_pred_class, model

# XGBoost
# ‾‾‾‾‾‾‾
def xgboost(X_train, y_train, X_test, upper_percentile):

    xgb_clf = XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                            min_child_weight=50, gamma=1.0, reg_alpha=1.0, reg_lambda=2.0, scale_pos_weight=3, n_jobs=-1, random_state=42)
    model = xgb_clf.fit(X_train, y_train)

    y_test_prob = model.predict_proba(X_test)[:,1]
    prob_threshold = np.percentile(y_test_prob, upper_percentile)
    y_test_pred_class = (y_test_prob>=prob_threshold).astype(int)

    return y_test_prob, y_test_pred_class, model



# ______________________________________________
# ==== Running models on the validation set ====
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

#Running random forest
y_val_prob_rf, y_val_pred_rf, model_rf = random_forest(X_train, y_train, X_val, upper_percentile)

#Running AdaBoost
y_val_prob_ab, y_val_pred_ab, model_ab = boosting(X_train, y_train, X_val, upper_percentile)

#Running XGBoost
y_val_prob_xgb, y_val_pred_xgb, model_xgb = xgboost(X_train, y_train, X_val, upper_percentile)

#Storing and comparing model metrics
models = ['Random Forest', 'AdaBoost', 'XGBoost']
predictions = [y_val_pred_rf, y_val_pred_ab, y_val_pred_xgb]
predictions_prob = [y_val_prob_rf, y_val_prob_ab, y_val_prob_xgb]
val_metrics = {}

for i in range(0,3):
    prediction = predictions[i]
    prediction_prob = predictions_prob[i]
    method = models[i]

    val_metrics[i] = {
        'ML Model':             method,
        'Accuracy':             np.round(accuracy_score(y_val, prediction),4),
        'Precision':            np.round(precision_score(y_val, prediction),4),
        'ROC-AUC':              np.round(roc_auc_score(y_val, prediction_prob),4,),
        'F1-Score':             np.round(f1_score(y_val, prediction),4),
        'Confusion Matrix':     confusion_matrix(y_val, prediction)
    }

# Converting metrics dictionary into lists for comparison graphs later
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
orders = list(val_metrics.keys())

accuracies = [val_metrics[o]["Accuracy"] for o in orders]
precision_scores = [val_metrics[o]["Precision"] for o in orders]
roc_aucs = [val_metrics[o]["ROC-AUC"] for o in orders]
f1_scores = [val_metrics[o]["F1-Score"] for o in orders]

val_metrics_df = pd.DataFrame(val_metrics).T

# __________________________
# ==== Trading Strategy ====
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

# Model metrics (precision, ROC-AUC etc.) are a good measure of model performance, but to choose the best model...
# trading strategy performance on the validation set will be compared. This model will then be used on the test set.


# Defining call price
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
def call_price(S, K, r, T, sigma):
    if T <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S/K) + (r+(sigma**2/2)*T)) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    C = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    return C

# Defining put price
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
def put_price(S, K, r, T, sigma):
    if T <= 0:
        return max(K - S, 0)

    d1 = (np.log(S/K) + (r+(sigma**2/2)*T)) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    P = -S*norm.cdf(-d1) + K*np.exp(-r*T)*norm.cdf(-d2)

    return P


# Running straddle strategy
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
def run_straddle_strategy(test, regime_pred, TP, SL, timeout, model):

    prices = test['Close'].values
    vol = test['20_day_volatility'].values * np.sqrt(252)
    dates = test.index

    num_trades = 0
    num_wins = 0
    profit_trades = 0
    loss_trades = 0
    trade_results = {}

    entry_array = np.zeros(len(test))
    exit_array = np.zeros(len(test))
    trade_arrays = np.zeros((len(test),2))

    r = 0.02
    initial_capital = 10000
    capital = initial_capital

    equity_curve = []
    strategy_returns = np.zeros(len(test))
    in_position = False

    option_value = 0
    K = 0
    T_total = expiry/252 # N day expiry as defined above
    T_remaining = T_total
    T_timeout = T_total - timeout/252

    for i in range(1,len(test)):

        regime = regime_pred[i]
        price = prices[i]
        vol_current = vol[i]
        date = dates[i]

        # Entry conditions
        if regime==1 and in_position == False:

            T_remaining = T_total
            entry_date = date
            entry_price = price
            K = np.round(price*2)/2

            call = call_price(price, K, r, T_remaining, vol_current)
            put = put_price(price, K, r, T_remaining, vol_current)

            call_initial, put_initial = call, put

            option_value = call + put
            prev_option_value = option_value

            entry_array[i] = 1           

            in_position = True

        # Position tracking
        elif in_position:
            T_remaining -= 1/252

            call = call_price(price, K, r, T_remaining, vol_current)
            put = put_price(price, K, r, T_remaining, vol_current)
            
            option_value_new = call + put
            gain = 100*(option_value_new - option_value)/option_value
            strategy_returns[i] = ((option_value_new - prev_option_value) / prev_option_value)

            prev_option_value = option_value_new # Updating this for the next period

            # Exit conditions
            if gain>=TP or gain<=-SL or T_remaining<=T_timeout: #or regime==0:
                exit_date = date
                exit_price = price

                exit_array[i] = 1

                num_trades += 1
                if gain>0: 
                    num_wins+=1
                    profit_trades += gain
                else:
                    loss_trades += gain

                # Logging trade details
                trade_results[num_trades] = pd.DataFrame({
                    "Entry Date":           [entry_date],
                    "Exit Date":            [exit_date],
                    "Stock Entry Price":    [entry_price],
                    "Stock Exit Price":     [exit_price],
                    "Strike Price":         [K],
                    "Options Entry Price":  [option_value],
                    "Options Exit Value":   [option_value_new],
                    "Gain %":               [gain],
                    "Call Entry Price":     [call_initial],
                    "Put Entry Price":      [put_initial],
                    "Call Exit Value":      [call],
                    "Put Exit Value":       [put]
                })

                in_position = False

    # Calculating equity curve
    equity_curve = (np.cumprod(strategy_returns+1) - 1)*100
    
    # Calculating final gain and win rate
    final_gain = equity_curve[-1]
    win_rate = np.round(100*num_wins/num_trades,2)

    # Computing Sharpe ratio
    excess_returns = strategy_returns - risk_free_rate/annualisation
    sharpe = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(annualisation)
    excess_returns_df = pd.DataFrame(excess_returns)
    rolling_sharpe = np.array((excess_returns_df.rolling(sharpe_window).mean() / excess_returns_df.rolling(sharpe_window).std()) 
                              * np.sqrt(annualisation))

    # Computing Max Drawdown
    running_max = np.maximum.accumulate(equity_curve/100 +1)
    drawdown = 100 * ((1+ equity_curve/100) - running_max) / running_max
    max_drawdown = np.min(drawdown)

    # Profit factor
    profit_factor = profit_trades / np.abs(loss_trades)

    # Sortino ratio
    negative_returns = strategy_returns[strategy_returns < 0]
    if len(negative_returns) > 0:
        sortino_ratio = (np.mean(excess_returns) / np.std(negative_returns)) * np.sqrt(annualisation)
    else: 
        sortino_ratio = np.nan

    # Entry and exit arrays
    trade_arrays[:,0] = entry_array
    trade_arrays[:,1] = exit_array

    # Compiling strategy metrics into dataframe
    strategy_metrics = pd.DataFrame({
        "ML Model":         model,
        "Num Trades":       [num_trades],
        "Win Rate %":       [np.round(win_rate,2)],
        "Total Gain %":     [np.round(final_gain,2)],
        "Profit Factor":    [np.round(profit_factor,2)],
        "Sharpe Ratio":     [np.round(sharpe,2)],
        "Sortino Ratio":    [np.round(sortino_ratio,2)],
        "Max Drawdown %":   [np.round(max_drawdown,2)],
    })            

    return equity_curve, drawdown, trade_results, strategy_metrics, rolling_sharpe, strategy_returns, trade_arrays

# ___________________________________________________________________
# ==== Running strategy on validation set using all three models ====
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

#Random Forest
equity_val_rf, drawdown_rf, trade_results_val_rf, strategy_metrics_val_rf, sharpe_rf, returns_rf, arrays_rf = run_straddle_strategy(
    validation, y_val_pred_rf, TP, SL, timeout, model='Random Forest')
#AdaBoost
equity_val_ab, drawdown_ab, trade_results_val_ab, strategy_metrics_val_ab, sharpe_ab, returns_ab, arrays_ab = run_straddle_strategy(
    validation, y_val_pred_ab, TP, SL, timeout, model='AdaBoost')
#XGBoost
equity_val_xgb, drawdown_xgb, trade_results_val_xgb, strategy_metrics_val_xgb, sharpe_xgb, returns_xgb, arrays_xgb = run_straddle_strategy(
    validation, y_val_pred_xgb, TP, SL, timeout, model='XGBoost')

# Compiling trade metrics for best model selection
trade_metrics_val = pd.concat([strategy_metrics_val_rf, strategy_metrics_val_ab, strategy_metrics_val_xgb], axis=0).reset_index().drop(columns='index')

# Converting equity curves into dataframes for plotting later
equity_val_rf_df = pd.DataFrame(equity_val_rf, index=validation.index)
equity_val_ab_df = pd.DataFrame(equity_val_ab, index=validation.index)
equity_val_xgb_df = pd.DataFrame(equity_val_xgb, index=validation.index)


# Best model selection
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

# Selects the model with the best Sharpe ratio (selection_metric). If all Sharpe ratios are negative, it uses ROC-AUC (selection_metric_secondary).
if override_model_selection==False:
    if trade_metrics_val[selection_metric].max() > 0:
        best_model_index = trade_metrics_val[selection_metric].idxmax()
        best_model = models[best_model_index]
    else:
        best_model_index = val_metrics_df[selection_metric_secondary].idxmax()
        best_model = models[best_model_index]
else:
    best_model = trading_model
    if trading_model == "Random Forest": best_model_index=0
    elif trading_model == "AdaBoost": best_model_index=1
    else: best_model_index=2

cm = pd.DataFrame(val_metrics[best_model_index]["Confusion Matrix"]) # Confusion matrix for best model on validation set



# ______________________________________________________________
# ==== Retraining best model and running it on the test set ====
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

# Concatenating training and validation sets
X_train_val = np.concatenate((X_train, X_val), axis=0)
y_train_val = np.concatenate((y_train, y_val), axis=0)

# Retraining model using combined training and validation data
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

# Only retrains on combined training and validation set if 'retrain' is set to True.
if best_model=='Random Forest':
    if retrain:    
        y_test_prob, y_test_pred_class, test_model = random_forest(X_train_val, y_train_val, X_test, upper_percentile)
    else:
        y_test_prob, y_test_pred_class, test_model = random_forest(X_train, y_train, X_test, upper_percentile)
elif best_model=='AdaBoost':
    if retrain:    
        y_test_prob, y_test_pred_class, test_model = boosting(X_train_val, y_train_val, X_test, upper_percentile)
    else:
        y_test_prob, y_test_pred_class, test_model = boosting(X_train, y_train, X_test, upper_percentile)
elif best_model=='XGBoost':
    if retrain:    
        y_test_prob, y_test_pred_class, test_model = xgboost(X_train_val, y_train_val, X_test, upper_percentile)
    else:
        y_test_prob, y_test_pred_class, test_model = xgboost(X_train, y_train, X_test, upper_percentile)


# Test Performance Metrics
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
cm_test = confusion_matrix(y_test, y_test_pred_class)

test_metrics_df = pd.DataFrame({
    'ML Model':             best_model,
    'Accuracy':             np.round(accuracy_score(y_test, y_test_pred_class),4),
    'Precision':            np.round(precision_score(y_test, y_test_pred_class),4),
    'ROC-AUC':              np.round(roc_auc_score(y_test, y_test_prob),4),
    'F1-Score':             np.round(f1_score(y_test, y_test_pred_class),4),
    'Confusion Matrix':     [cm_test]
})


# Running trading strategy on test set
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
regime_array = y_test_pred_class
actual_regime = y_test.flatten()

equity_curve, drawdown, trade_results, strategy_metrics, rolling_sharpe, strategy_returns, trade_arrays = run_straddle_strategy(
    test, y_test_pred_class, TP, SL, timeout, best_model)


# Formatting metrics for easy plotting
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

# Strategy equity curve
equity_curve_df = pd.DataFrame(equity_curve, index=test.index)
drawdown_df = pd.DataFrame(drawdown, index=test.index)

# Buy and hold returns
close_array = np.array(test['Close'])
daily_returns = np.insert((np.diff(close_array) / close_array[:-1]), 0, 0)
buy_hold_equity_df = pd.DataFrame(((np.cumprod(daily_returns + 1) - 1) * 100), index=test.index)  

# Rolling Sharpe ratio
rolling_sharpe_df = pd.DataFrame(rolling_sharpe, index=test.index)
rolling_sharpe = rolling_sharpe.flatten()

# Returns vs probability bucket
test.loc[:,"future_return"] = strategy_returns
test.loc[:,"prob_bucket"] = pd.cut(y_test_prob, bins=10)
bucket_returns = test.groupby("prob_bucket", observed=True)["future_return"].mean()
bucket_returns = bucket_returns[bucket_returns != 0]
bucket_mid = [interval.mid for interval in bucket_returns.index]
bucket_counts = test.groupby("prob_bucket", observed=True).size().loc[bucket_returns.index]

# Feature importances
feature_names = asset_features.columns
perm_importances = permutation_importance(estimator=test_model, X=X_train, y=y_train, n_repeats=10, n_jobs=2, random_state=42)
sorted_idx = perm_importances.importances_mean.argsort()

# Information Coefficient (IC)
test.loc[:,"predicted_prob"] = y_test_prob
ic = test["predicted_prob"].corr(test["future_return"])
rolling_ic = (test["predicted_prob"].rolling(50).corr(test["future_return"]))
cum_ic = rolling_ic.cumsum()

# Entry and exit arrays for trade tracking
entry_array = np.where(trade_arrays[:,0] == 1, 1, np.nan)
exit_array = np.where(trade_arrays[:,1] == 1, 1, np.nan)

entry_idx = np.where(entry_array == 1)[0]
exit_idx = np.where(exit_array == 1)[0]

entry_dates = test.index[entry_idx]
exit_dates = test.index[exit_idx]



# _________________________
# ==== Plotting Graphs ====
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

dates = test.index
price = test['Close'].values

# ==== Prediction interpretation plots ====
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

fig, axes = plt.subplots(2, 2, figsize=(16, 9), gridspec_kw={"width_ratios":[5,2]})

# Predicted vs actual high vol with price
axes[0,0].plot(dates, price)
axes[0,0].fill_between(dates,
                      price,price.max(),
                      where=(actual_regime==1),
                      color='blue',
                      label='Actual High Future Return',
                      alpha=0.2)
axes[0,0].fill_between(dates,
                      price,price.min(),
                      where=(regime_array==1),
                      color='red',
                      label='Predicted High Future Return',
                      alpha=0.2)
axes[0,0].set_title(f'{ticker} Price and High Volatility Regions in the Test Set')
axes[0,0].legend(loc='upper left')
axes[0,0].set_ylabel(f"{ticker} Close Price")
axes[0,0].set_xlabel("Date")

# Equity curve comparison of ML models on validation set
axes[1,0].plot(equity_val_rf_df, label='Random Forest')
axes[1,0].plot(equity_val_ab_df, color='orange', label='AdaBoost')
axes[1,0].plot(equity_val_xgb_df, color='green', label='XGBoost')
axes[1,0].set_ylabel('Absolute % Return')
axes[1,0].set_xlabel('Date')
axes[1,0].set_title('ML Equity Curve Comparison on Validation Set')
axes[1,0].legend(loc='upper left')
axes[1,0].grid(True)

# Probability distribution histogram
high_vol_probs = y_test_prob[y_test == 1]
low_vol_probs  = y_test_prob[y_test == 0]

axes[0,1].hist(low_vol_probs, bins=30, alpha=0.5, label="Actual Low Vol")
axes[0,1].hist(high_vol_probs, bins=30, alpha=0.5, label="Actual High Vol")
axes[0,1].set_title("Test Set Probability Separation by Actual Regime")
axes[0,1].set_xlabel("Predicted Probability")
axes[0,1].set_ylabel("Frequency")
axes[0,1].legend()

# Returns by probability bucket
axes[1,1].plot(bucket_mid, bucket_returns.values * 100, marker='o')
axes[1,1].set_title("Returns by Probability Bucket")
axes[1,1].set_ylabel("Mean Percentage Return")
axes[1,1].set_xlabel("Predicted Probability")
axes[1,1].axhline(0, color='grey')
axes[1,1].fill_between(bucket_mid, bucket_returns.values, 0, alpha=0.3)
#ax2 = axes[1,1].twinx()
#ax2.bar(bucket_mid, bucket_counts.values, alpha=0.3, width=0.05)

# Plot formatting
plt.subplots_adjust(
    left=0.06,
    right=0.95,
    top=0.95,
    bottom=0.14,
    hspace=0.31,
    wspace=0.19
)


# ==== Signal plots ====
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

fig, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios":[3,5]})

# Feature importances
axes[0].boxplot(perm_importances.importances[sorted_idx].T, vert=False, tick_labels=feature_names[sorted_idx])
axes[0].set_title("Feature Importances on Test Set")

# Cumulative information coefficient
axes[1].plot(dates, cum_ic.values, color='orange')
axes[1].set_title("Cumulative Information Coefficient on Test Set")
axes[1].set_xlabel("Date")
axes[1].fill_between(dates, cum_ic.values, 0, color='orange', alpha=0.15)

# Plot formatting
plt.subplots_adjust(
    left=0.115,
    right=0.925,
    top=0.9,
    bottom=0.15
)

# ==== Trading Strategy Plots ====
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

fig, axes = plt.subplots(5, 1, figsize=(16, 9), gridspec_kw={"height_ratios":[4,2,2,0.5,1]}) #sharex=True

# Strategy Returns vs buy-and-hold returns
axes[0].plot(equity_curve_df, label="ML Options Equity Curve", color="blue")
axes[0].plot(buy_hold_equity_df, label="Buy & Hold Equity Curve", color='green')
axes[0].set_title(f"ML Straddle Strategy Performance on {ticker}")
axes[0].set_ylabel("Absolute % Return")
axes[0].tick_params(labelbottom=False)
axes[0].fill_between(dates,
                      equity_curve_df.max(),equity_curve_df.min(),
                      where=(regime_array==1),
                      color='blue',
                      label='Predicted High Vol',
                      alpha=0.15)
axes[0].grid(True)
    # Trade entry/exit markers
axes[0].scatter(entry_dates, equity_curve[entry_idx], marker='^', label='Trade Entry', color='#0096FF')
axes[0].scatter(exit_dates, equity_curve[exit_idx], marker='v', label='Trade Exit', color='red', alpha=0.7)
axes[0].legend(loc='upper left')

# Drawdown plot
axes[1].plot(drawdown_df, label="Drawdown", color='orange')
axes[1].set_ylabel("% Drawdown")
axes[1].sharex(axes[0])
axes[1].legend(loc='lower left')
axes[1].grid(True)
axes[1].fill_between(drawdown_df.index, drawdown, 0, color='orange', alpha=0.15)
axes[0].tick_params(labelbottom=False)

# Rolling Sharpe
axes[2].plot(rolling_sharpe_df, label="Rolling Sharpe Ratio", color='red', alpha=0.7)
axes[2].set_ylabel("Sharpe Ratio")
axes[2].sharex(axes[0])
axes[2].legend(loc='upper left')
axes[2].grid(True)
axes[2].fill_between(rolling_sharpe_df.index, rolling_sharpe, 0, color='red', alpha=0.15)

# Empy space to separate graphs from table
axes[3].axis('off')

# Strategy metrics table
axes[4].axis('off')  # hide axes
table = axes[4].table(
    cellText=strategy_metrics.round(3).values,
    colLabels=strategy_metrics.columns,
    rowLabels=strategy_metrics.index,
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# Plot formatting
plt.subplots_adjust(
    left=0.13,
    right=0.87,
    top=0.95,
    bottom=0.085,
    hspace=0.03
)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "lines.linewidth": 1.8
})


# ==== ML Model Plots ====
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

fig, axes = plt.subplots(2, 2, figsize=(16, 9))

# Validation confusion matrix
sn.heatmap(cm, annot=True, fmt="d", ax=axes[1,0])
axes[1,0].set_title(f"Confusion Matrix for {best_model} Model on Validation Set")
axes[1,0].set_ylabel("Actual Value")
axes[1,0].set_xlabel("Predicted Value")

# Test confusion matrix
sn.heatmap(cm_test, annot=True, fmt="d", ax=axes[1,1])
axes[1,1].set_title(f"Confusion Matrix for {best_model} Model on Test Set")
axes[1,1].set_ylabel("Actual Value")
axes[1,1].set_xlabel("Predicted Value")

# Probability distribution histogram
prob_threshold_test = np.percentile(y_test_prob, upper_percentile)

axes[0,1].hist(y_test_prob, bins=30)
axes[0,1].set_title("Prediction Probability Distribution for Test Set")
axes[0,1].set_ylabel('Frequency')
axes[0,1].axvline(prob_threshold_test, color='red', linestyle='--', linewidth=1, label="Upper Probability Threshold")
axes[0,1].legend()

# Model performance metrics
axes[0,0].plot(models, precision_scores, label="Precision",  marker='o')
axes[0,0].plot(models, roc_aucs, label="ROC-AUC",  marker='o')
axes[0,0].plot(models, accuracies, label="Accuracy",  marker='o')
axes[0,0].plot(models, f1_scores, label="F1 Score",  marker='o')
axes[0,0].set_title("Model Metrics for Decision Tree Classifiers on the Validation Set")
axes[0,0].legend(["Precision", "ROC-AUC", "Accuracy", "F1 Score"])
axes[0,0].grid(True)

# Plot formatting
plt.subplots_adjust(
    left=0.08,
    right=0.96,
    top=0.95,
    bottom=0.15,
    hspace=0.3
)

# Printing model metric dataframes
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

print('\nPerformance metrics for ML models on the validation set: \n\n', val_metrics_df, "\n\n", trade_metrics_val, "\n\n")
print(f'\nPerformance metrics for {best_model} on the test set:\n\n', test_metrics_df, "\n\n", strategy_metrics, "\n\n")

if trade_to_csv:
    trade_results.to_csv('ML Decision Tree Trade Results.csv', index=True)

plt.show()
