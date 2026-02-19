
# ==== Return Direction Probability using ML ====

# Input variables: 5-day log return, 14 day RSI, 20 day volatility, distance from 20 day MA
# Target variable: positive or negative return after N days, y = 1 if positive, y = 0 otherwise (Binary classification)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import matplotlib
matplotlib.use('Qt5Agg')

from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, fbeta_score, confusion_matrix, classification_report, average_precision_score
import seaborn as sn

# ==== Importing price data from yfinance ====

ticker = 'MSFT'             #Input ticker symbol
date_start = '2022-03-27'   #Input the start date in the form YYYY-MM-DD
timeframe = '1D'            

#asset = yf.download(ticker.upper(), period='2y', interval = timeframe.lower(), 
#                    auto_adjust=True, threads=False )

stock = yf.Ticker(ticker)
asset = stock.history(period="5y", auto_adjust=True)

# ==== Setting up input parameters ====

#Number of days over which return will be predicted
N = 5       

#Probability thresholds
upper_percentile = 70
lower_percentile = 30

#Moving average and distance from MA
asset['20_MA'] = asset['Close'].rolling(window=20).mean().shift() 
asset['MA_Distance'] = (asset['Close'] - asset['20_MA']) / asset['20_MA']

#Log returns
asset['log_return_5d'] = np.log(asset['Close'] / asset['Close'].shift(N))

#20 day volatility
asset['log_return_1d'] = np.log(asset['Close'] / asset['Close'].shift(1))
asset['20_day_volatility'] = asset['log_return_1d'].rolling(20).std()   #Uses 1 day log return

# ==== RSI ==== 
rsi_period = 14

def get_RSI(rsi_period, asset):
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

    return asset

get_RSI(rsi_period,asset)


# ==== Creating target Variable ====
asset["future_return_5d"] = np.log(asset["Close"].shift(-5) / asset["Close"])
asset["target"] = (asset["future_return_5d"] > 0).astype(int)
asset = asset.dropna()

# ==== Splitting data into training, validation, and testing sets ====

data_split = np.array([0.6, 0.15, 0.25]) 

# Defines a function for splitting the dataset
def split_dataframe(df, data_split):
    training_end = int(len(df)*data_split[0])
    val_end = int(training_end + len(df)*data_split[1])
    print(training_end,val_end)
    
    training = df[:training_end]
    validation = df[training_end:val_end]
    test = df[val_end:]

    return training, validation, test

training, validation, test = split_dataframe(asset, data_split)

# Testing to see if the splits are sized correctly and are in order
#print(len(training), len(validation), len(test))

#print(training[-5:-1])
#print(validation[0:5])
#print(validation[-5:-1])
#print(test[0:5])

# ==== Defining input and target variables for the model ====

def convert_to_numpy(df):
    X = df[['log_return_5d','MA_Distance','RSI','20_day_volatility']].to_numpy()
    y = df['target'].to_numpy()

    return X,y

X_train,y_train = convert_to_numpy(training)
X_val,y_val = convert_to_numpy(validation)
X_test,y_test = convert_to_numpy(test)


# ==== Scaling data ====

# Compute scaler on train only
scaler = StandardScaler().fit(X_train)

# Transform train, val, test separately
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ==== Defining model using training data ====

def fit_model(X,y,order):
    model = Pipeline([('poly', PolynomialFeatures(degree=order)),
                      ('logistic', LogisticRegression(fit_intercept=False))])
    model = model.fit(X,y)

    return model

# ==== Running model on validation set ====

#Initialise dictionaries to store the models and predictions
models_dict = {}
predictions_dict = {}
val_metrics = {}

for order in range(1,7):
    models_dict[order] = fit_model(X_train,y_train,order)

    predictions_dict[order] = models_dict[order].predict_proba(X_val)[:,1] #This gives the probability of class 1 (positive return).

    # Convert probabilities to class labels
    y_val_prob = predictions_dict[order]
    prob_threshold_high = np.percentile(y_val_prob, upper_percentile)
    y_val_pred_class = (y_val_prob >= prob_threshold_high).astype(int) #Converts to 1 if p>=threshold, 0 if less than 0.5

    # ==== Evaluation Metrics ====
    
    #Store metrics for each order in a dictionary
    val_metrics[order] = {
        "accuracy": accuracy_score(y_val, y_val_pred_class),
        "roc_auc": roc_auc_score(y_val, y_val_prob),  # Use raw values or probabilities for ROC-AUC
        "f1": f1_score(y_val, y_val_pred_class),
        "cm": confusion_matrix(y_val, y_val_pred_class),
        "precision": precision_score(y_val, y_val_pred_class),
        "f_beta": fbeta_score(y_val, y_val_pred_class, beta=0.5),
        "pr_auc": average_precision_score(y_val, y_val_prob)
    }

# ==== Printing performance metrics ===

# Converting metric dictionary into lists
orders = list(val_metrics.keys())

accuracies = [val_metrics[o]["accuracy"] for o in orders]
precision_scores = [val_metrics[o]["precision"] for o in orders]

roc_aucs = [val_metrics[o]["roc_auc"] for o in orders]
pr_aucs = [val_metrics[o]["pr_auc"] for o in orders]

f1_scores = [val_metrics[o]["f1"] for o in orders]
f_beta_scores = [val_metrics[o]["f_beta"] for o in orders]

metrics_df = pd.DataFrame(val_metrics).T
print(metrics_df)

#print(classification_report(y_val, y_val_pred_class))

# ==== Selecting Model ====

selection_metric = "precision"
possible_orders = [2,3,4,5]
best_order = max(possible_orders, key=lambda o: val_metrics[o][selection_metric])
best_model = models_dict[best_order]
print(f"Selected polynomial degree: {best_order} (best {selection_metric})")

cm = pd.DataFrame(val_metrics[best_order]["cm"])

prob_threshold_high = np.percentile(predictions_dict[best_order], upper_percentile)
prob_threshold_low = np.percentile(predictions_dict[best_order], lower_percentile)

# ==== Applying Model to Test Data ===

y_test_prob = best_model.predict_proba(X_test)[:,1] 
prob_threshold_high_test = np.percentile(y_test_prob, upper_percentile) #Resetting probability thresholds using test data distribution
prob_threshold_low_test = np.percentile(y_test_prob, lower_percentile)
y_test_pred_class = (y_test_prob >= prob_threshold_high_test).astype(int)

#Creating performance metrics
precision_test = precision_score(y_test, y_test_pred_class)
roc_auc_test = roc_auc_score(y_test, y_test_prob)
accuracy_test = accuracy_score(y_test, y_test_pred_class)
cm_test = confusion_matrix(y_test, y_test_pred_class)

test_metrics = {
    "Precision": [precision_test],
    "ROC-AUC": [roc_auc_test],
    "Accuracy": [accuracy_test],
    "cm": [cm_test]
}
test_metrics_df = pd.DataFrame(test_metrics)
print(test_metrics_df)





                                            # ==== Applying Predictions to Trading Strategy ====

test['target'] = y_test_pred_class

signals = np.zeros(len(y_test_prob))
signals[y_test_prob >= prob_threshold_high_test] = 1
signals[y_test_prob <= prob_threshold_low_test] = -1

open_array = np.array(test['Open'])
close_array = np.array(test['Close'])
signal_long, signal_short, signal_exit, position, trade_complete = False, False, False, False, False
num_trades, num_wins = 0, 0

strategy_returns = []
trade_results = {}

daily_returns = np.diff(close_array) / close_array[:-1]

for i in range(len(test)):
    close = close_array[i]
    open = open_array[i]
    y = signals[i]

    #Long entry
    if signal_long==True:
        position = 'long'
        entry_price = open
        entry_date = asset.index[i]
        signal_long = False
    #Short entry
    if signal_short==True:
        position = 'short'
        entry_price = open
        entry_date = asset.index[i]
        signal_short = False
    #Exit
    if signal_exit==True:
        exit_price = open
        exit_date = asset.index[i]
        trade_complete = True
        trade_type = position
        position, signal_exit=False,False

    #Short and long signals
    if position == False and y==1: #Long buy
        signal_long = True
    elif position == False and y==-1: #Short buy
        signal_short = True
    elif (position=='long' and y!=1) or (position=='short' and y!=-1): #Sell signal
        signal_exit = True


    #Trade complete and logging
    if trade_complete==True:
        current_gain = (exit_price-entry_price)/entry_price
        num_trades += 1
        if trade_type=='short':
            current_gain*=-1
        if current_gain>0:
            num_wins+=1
        trade_results[num_trades] = {
            "Entry Date": entry_date,
            "Position": trade_type,
            "Entry Price": entry_price,
            "Exit Date": exit_date,
            "Exit Price": exit_price,
            "Gain": current_gain
        }
        trade_complete=False


# ==== Computing perforance metrics ====

#Converting trade log to dataframe
trade_results_df = pd.DataFrame(trade_results).T
print(trade_results_df)

#Computing strategy returns and buy and hold returns
strategy_returns = daily_returns * signals[:-1]
equity_curve = (np.cumprod(1+strategy_returns))

total_return = (equity_curve-1) *100
buy_hold_return = 100*(open_array[-1] - open_array[0])/open_array[0]


#Winrate
winrate = (num_wins/num_trades)*100

#Computing Sharpe ratio and max drawdown
def compute_sharpe(returns, risk_free_rate=0.04):
    excess_returns = returns - risk_free_rate/252
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe * np.sqrt(252)

def compute_max_drawdown(equity_curve):
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = 100 * (equity_curve - running_max) / running_max
    max_drawdown = np.min(drawdown)
    return max_drawdown, drawdown

sharpe_ratio = compute_sharpe(strategy_returns)
max_dd, dd_series = compute_max_drawdown(equity_curve)

#Collating strategy metrics into a dataframe

strategy_metrics = pd.DataFrame({
    "Total Return %": [np.round(total_return[-1],2)],
    "Win Rate %": [np.round(winrate,2)],
    "Number of Trades": [num_trades],
    "Sharpe Ratio": [sharpe_ratio],
    "Max Drawdown %": [np.round(max_dd,2)]
})

print(strategy_metrics)


# ==== Trading Strategy Plots ====
fig, axes = plt.subplots(3, 1, figsize=(9, 7))

#Strategy Returns
axes[0].plot(equity_curve, label="ML Strategy Equity Curve")
axes[0].plot(np.cumprod(1 + daily_returns), label="Buy & Hold Equity Curve")
axes[0].set_title(f"ML Strategy vs Buy & Hold on {ticker}")
axes[0].grid(True)
axes[0].legend()

#Drawdown
axes[1].plot(dd_series)
axes[1].set_title("Drawdown")
axes[1].grid(True)

# Metrics table
axes[2].axis('off')  # hide axes

table = axes[2].table(
    cellText=strategy_metrics.round(3).values,
    colLabels=strategy_metrics.columns,
    rowLabels=strategy_metrics.index,
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

#axes[2].set_title("Strategy Performance Metrics")
plt.tight_layout()

# ==== ML Model Plots ====
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

#Validation Confusion Matrix
sn.heatmap(cm, annot=True, fmt="d", ax=axes[1,0])
axes[1,0].set_title(f"Confusion Matrix for order {best_order} on validation set")
axes[1,0].set_ylabel("Actual Value")
axes[1,0].set_xlabel("Predicted Value")

#Test Confusion Matrix
sn.heatmap(cm_test, annot=True, fmt="d", ax=axes[1,1])
axes[1,1].set_title(f"Confusion Matrix for test set")
axes[1,1].set_ylabel("Actual Value")
axes[1,1].set_xlabel("Predicted Value")

#Probability Histogram
axes[0,1].hist(y_val_prob, bins=30)
axes[0,1].set_title("Prediction Probability Distribution for Validation Set")
axes[0,1].axvline(prob_threshold_high, color='red', linestyle='--', linewidth=1, label="Upper Probability Threshold")
axes[0,1].axvline(prob_threshold_low, color='blue', linestyle='--', linewidth=1, label="Lower Probability Threshold")
axes[0,1].legend()

#Model performance metrics
axes[0,0].plot(orders, precision_scores, label="Precision",  marker='o')
axes[0,0].plot(orders, pr_aucs, label="PR-AUC",  marker='o')
axes[0,0].plot(orders, accuracies, label="Accuracy",  marker='o')
axes[0,0].plot(orders, f_beta_scores, label="F-Beta",  marker='o')
axes[0,0].set_title("Model Metrics vs Polynomial Degree")
axes[0,0].legend(["Precision", "PR-AUC", "Accuracy", "F-Beta"])
axes[0,0].grid(True)

plt.tight_layout()
plt.show()