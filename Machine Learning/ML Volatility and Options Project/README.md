# **Machine Learning for Volatility and Options Strategies**

This project explores the use of machine learning models to predict market behaviour and volatility regimes, and to test whether these predictions can be translated into systematic trading strategies, particularly options straddle strategies.

The repository documents the progression of the project through four stages, starting from a simple directional market prediction model and evolving into a framework designed to predict large future price movements, which are more directly aligned with options strategy profitability.

The goal is to investigate whether machine learning can identify conditions under which trading strategies, such as option straddles, may produce positive expected returns.


## **Project Structure**

The project is organized into four scripts, each representing a stage in the research process.

- 1_direction_prediction_logistic.py
- 2_volatility_prediction_logistic.py
- 3_volatility_prediction_decision_tree.py
- 4_future_move_prediction_straddle.py

Each script builds on the previous work by improving the target definition, feature engineering, modelling approach, and trading strategy design.

The main focus should be on the final two scripts, which are more polished and represent a much clearer understanding of utilising machine learning for trading applications. 

## **1. Market Direction Prediction (Logistic Regression)**

Script:\
1_direction_prediction_logistic.py

The first stage of the project focuses on predicting short-term market direction using logistic regression.

### **Objective:**

Predict whether the asset price will increase over a specified future horizon.

### **Target Variable:**

Binary classification:

Target = 1 if future return > 0
Target = 0 otherwise

### **Features:**

Technical indicators commonly used in systematic trading:

- Relative Strength Index (RSI)
- Rolling volatility
- Rolling log returns
- Price distance from moving averages

### **Model:**

Logistic Regression classifier, using polynomial feature expansion.

### **Strategy:**

A simple directional strategy was tested:

Long position when the predicted probability of an upward move exceeds an upper threshold, and a short position when the predicted probability of an upwards movement is below a lower threshold.

Flat position otherwise.

This stage served primarily as an initial baseline for model evaluation and pipeline development.

## **2. Volatility Prediction (Logistic Regression)**

Script:\
2_volatility_prediction_logistic.py

The second stage shifts the focus from direction prediction to volatility prediction, with the intention of implementing option straddle strategies that profit from voaltility.

This change was motivated by the observation that directional prediction is often difficult in financial markets, whereas predicting volatility regimes can be more tractable, and can also an extremely useful tool generally, not just for directly generating trading signals.

### **Objective:**

Predict whether future volatility will be high relative to historical levels.

### **Target Variable:**

Future 20-day realized volatility is computed, and a high-volatility regime is defined using an expanding percentile threshold.

Target = 1 if future volatility ≥ expanding 70th percentile\
Target = 0 otherwise

Using an expanding threshold ensures that the definition of "high volatility" adapts to changing market regimes over time.

### **Features:**

- Average True Range (ATR)
- Average Directional Index (ADX)
- Rolling realized volatility
- Volatility ratios

### **Strategy:**

An options straddle strategy is introduced:

Since the scripts import market data using the Yahoo Finance API, I was unfortunately not able to retrieve historical options data - this would require the use of a different API.

As a result, I decided the next best option (pun not intended) was to simulate synthetic options, and apply continual repricing using the Black-Scholes formula.

When high volatility is predicted → enter long straddle

Otherwise → no trade

Exit → when either: profit target is hit; stoploss is hit; after a specified amount of time (to avoid theta decay)

Straddles profit from large price movements regardless of direction, making them a natural fit for volatility predictions.

## **3. Volatility Prediction with Decision Trees**

Script:
3_volatility_prediction_decision_tree.py

The third stage improves the modelling framework by replacing logistic regression with decision tree based models.

### **Motivation:**

During this period, I had just learned about decision tree ensemble techniques through my professional certificate program, and wanted to test whether they offered superior predictive power over logistic regression in this instance.
Additionally, financial markets exhibit nonlinear relationships and regime-dependent behaviour.

Tree-based models are better suited for capturing interactions such as:

low volatility AND increasing trend strength → volatility expansion

### **Model:**

This itereration uses three types of decision tree classifiers:
- Random Forests
- AdaBoost
- XGBoost

Each model is trained using the training set and used to make predictions on the validation set, which generates a set of trading signals. The model with the best overall trading performance on the validation set is chosen to be used on the test set.

Tree models allow the algorithm to learn rule-based structures that resemble common technical trading logic.

Example learned rule:

IF volatility_percentile < 0.2
AND ADX_slope > 0
THEN high probability of volatility expansion

### **Strategy:**

The long straddle strategy is retained, but now uses the improved model predictions.

## **4. Predicting Future Price Move Magnitude (Straddle Alpha)**

Script:
4_future_move_prediction_straddle.py

The final stage of the project reframes the problem to focus directly on straddle profitability.
From the previous stage, I noticed that the target logic was not classifying bull or bear runs as high volatility periods, which meant missed opportunities for large gains. I learned that volatility measures variance in returns, which can technically be low during a bull or bear runs (e.g. if daily returns are +3%, +3.5%, +3.2%).

Instead of predicting volatility levels, this model predicts future absolute price movement, which is more directly linked to options payoffs.

### **Target Variable:**

Future absolute return over a fixed horizon:

**Target = |future return over 15 days|**

A classification threshold is then applied to identify large moves.

This formulation aligns closely with the payoff structure of straddles.

### **Features:**

The feature set focuses on identifying conditions that precede volatility expansion, including:

- ADX
- ADX slope
- Rate of change (ROC)
- ATR
- ATR compression
- Realised volatility
- RSI
- Bollinger Band width
- Range indicators
- Price distance from moving avergae
  
These features help capture market compression, momentum buildup, and regime transitions, which are common precursors to large price movements.

### **Strategy:**

This version of the script still focuses on long straddles, profiting when predicted future return magnitude is large.
I have created a version of the script that incorporates long and short straddles, which can profit from volatility expansions and contractions.
This is still in development and will require slightly more testing and adaptation, however the short straddle logic has been integrated into the strategy and is working correctly.


## **Options Pricing Framework**

Options are priced using the Black-Scholes model, with the following considerations:

- Options are entered with a fixed time to expiry.
- Time-to-expiry is reduced as the trade progresses, introducing theta decay.
- Volatility inputs are derived from historical realized volatility.
- The strike price for both calls an puts are set to the nearest $0.5 (relative to the asset's close price)

This framework allows the backtest to approximate realistic option pricing dynamics while remaining computationally tractable.

## **Indicators and Trade Logic**

All technical indicator are generated within the script, derived from the imported price data, instead of imported. Creating these were honestly some of the most challenging parts of the project since they often involve complex, multi-stage calculations, and it is important that all data correctly lines up within the dataframe (i.e. RSI at t = 20 must be in row 20). I used websites, such as Investopedia, to find out how these indicators are calculated.

The most complex indicators to create were the RSI, ATR, and ADX indicators, which required a lot of testing to make sure that each stage of the calculation was being executed correctly. This often involved exporting the dataframe into Excel and manually validating the calculations and alignment. However, once the logic was solid it could easily be copied into another script without the need to repeat this process.

I created all of the trade logic myself, drawing on the experience gained from creating many strategy backtesting engines. The core principles are the same, with the addition of Black-Scholes for option pricing. For all of these scripts, every single trade is logged and added to a dataframe, recording the entry and exit dates, extry and exit prices, percentage gain and whether the trade was long or short (if relevant), which can then be exported into Excel for validation purposes. 

## **Key Insights from the Project**

Several insights emerged during development:

- Directional market prediction is difficult, even with machine learning.
- Predicting volatility regimes can be more feasible.
- Tree-based models capture nonlinear financial relationships better than linear models.
- Options strategies benefit from predicting large moves rather than direction.
- Volatility compression features are often strong predictors of future expansion.

## **Future Improvements**

Potential extensions to the project include:

- Incorporating implied volatility data from real options markets.
- Implementing delta-hedged straddle strategies.
- Improving transaction cost modelling.
- Expanding the dataset across multiple assets classes.
- Implementing hyperparamter optimisation (such as grid search).

## **Overall conclusion:**

I believe that machine learning can be an extremely useful tool for quantitative and systematic trading strategy development. There is clearly a lot of potential in harnessing predictive power, however it requires careful tuning and attention to detail when working with large data sets. Since I was only able to import five years worth of daily price data, the validation and test sets ended up only having a few hundred data points. I think with more available market data and therefore a larger data set(potentially looking at hourly data over a similar period), the ability to more reliably train and test these ML models increases massively.

Machine learning clearly has huge potential in this realm, but needs to be handled carefully. Perhaps its reliability as a standalone signal could be questioned, but I think that if used as a filter in conjunction with other tools it can be extremely powerful.

## **Technologies Used**

**Python:** NumPy, Pandas, Scikit-learn, Matplotlib, yfinance, XGBoost, seaborn, scipy

## **Results and Graphs:**

