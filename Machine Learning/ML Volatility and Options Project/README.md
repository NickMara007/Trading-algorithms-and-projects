**Machine Learning for Volatility and Options Strategies**

This project explores the use of machine learning models to predict market behaviour and volatility regimes, and to test whether these predictions can be translated into systematic trading strategies, particularly options straddle strategies.

The repository documents the progression of the project through four stages, starting from a simple directional market prediction model and evolving into a framework designed to predict large future price movements, which are more directly aligned with options strategy profitability.

The goal is to investigate whether machine learning can identify conditions under which trading strategies, such as option straddles, may produce positive expected returns.



**Project Structure**

The project is organized into four scripts, each representing a stage in the research process.

1_direction_prediction_logistic.py
2_volatility_prediction_logistic.py
3_volatility_prediction_decision_tree.py
4_future_move_prediction_straddle.py

Each script builds on the previous work by improving the target definition, feature engineering, modelling approach, and trading strategy design.

1. Market Direction Prediction (Logistic Regression)

Script:
1_direction_prediction_logistic.py

The first stage of the project focuses on predicting short-term market direction using logistic regression.

Objective

Predict whether the asset price will increase over a specified future horizon.

Target Variable

Binary classification:

Target = 1 if future return > 0
Target = 0 otherwise
Features

Technical indicators commonly used in systematic trading:

ADX (trend strength)

ATR (volatility measure)

Rolling volatility

Momentum indicators

Price distance from moving averages

Model

Logistic Regression classifier.

Strategy

A simple directional strategy was tested:

Long position when the predicted probability of an upward move exceeds a threshold.

Flat position otherwise.

This stage served primarily as an initial baseline for model evaluation and pipeline development.

2. Volatility Prediction (Logistic Regression)

Script:
2_volatility_prediction_logistic.py

The second stage shifts the focus from direction prediction to volatility prediction.

This change was motivated by the observation that directional prediction is often difficult in financial markets, whereas predicting volatility regimes can be more tractable.

Objective

Predict whether future volatility will be high relative to historical levels.

Target Variable

Future 20-day realized volatility is computed, and a high-volatility regime is defined using an expanding percentile threshold.

Target = 1 if future volatility ≥ expanding 70th percentile
Target = 0 otherwise

Using an expanding threshold ensures that the definition of "high volatility" adapts to changing market regimes over time.

Features

ATR

Rolling realized volatility

Volatility percentiles

ADX

Momentum indicators

Bollinger Band width (volatility compression)

Strategy

An options straddle strategy is introduced:

When high volatility is predicted → enter long straddle

Otherwise → no trade

Straddles profit from large price movements regardless of direction, making them a natural fit for volatility predictions.

3. Volatility Prediction with Decision Trees

Script:
3_volatility_prediction_decision_tree.py

The third stage improves the modelling framework by replacing logistic regression with decision tree based models.

Motivation

Financial markets exhibit nonlinear relationships and regime-dependent behaviour.
Tree-based models are better suited for capturing interactions such as:

low volatility AND increasing trend strength → volatility expansion
Model

Decision Tree classifiers.

Tree models allow the algorithm to learn rule-based structures that resemble common technical trading logic.

Example learned rule:

IF volatility_percentile < 0.2
AND ADX_slope > 0
THEN high probability of volatility expansion
Strategy

The long straddle strategy is retained, but now uses the improved model predictions.

4. Predicting Future Price Move Magnitude (Straddle Alpha)

Script:
4_future_move_prediction_straddle.py

The final stage of the project reframes the problem to focus directly on straddle profitability.

Instead of predicting volatility levels, the model predicts future absolute price movement, which is more directly linked to options payoffs.

Target Variable

Future absolute return over a fixed horizon:

Target = |future return over 15 days|

A classification threshold is then applied to identify large moves.

This formulation aligns closely with the payoff structure of straddles.

Features

The feature set focuses on identifying conditions that precede volatility expansion, including:

Volatility indicators

ATR

Realized volatility

Volatility compression metrics

Trend strength

ADX

ADX slope

Momentum

Rate of Change (ROC)

Short-term price acceleration

Range indicators

Rolling price ranges

Bollinger Band width

These features help capture market compression, momentum buildup, and regime transitions, which are common precursors to large price movements.

Strategy

The strategy now incorporates both long and short straddles:

Model Prediction	Strategy
High predicted move	Long straddle
Low predicted move	Short straddle
Intermediate predictions	No trade

This allows the strategy to profit from both:

volatility expansion

volatility contraction

Options Pricing Framework

Options are priced using the Black-Scholes model, with the following considerations:

Options are entered with a fixed time to expiry.

Time-to-expiry is reduced as the trade progresses, introducing theta decay.

Volatility inputs are derived from historical realized volatility.

This framework allows the backtest to approximate realistic option pricing dynamics while remaining computationally tractable.

Key Insights from the Project

Several insights emerged during development:

Directional market prediction is difficult, even with machine learning.

Predicting volatility regimes can be more feasible.

Tree-based models capture nonlinear financial relationships better than linear models.

Options strategies benefit from predicting large moves rather than direction.

Volatility compression features are often strong predictors of future expansion.

Future Improvements

Potential extensions to the project include:

Incorporating implied volatility data from real options markets.

Implementing delta-hedged straddle strategies.

Testing gradient boosting models (XGBoost / LightGBM).

Improving transaction cost modelling.

Expanding the dataset across multiple assets and regimes.

Technologies Used

Python

NumPy

Pandas

Scikit-learn

Matplotli
