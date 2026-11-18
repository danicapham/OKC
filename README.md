# OKC Shot Prediction Project

## Project Overview

This project develops a  machine learning model using eXtreme Gradient Boosting (XGBoost) to predict the  make probability of NBA basketball shots. The primary objective is to minimize Log-Loss on unseen test data while maintaining strict probability calibration and preventing data leakage.

The pipeline accounts for spatial court geometry, shooter historical efficiency, and real-time defensive tracking metrics jointly.

The analysis uses:
- Exploratory Data Analysis (EDA) & Spatial Visualization
- Domain-Specific Feature Engineering & Target Encoding
- Sparse Matrix Processing (na.pass missing-value routing)
- Hyperparameter-Tuned Gradient Boosted Decision Trees (XGBoost)

## Research Question

How accurately can spatial tracking metrics, defensive closeout momentum, and individual shooter efficiency predict the exact probability of an NBA shot resulting in a made basket, as evaluated by Log-Loss?

## Data

The two datasets (training.csv.gz and testing.csv.gz) contain hundreds of thousands of NBA shot attempts alongside spatial, temporal, and player-tracking features describing each individual shot. The training set includes an additional binary target column (outcome: TRUE/FALSE or 1/0) indicating whether the shot was made.

Key variables and engineered features include:
- Shot distance
- locationx
- locationy
- shottype
- three (binary)
- shotclock
- dribblesbefore
- shooterspeed
        
## Methods

Analyses included:
- Data Cleaning & Parsing: Extracting time-series defender approach distances from raw array strings using tidyr and stringr
- Leakage-Free Target Encoding: Computing historical field goal percentages strictly on the training split before merging onto validation and test sets
- Spatial & Diagnostic Visualization: Generating vertical half-court hexbin efficiency plots, Feature Importance rankings (by Gain), Probability Density distributions, and Reliability/Calibration curves
- Gradient Boosted Modeling: Training a conservative binary:logistic XGBoost model (eta = 0.02, max_depth = 4, min_child_weight = 5, subsample = 0.8, colsample_bytree = 0.8) with early stopping on an 80/20 train/validation split

## Key Findings

Model Performance: The tuned XGBoost architecture achieved a validation Log-Loss of ~0.6359, demonstrating strong generalization and outperforming a baseline Logistic Regression model while avoiding extreme, overconfident probability penalties.

Feature Importance: Spatial proximity (distance), primary defender proximity (closestdefdist), shot type (shottype), and target-encoded shooter efficiency (shooter_fg_pct) emerged as the primary drivers of predictive Information Gain.

Calibration & Noise: Probability density diagnostics confirm that due to the inherent variance of NBA shooting, the model conservatively clusters uncertain shots near the 35%–45% baseline while cleanly separating high-efficiency looks (layups/dunks) into the upper probability tail.

## Tools Used

- R
- xgboost
- Matrix (Sparse Model Matrices)
- tidyverse (dplyr, tidyr, stringr, readr)
- ggplot2 & caret (Spatial Hexbins & Calibration Curves)
- Metrics (Log-Loss Evaluation)

## AI Usage

- Gemini (Code debugging and documentation)
- Claude Code (feature engineering architecture and hyperparameter tuning)
