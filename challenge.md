# Simulated Market Price Prediction Challenge

## Overview

The primary goal of this challenge is to build a robust predictive model for a simulated market environment. Participants are tasked with predicting the price movement for the next time interval based on provided market and auxiliary data.

## Dataset Description

Participants will be provided with synthetic market data alongside auxiliary datasets that contain hidden patterns and predictive signals. To ensure a fair and rigorous evaluation, the data is partitioned into three distinct sets:

- **Training set** — Publicly available data intended for model building, training, feature engineering, and hyperparameter tuning.
- **Public leaderboard** — Unseen data used for intermediate testing. Model performance on this dataset provides continuous feedback via a live, public leaderboard.
- **Private leaderboard** — A completely hidden, holdout dataset reserved for a single “one-shot” attempt. This set determines the final competition standings and is used to evaluate the model’s true generalization capabilities.

## Task & Evaluation

- **Objective:** Predict the price movement for the subsequent time interval.
- **Evaluation:** Models will be strictly evaluated based on their predictive performance on the final hidden test set (Private Leaderboard).

## Submission & Platform

- **Platform:** The challenge is hosted and managed on Kaggle.
- **Submission process:** Participants must upload their datasets and submit their trained weight evaluations directly on the Kaggle platform to be scored against the leaderboard.
