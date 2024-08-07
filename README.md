# NBA Predictions Project

**Note: This project is ever-evolving, and updates will be made periodically. New features, improvements, and modifications are expected to enhance the functionality and user experience. Feel free to check back for the latest developments!**

## Overview

The NBA Predictions Project aims to forecast the winners of each NBA game. Utilizing data analysis and predictive modeling, the project provides projections for upcoming games. The predictions will be shared on Twitter via [@ballgorithm](https://twitter.com/ballgorithm). Stay tuned for insights into the outcomes of your favorite NBA matchups!

## Features

### 1. Model Training

- **Target Variable Creation:** Checks for the existence of the 'Win' column and creates a binary target variable ('HomeWin') indicating whether the home team wins.

- **Data Splitting:** Splits the data into training and testing sets for model evaluation.

- **Logistic Regression Model:** Utilizes logistic regression for binary outcome prediction, specifically whether the home team wins or not.

- **Training on Completed Data:** Trains the model on completed data, representing games that have already occurred.

### 2. Model Evaluation

- **Accuracy Evaluation:** Evaluates the accuracy of the binary outcome prediction model on the test set.

- **Console Output:** Prints the accuracy to the console for quick assessment.

### 3. Upcoming Games Prediction

- **Data Preparation:** Provides upcoming game data as a dictionary and converts it into a DataFrame.

- **One-Hot Encoding:** Encodes categorical columns for upcoming games.

- **Prediction Using Logistic Regression:** Uses the trained logistic regression model to predict the winners for upcoming games.

### 4. PrettyTable Display

- **Predictions Display:** Uses PrettyTable to display predictions for upcoming games.

- **Additional Information:** Displays team win-loss records along with the predicted winners.

- **Color Styling:** Utilizes color styling, with headers in blue and predicted winners in green.

### 5. Output

- **Console Display:** Prints the final table to the console, showing matchups and predicted winners.

## Ballgorithm NBA Projection Model v1.1 - Updated on 3rd Feb, 2024.

### Changes
- Updated prediction algorithm to consider team win-loss records.
- Implemented color-coding for projected winners based on win percentage.
- Fixed formatting issues in PrettyTable display.
- Adjusted color codes for better readability.

### Files added
- cumulative_accuracy.ipynb
- espn_comparison.ipynb
- espn_comparison.xlsx

### Usage
Ensure you have the required dependencies installed:
pip install pandas, scikit-learn, colorama, prettytable


## Prerequisites

Dependencies or prerequisites.

- Python 3.x
- Required Python packages (pip install pandas, scikit-learn, colorama, prettytable)
- from sklearn.model_selection import train_test_split
- from sklearn.linear_model import LogisticRegression
- from colorama import Fore, Style
- from prettytable import PrettyTable


## Acknowledgments

### Data Source

The NBA data used in this project is sourced from [Basketball Reference](https://www.basketball-reference.com/). I express my gratitude for providing valuable data that contributes to the success of this project.

### Contact

If you have any questions or feedback, feel free to reach out:

- Email: ballgorithm@gmail.com
- Twitter: [@ballgorithm](https://twitter.com/ballgorithm)
