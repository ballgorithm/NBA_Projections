# app.py
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from prettytable import PrettyTable

app = Flask(__name__)

# Read the existing data
df_existing = pd.read_excel('NBA2324.xlsx')

# Check if 'Win' column exists before proceeding
winning_team = 'Win'  # Update with the actual column name
if winning_team in df_existing.columns:
    # Create a binary target variable indicating whether the home team wins
    df_existing['HomeWin'] = (df_existing['Home team']
                              == df_existing[winning_team]).astype(int)

    # Specify the categorical columns for one-hot encoding
    categorical_columns = ['Away team', 'Home team', 'Overtime', 'Arena']

    # Check if categorical columns exist in the DataFrame
    missing_columns = [
        col for col in categorical_columns if col not in df_existing.columns]

    if not missing_columns:
        # Extract necessary columns for the model
        X_completed = df_existing[categorical_columns]
        y_completed = df_existing['HomeWin']

        # One-hot encode categorical columns
        X_completed_encoded = pd.get_dummies(X_completed, drop_first=True)

        # Split the completed data into training and testing sets
        X_train_completed, _, y_train_completed, _ = train_test_split(
            X_completed_encoded, y_completed, test_size=0.2, random_state=42
        )

        # Choose a model for completed games (Logistic Regression in this example)
        model_completed = LogisticRegression()

        # Train the model on the completed data
        model_completed.fit(X_train_completed, y_train_completed)
    else:
        print(f"Error: Columns {missing_columns} not found in DataFrame.")
else:
    print(f"Error: Column '{winning_team}' not found in DataFrame.")


def load_machine_learning_model():
    # Return the trained machine learning model
    return model_completed


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        away_team = request.form['awayTeam']
        home_team = request.form['homeTeam']

        # Use the loaded machine learning model to make predictions
        model = load_machine_learning_model()

        # Upcoming game data
        upcoming_game_data = {
            'Away team': [away_team],
            'Home team': [home_team],
            'Overtime': [0],  # Placeholder for upcoming games
            'Arena': ['Arena']  # Placeholder for upcoming games
        }

        # Create a DataFrame for the upcoming game
        df_upcoming_game = pd.DataFrame(upcoming_game_data)

        # Ensure the columns in df_upcoming_game_encoded match the columns used during training
        df_upcoming_game_encoded = pd.get_dummies(df_upcoming_game)
        missing_columns = set(X_train_completed.columns) - \
            set(df_upcoming_game_encoded.columns)
        for column in missing_columns:
            df_upcoming_game_encoded[column] = 0

        # Reorder columns to match the order during training
        df_upcoming_game_encoded = df_upcoming_game_encoded[X_train_completed.columns]

        # Make predictions using the trained model for binary outcome
        predicted_probabilities = model.predict_proba(df_upcoming_game_encoded)

        # Determining the team with the higher win probability as the projected winner
        team1_probability = predicted_probabilities[0][1] * 100
        team2_probability = 100 - team1_probability

        if team1_probability > 50:
            predicted_winner = away_team
            projected_win_percentage = team1_probability
        else:
            predicted_winner = home_team
            projected_win_percentage = team2_probability

        # ANSI escape codes for blue color
        blue_color = '\033[94m'
        reset_color = '\033[0m'

        # Data for PrettyTable
        table_new = PrettyTable()
        table_new.field_names = [
            f"{blue_color}NBA, {reset_color}", f"{blue_color}Projected Winners{reset_color}"]
        table_new.align["Projected Winners"] = "l"
        table_new.horizontal_char = '-'  # Use a horizontal line as a separator

        # Defining color codes for text
        green_color = '\033[92m'
        reset_color = '\033[0m'

        # Team information
        team1_info = f"{green_color}{away_team}{reset_color}"
        team2_info = f"{green_color}{home_team}{reset_color}"
        team_info = f"{team1_info} vs {team2_info}"

        # Add row to PrettyTable
        table_new.add_row(
            [team_info, f"{predicted_winner} ({projected_win_percentage:.2f}%)"])

        return render_template('result.html', table=table_new.get_html_string())
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
