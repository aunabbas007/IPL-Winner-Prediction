# model.py (Clean & Stable Version)

import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------

# 1. LOAD DATA

# -----------------------------

df = pd.read_csv("match_data.csv")

# -----------------------------

# 2. CREATE MATCH-LEVEL DATA

# -----------------------------

# Get teams per match

teams = df.groupby("match_id")["batting_team"].unique().reset_index()

# Ensure only matches with 2 teams

teams = teams[teams["batting_team"].apply(lambda x: len(x) == 2)]

# Split into team1, team2

teams["team1"] = teams["batting_team"].apply(lambda x: x[0])
teams["team2"] = teams["batting_team"].apply(lambda x: x[1])
teams = teams.drop("batting_team", axis=1)

# Base match info

match_df = df[["match_id", "venue"]].drop_duplicates()
match_df = match_df.merge(teams, on="match_id")

# -----------------------------

# 3. CREATE WINNER

# -----------------------------

runs = df.groupby(["match_id", "batting_team"])["runs_off_bat"].sum().reset_index()

winner_df = runs.loc[runs.groupby("match_id")["runs_off_bat"].idxmax()]
winner_df = winner_df[["match_id", "batting_team"]]
winner_df = winner_df.rename(columns={"batting_team": "winner"})

match_df = match_df.merge(winner_df, on="match_id")

# -----------------------------

# 4. CLEAN DATA

# -----------------------------

match_df = match_df.dropna()

# Convert to string (important)

for col in ["team1", "team2", "venue", "winner"]:
 match_df[col] = match_df[col].astype(str)

# -----------------------------

# 5. ENCODING

# -----------------------------

le_team = LabelEncoder()
le_venue = LabelEncoder()
le_winner = LabelEncoder()

# Fit on all teams

all_teams = pd.concat([match_df["team1"], match_df["team2"]])
le_team.fit(all_teams)

match_df["team1"] = le_team.transform(match_df["team1"])
match_df["team2"] = le_team.transform(match_df["team2"])

match_df["venue"] = le_venue.fit_transform(match_df["venue"])
match_df["winner"] = le_winner.fit_transform(match_df["winner"])

# -----------------------------

# 6. FEATURES & TARGET

# -----------------------------

X = match_df[["team1", "team2", "venue"]]
y = match_df["winner"]

# -----------------------------

# 7. TRAIN TEST SPLIT

# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# -----------------------------

# 8. TRAIN MODEL

# -----------------------------

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

# -----------------------------

# 9. EVALUATE

# -----------------------------

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------

# 10. SAVE MODEL

# -----------------------------

joblib.dump(model, "model.pkl")
joblib.dump(le_team, "team_encoder.pkl")
joblib.dump(le_venue, "venue_encoder.pkl")
joblib.dump(le_winner, "winner_encoder.pkl")

print("\nModel saved successfully!")
