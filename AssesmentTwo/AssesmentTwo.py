import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load all the datasets I need for this project
drivers = pd.read_csv('dataset/drivers.csv')
results = pd.read_csv('dataset/results.csv')
constructors = pd.read_csv('dataset/constructors.csv')
constructor_standings = pd.read_csv('dataset/constructor_standings.csv')
races = pd.read_csv('dataset/races.csv')
lap_times = pd.read_csv('dataset/lap_times.csv')
qualifying = pd.read_csv('dataset/qualifying.csv')

# Find Lewis Hamilton's driverId so I can filter his races
hamilton_driver = drivers[(drivers['forename'] == 'Lewis') & (drivers['surname'] == 'Hamilton')]
hamilton_id = hamilton_driver['driverId'].values[0]

# Filter the results to only get races where Lewis competed
hamilton_results = results[results['driverId'] == hamilton_id].copy()

# Merge in race details like year and circuit for more context
hamilton_results = hamilton_results.merge(races[['raceId', 'circuitId', 'year']], on='raceId', how='left')

# Add constructor info so I can see which team he was with
hamilton_results = hamilton_results.merge(constructors[['constructorId', 'name']], on='constructorId', how='left')

# Merge in constructor standings to get constructor points for each race
hamilton_results = hamilton_results.merge(constructor_standings[['raceId', 'constructorId', 'points']], on=['raceId', 'constructorId'], how='left', suffixes=('', '_constructor'))

# Merge in qualifying data to get his starting position
hamilton_results = hamilton_results.merge(qualifying[['raceId', 'driverId', 'position']], on=['raceId', 'driverId'], how='left', suffixes=('', '_qualifying'))

# Calculate average lap time for each race and driver, then add it to the results
lap_avg = lap_times.groupby(['raceId', 'driverId'])['milliseconds'].mean().reset_index()
lap_avg.rename(columns={'milliseconds': 'avg_lap_time'}, inplace=True)
hamilton_results = hamilton_results.merge(lap_avg, on=['raceId', 'driverId'], how='left')

# Set the target: 1 if he finished in the points, 0 otherwise
hamilton_results['points'] = (hamilton_results['points'] > 0).astype(int)

# Pick the features I want to use for prediction
features = ['grid', 'constructorId', 'raceId', 'year', 'points_constructor', 'position_qualifying', 'avg_lap_time']
X = hamilton_results[features]
y = hamilton_results['points']

# Encode categorical features so the model can use them
X = pd.get_dummies(X)

# Remove any rows with missing values to keep the data clean
X = X.dropna()
y = y.loc[X.index]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for logistic regression
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train logistic regression model
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
lr_cm = confusion_matrix(y_test, lr_pred)

# Train random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_cm = confusion_matrix(y_test, rf_pred)

# Count how many races Lewis Hamilton has won (position 1)
# I added this so i can test to see if the data was accurite
hamilton_wins = results[(results['driverId'] == hamilton_id) & (results['grid'] == 1)].shape[0]

# Print out the results so I can see how well the models did

print("Number of races Lewis Hamilton has won:", hamilton_wins)
print("Logistic Regression Accuracy:", lr_acc)
print("Random Forest Accuracy:", rf_acc)
print("Logistic Regression Confusion Matrix:\n", lr_cm)
print("Random Forest Confusion Matrix:\n", rf_cm)
