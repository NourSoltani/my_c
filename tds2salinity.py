#!pip install joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import joblib


df = pd.read_csv("./merged_data_1H.csv")

# df.isna().sum()
df = df.dropna(subset=['Densité'])
df = df.reset_index(drop=True)

df = df.drop(columns=['date','system_temperature', 'lidar_temperature','MORET 2', 'MORET 3', 'MORET 4', 'MORET 5','weather_windDirection','weather_windSpeed'])

# Convert categorical data to numerical
#df['weather_windDirection'] = pd.Categorical(df['weather_windDirection']).codes
# Handle missing values if any
df.fillna(df.mean(numeric_only=True), inplace=True)

# Separate features and target
X = df.drop(columns=['Densité'])
y = df['Densité']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred_rf = rf_model.predict(X_test_scaled)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# print(f"Random Forest MSE: {mse_rf:.2f}")
# print(f"Random Forest R^2: {r2_rf:.2f}")

# Random Forest Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

# Define parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}

# Initialize GridSearchCV
grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                              param_grid=param_grid,
                              cv=5,
                              scoring='neg_mean_squared_error',
                              n_jobs=-1,
                              verbose=2)

# Fit grid search
grid_search_rf.fit(X_train_scaled, y_train)

# Best parameters and model
best_params_rf = grid_search_rf.best_params_
best_model_rf = grid_search_rf.best_estimator_

#print(f"Best Random Forest Parameters: {best_params_rf}")

# Evaluate the best Random Forest model
y_pred_best_rf = best_model_rf.predict(X_test_scaled)
mse_best_rf = mean_squared_error(y_test, y_pred_best_rf)
r2_best_rf = r2_score(y_test, y_pred_best_rf)

#print(f"Best Random Forest MSE: {mse_best_rf:.2f}")
#print(f"Best Random Forest R^2: {r2_best_rf:.2f}")

# Save the best Random Forest model
joblib.dump(best_model_rf, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

'''
# Load the Random Forest model, and the scaler.
loaded_model = joblib.load('random_forest_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

input_value = np.array([[327.731893,27.383333,0.0,97.266667,271.000000]])  # Replace 'float_value' with your actual float value

# Standardize the input value using the loaded scaler
input_value_scaled = loaded_scaler.transform(input_value)

# Predict using the loaded model
prediction = round(loaded_model.predict(input_value_scaled)[0])

# Output the prediction
print(f"Model Prediction: {prediction}")
'''