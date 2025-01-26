import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from scipy.stats import ks_2samp
import numpy as np
import matplotlib.pyplot as plt
import chardet
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
country_mapping = {
    'Soviet Union': 'Russia',
    ...,
    'Libya': 'Liberia',
}
noc_mapping = {
    'URS': 'RUS',
    ...
    'EOR': 'RUS',
}
country_codes = {
    'EOR': 'Russia',
    'ROC': 'Russia',
    'AFG': 'Afghanistan',
    'ALB': 'Albania',
    'ALG': 'Algeria',
    'AND': 'Andorra',
    'ANG': 'Angola',
    'ANT': 'Antigua and Barbuda',
    ...
    'VIE': 'Vietnam',
    'VIN': 'Saint Vincent and the Grenadines',
    'YEM': 'Yemen',
    'ZAM': 'Zambia',
    'ZIM': 'Zimbabwe'
}
athletes['NOC'] = athletes['NOC'].replace(noc_mapping)
medal_counts['NOC'] = medal_counts['NOC'].replace(country_mapping)
# Split Host in hosts.csv into City and Country
hosts[['City', 'NOC']] = hosts['Host'].str.split(',', expand=True)
hosts['NOC'] = hosts['NOC'].str.strip()
# Map NOC in athletes.csv to countries
athletes['NOC'] = athletes['NOC'].map(country_codes).fillna(athletes['NOC'])
# Preprocess athletes data
athletes['Sex'] = athletes['Sex'].map({'M': 1, 'F': 0})
athletes_agg = athletes.groupby(['Year', 'NOC']).agg({
    'Name': lambda x: x.nunique(),
    'Sex': lambda x: x.mean(),
    'Sport': lambda x: x.nunique(),
    'Event': lambda x: x.nunique()
}).reset_index()
athletes_agg.rename(columns={'Name': 'Num_Athletes', 'Sex': 'Female_Ratio', 'Sport': 'Num_Sports', 'Event': 'Num_Events'}, inplace=True)
# Convert 'Year' column to int in medal_counts
medal_counts['Year'] = medal_counts['Year'].astype(int)
# Merge athletes_agg and medal_counts
data = pd.merge(athletes_agg, medal_counts, on=['Year', 'NOC'], how='left')
# Read specific rows and columns from programs.csv
programs_sum = pd.read_csv(programs_file_path, skiprows=lambda x: x not in [0, 72, 73, 74], usecols=range(4, programs.shape[1]))
# Transform the data into the required format
programs_sum = programs_sum.transpose().reset_index()
programs_sum.columns = ['Year', 'Total_Events', 'Total_Discipline', 'Total_Sports']
# Convert 'Year' column to int in programs_sum
programs_sum['Year'] = programs_sum['Year'].astype(int)
# Merge programs_sum with data on Year
data = pd.merge(data, programs_sum, on='Year', how='left')
# Determine if the country is the host for each year
data['Is_Host'] = data.apply(lambda row: 1 if row['NOC'] in hosts[hosts['Year'] == row['Year']]['NOC'].values else 0, axis=1)
data = data.fillna(0)
# Prepare features and target with additional variables
X = data[['Year', 'Is_Host', 'Num_Athletes', 'Female_Ratio', 'Num_Sports', 'Num_Events','Total_Events','Total_Discipline','Total_Sports']]
y = data[['Total','Gold','Silver','Bronze']].apply(pd.to_numeric, errors='coerce')
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.19, random_state=42)
# Random Forest model
rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
# Create new variable matrix for XGBoost
X_train_rf = X_train.copy()
X_test_rf = X_test.copy()
for i, col in enumerate(['Total', 'Gold', 'Silver', 'Bronze']):
    X_train_rf[f'RF_Predictions_{col}'] = rf_model.predict(X_train)[:, i]
    X_test_rf[f'RF_Predictions_{col}'] = rf_predictions[:, i]
# XGBoost model
xgb_model = XGBRegressor(n_estimators=1000, random_state=42)
xgb_model.fit(X_train_rf, y_train)
xgb_predictions = xgb_model.predict(X_test_rf)
# K-S Test
ks_stat, p_value = ks_2samp(y_test, xgb_predictions)
# Predict medals for 2028 for all countries
future_data_all = data[(data['Year'] == 2024)  & (data['NOC'] != 'France') | ((data['Year'] == 2020) & (data['NOC'] == 'France')) | ((data['Year'] == 2016) & (data['NOC'] == 'Russia'))].copy()
future_data_all['Year'] = 2032
# Predict using Random Forest
rf_predictions_all = rf_model.predict(future_data_all[['Year', 'Is_Host', 'Num_Athletes', 'Female_Ratio', 'Num_Sports', 'Num_Events', 'Total_Events', 'Total_Discipline', 'Total_Sports']])
future_data_all['RF_Predictions_Total'] = rf_predictions_all[:, 0]
future_data_all['RF_Predictions_Gold'] = rf_predictions_all[:, 1]
future_data_all['RF_Predictions_Silver'] = rf_predictions_all[:, 2]
future_data_all['RF_Predictions_Bronze'] = rf_predictions_all[:, 3]
# Ensure future_data_all has the same columns as X_train_rf
future_data_rf_all = future_data_all[['Year', 'Is_Host', 'Num_Athletes', 'Female_Ratio', 'Num_Sports', 'Num_Events', 'Total_Events', 'Total_Discipline', 'Total_Sports',
                                      'RF_Predictions_Total', 'RF_Predictions_Gold', 'RF_Predictions_Silver', 'RF_Predictions_Bronze']]
# Predict using XGBoost
future_predictions_all = xgb_model.predict(future_data_rf_all)
# Aggregate predicted medal counts for each country
future_data_all['Predicted_Gold'] = future_predictions_all[:, 1]
future_data_all['Predicted_Silver'] = future_predictions_all[:, 2]
future_data_all['Predicted_Bronze'] = future_predictions_all[:, 3]
# Aggregate medal counts for each country
future_medal_totals = future_data_all.groupby('NOC')[['Predicted_Gold', 'Predicted_Silver', 'Predicted_Bronze']].sum().reset_index()
# Sort by predicted total medals and save to CSV
future_medal_totals['Predicted_Total'] = future_medal_totals['Predicted_Gold'] + future_medal_totals['Predicted_Silver'] + future_medal_totals['Predicted_Bronze']
future_medal_totals_sorted = future_medal_totals.sort_values(by='Predicted_Total', ascending=False)
