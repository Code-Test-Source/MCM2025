from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
noc_mapping = {
    'URS': 'RUS',
    ...
    'EOR': 'RUS',
}
country_codes = {
    'EOR': 'Russia',
    'ROC': 'Russia',
    'AFG': 'Afghanistan',
    ...
    'ZIM': 'Zimbabwe'
}
athletes['NOC'] = athletes['NOC'].replace(noc_mapping)
medal_counts['NOC'] = medal_counts['NOC'].replace(country_mapping)
hosts[['City', 'NOC']] = hosts['Host'].str.split(',', expand=True)
hosts['NOC'] = hosts['NOC'].str.strip()
athletes['NOC'] = athletes['NOC'].map(country_codes).fillna(athletes['NOC'])
athletes['Sex'] = athletes['Sex'].map({'M': 1, 'F': 0})
athletes_agg = athletes.groupby(['Year', 'NOC']).agg({
    'Name': lambda x: x.nunique(),
    'Sex': lambda x: x.mean(),
    'Sport': lambda x: x.nunique(),
    'Event': lambda x: x.nunique()
}).reset_index()
athletes_agg.rename(columns={'Name': 'Num_Athletes', 'Sex': 'Female_Ratio', 'Sport': 'Num_Sports', 'Event': 'Num_Events'}, inplace=True)
medal_counts['Year'] = medal_counts['Year'].astype(int)
data = pd.merge(athletes_agg, medal_counts, on=['Year', 'NOC'], how='left')
programs_sum = pd.read_csv(programs_file_path, skiprows=lambda x: x not in [0, 72, 73, 74], usecols=range(4, programs.shape[1]))
programs_sum = programs_sum.transpose().reset_index()
programs_sum.columns = ['Year', 'Total_Events', 'Total_Discipline', 'Total_Sports']
programs_sum['Year'] = programs_sum['Year'].astype(int)
data = pd.merge(data, programs_sum, on='Year', how='left')
data['Is_Host'] = data.apply(lambda row: 1 if row['NOC'] in hosts[hosts['Year'] == row['Year']]['NOC'].values else 0, axis=1)
data = data.fillna(0)
X = data[[features]]
y = data[['Total','Gold','Silver','Bronze']].apply(pd.to_numeric, errors='coerce')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.19, random_state=42)
rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
X_train_rf = X_train.copy()
X_test_rf = X_test.copy()
for i, col in enumerate(['Total', 'Gold', 'Silver', 'Bronze']):
    X_train_rf[f'RF_Predictions_{col}'] = rf_model.predict(X_train)[:, i]
    X_test_rf[f'RF_Predictions_{col}'] = rf_predictions[:, i]
xgb_model = XGBRegressor(n_estimators=1000, random_state=42)
xgb_model.fit(X_train_rf, y_train)
xgb_predictions = xgb_model.predict(X_test_rf)
ks_stat, p_value = ks_2samp(y_test, xgb_predictions)
future_data_all = data[history data weighted].copy()
future_data_all['Year'] = 2032
rf_predictions_all = rf_model.predict(future_data_all[features])
future_data_rf_all = future_data_all[features and prediction features]
future_predictions_all = xgb_model.predict(future_data_rf_all)
future_medal_totals = future_data_all.groupby('NOC')[['Predicted_Gold', 'Predicted_Silver', 'Predicted_Bronze']].sum().reset_index()
future_medal_totals['Predicted_Total'] = future_medal_totals['Predicted_Gold'] + future_medal_totals['Predicted_Silver'] + future_medal_totals['Predicted_Bronze']
future_medal_totals_sorted = future_medal_totals.sort_values(by='Predicted_Total', ascending=False)
