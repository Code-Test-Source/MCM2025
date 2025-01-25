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

# 文件路径
file_dict_path = r".\2025_Problem_C_Data\data_dictionary.csv"
athletes_file_path = r".\2025_Problem_C_Data\summerOly_athletes.csv"
hosts_file_path = r".\2025_Problem_C_Data\summerOly_hosts.csv"
medals_file_path = r".\2025_Problem_C_Data\summerOly_medal_counts.csv"
programs_file_path = r".\2025_Problem_C_Data\summerOly_programs.csv"

# 检测文件编码
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']

# 直接使用 Pandas 读取 CSV 文件
def read_csv(file_path):
    encoding = detect_encoding(file_path)
    return pd.read_csv(file_path, encoding=encoding)

# 读取数据
data_dict = read_csv(file_dict_path)
athletes = read_csv(athletes_file_path)
hosts = read_csv(hosts_file_path)
medal_counts = read_csv(medals_file_path)
programs = read_csv(programs_file_path)



country_mapping = {
    'Soviet Union': 'Russia',
    'West Germany': 'Germany',
    'East Germany': 'Germany',
    'Yugoslavia': 'Serbia',
    'Czechoslovakia': 'Czech Republic',
    'Bohemia': 'Czech Republic',
    'Russian Empire': 'Russia',
    'United Team of Germany': 'Germany',
    'Unified Team': 'Russia',
    'Serbia and Montenegro': 'Serbia',
    'Netherlands Antilles': 'Netherlands',
    'Virgin Islands': 'United States',
    'West Indies Federation': 'United States',
    'ROC': 'Russia',
    'LIB': 'Liberia',
    'Libya': 'Liberia',
}

noc_mapping = {
    'URS': 'RUS',
    'EUA': 'GER',
    'FRG': 'GER',
    'GDR': 'GER',
    'YUG': 'SRB',
    'TCH': 'CZE',
    'BOH': 'CZE',
    'EUN': 'RUS',
    'SCG': 'SRB',
    'ANZ': 'AUS',
    'NBO': 'KEN',
    'WIF': 'USA',
    'IOP': 'IOA',
    'ROC': 'RUS',
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
    'ARG': 'Argentina',
    'ARM': 'Armenia',
    'ARU': 'Aruba',
    'ASA': 'American Samoa',
    'AUS': 'Australia',
    'AUT': 'Austria',
    'AZE': 'Azerbaijan',
    'BAH': 'Bahamas',
    'BAN': 'Bangladesh',
    'BAR': 'Barbados',
    'BDI': 'Burundi',
    'BEL': 'Belgium',
    'BEN': 'Benin',
    'BER': 'Bermuda',
    'BHU': 'Bhutan',
    'BIH': 'Bosnia and Herzegovina',
    'BIZ': 'Belize',
    'BLR': 'Belarus',
    'BOL': 'Bolivia',
    'BOT': 'Botswana',
    'BRA': 'Brazil',
    'BRN': 'Bahrain',
    'BRU': 'Brunei',
    'BUL': 'Bulgaria',
    'BUR': 'Burkina Faso',
    'CAF': 'Central African Republic',
    'CAM': 'Cambodia',
    'CAN': 'Canada',
    'CAY': 'Cayman Islands',
    'CGO': 'Congo',
    'CHA': 'Chad',
    'CHI': 'Chile',
    'CHN': 'China',
    'CIV': 'Ivory Coast',
    'CMR': 'Cameroon',
    'COD': 'Congo',
    'COK': 'Cook Islands',
    'COL': 'Colombia',
    'COM': 'Comoros',
    'CPV': 'Cape Verde',
    'CRC': 'Costa Rica',
    'CRO': 'Croatia',
    'CUB': 'Cuba',
    'CYP': 'Cyprus',
    'CZE': 'Czech Republic',
    'DEN': 'Denmark',
    'DJI': 'Djibouti',
    'DMA': 'Dominica',
    'DOM': 'Dominican Republic',
    'ECU': 'Ecuador',
    'EGY': 'Egypt',
    'ERI': 'Eritrea',
    'ESA': 'El Salvador',
    'ESP': 'Spain',
    'EST': 'Estonia',
    'ETH': 'Ethiopia',
    'FIJ': 'Fiji',
    'FIN': 'Finland',
    'FRA': 'France',
    'FSM': 'Micronesia',
    'GAB': 'Gabon',
    'GAM': 'Gambia',
    'GBR': 'Great Britain',
    'GBS': 'Guinea-Bissau',
    'GEO': 'Georgia',
    'GEQ': 'Equatorial Guinea',
    'GER': 'Germany',
    'GHA': 'Ghana',
    'GRE': 'Greece',
    'GRN': 'Grenada',
    'GUA': 'Guatemala',
    'GUI': 'Guinea',
    'GUM': 'Guam',
    'GUY': 'Guyana',
    'HAI': 'Haiti',
    'HKG': 'Hong Kong',
    'HON': 'Honduras',
    'HUN': 'Hungary',
    'INA': 'Indonesia',
    'IND': 'India',
    'IRI': 'Iran',
    'IRL': 'Ireland',
    'IRQ': 'Iraq',
    'ISL': 'Iceland',
    'ISR': 'Israel',
    'ISV': 'Virgin Islands',
    'ITA': 'Italy',
    'IVB': 'British Virgin Islands',
    'JAM': 'Jamaica',
    'JOR': 'Jordan',
    'JPN': 'Japan',
    'KAZ': 'Kazakhstan',
    'KEN': 'Kenya',
    'KGZ': 'Kyrgyzstan',
    'KIR': 'Kiribati',
    'KOR': 'South Korea',
    'KOS': 'Kosovo',
    'KSA': 'Saudi Arabia',
    'KUW': 'Kuwait',
    'LAO': 'Laos',
    'LAT': 'Latvia',
    'LBA': 'Lebanon',
    'LBR': 'Liberia',
    'LCA': 'Saint Lucia',
    'LES': 'Lesotho',
    'LIE': 'Liechtenstein',
    'LTU': 'Lithuania',
    'LUX': 'Luxembourg',
    'LIB': 'Liberia',
    'MAD': 'Madagascar',
    'MAR': 'Morocco',
    'MAS': 'Malaysia',
    'MAW': 'Malawi',
    'MDA': 'Moldova',
    'MDV': 'Maldives',
    'MEX': 'Mexico',
    'MGL': 'Mongolia',
    'MHL': 'Marshall Islands',
    'MKD': 'North Macedonia',
    'MLI': 'Mali',
    'MLT': 'Malta',
    'MNE': 'Montenegro',
    'MON': 'Monaco',
    'MOZ': 'Mozambique',
    'MRI': 'Mauritius',
    'MTN': 'Mauritania',
    'MYA': 'Myanmar',
    'NAM': 'Namibia',
    'NCA': 'Nicaragua',
    'NED': 'Netherlands',
    'NEP': 'Nepal',
    'NGR': 'Nigeria',
    'NIG': 'Niger',
    'NOR': 'Norway',
    'NRU': 'Nauru',
    'NZL': 'New Zealand',
    'OMA': 'Oman',
    'PAK': 'Pakistan',
    'PAN': 'Panama',
    'PAR': 'Paraguay',
    'PER': 'Peru',
    'PHI': 'Philippines',
    'PLE': 'Palestine',
    'PLW': 'Palau',
    'PNG': 'Papua New Guinea',
    'POL': 'Poland',
    'POR': 'Portugal',
    'PRK': 'North Korea',
    'PUR': 'Puerto Rico',
    'QAT': 'Qatar',
    'ROU': 'Romania',
    'RSA': 'South Africa',
    'RUS': 'Russia',
    'RWA': 'Rwanda',
    'SAM': 'Samoa',
    'SEN': 'Senegal',
    'SEY': 'Seychelles',
    'SGP': 'Singapore',
    'SIN': 'Singapore',
    'SKN': 'Saint Kitts and Nevis',
    'SLE': 'Sierra Leone',
    'SLO': 'Slovenia',
    'SMR': 'San Marino',
    'SOL': 'Solomon Islands',
    'SRB': 'Serbia',
    'SRI': 'Sri Lanka',
    'STP': 'Sao Tome and Principe',
    'SUD': 'Sudan',
    'SUI': 'Switzerland',
    'SUR': 'Suriname',
    'SVK': 'Slovakia',
    'SWE': 'Sweden',
    'SWZ': 'Eswatini',
    'SYR': 'Syria',
    'TAN': 'Tanzania',
    'TGA': 'Tonga',
    'THA': 'Thailand',
    'TJK': 'Tajikistan',
    'TKM': 'Turkmenistan',
    'TLS': 'Timor-Leste',
    'TOG': 'Togo',
    'TPE': 'Chinese Taipei',
    'TTO': 'Trinidad and Tobago',
    'TUN': 'Tunisia',
    'TUR': 'Turkey',
    'TUV': 'Tuvalu',
    'UAE': 'United Arab Emirates',
    'UGA': 'Uganda',
    'UKR': 'Ukraine',
    'URU': 'Uruguay',
    'USA': 'United States',
    'UZB': 'Uzbekistan',
    'VAN': 'Vanuatu',
    'VEN': 'Venezuela',
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



athletes_agg.to_csv('./2025_Problem_C_Data/athletes_agg.csv')

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

data.to_csv('./2025_Problem_C_Data/data.csv')




# Prepare features and target with additional variables
X = data[['Year', 'Is_Host', 'Num_Athletes', 'Female_Ratio', 'Num_Sports', 'Num_Events','Total_Events','Total_Discipline','Total_Sports']]
y = data[['Total','Gold','Silver','Bronze']].apply(pd.to_numeric, errors='coerce')


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Create new variable matrix for XGBoost
X_train_rf = X_train.copy()
X_test_rf = X_test.copy()
for i, col in enumerate(['Total', 'Gold', 'Silver', 'Bronze']):
    X_train_rf[f'RF_Predictions_{col}'] = rf_model.predict(X_train)[:, i]
    X_test_rf[f'RF_Predictions_{col}'] = rf_predictions[:, i]

# XGBoost model
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train_rf, y_train)
xgb_predictions = xgb_model.predict(X_test_rf)

# K-S Test
ks_stat, p_value = ks_2samp(y_test, xgb_predictions)

# Print results
print(f"Random Forest Predictions: {rf_predictions}")
print(f"XGBoost Predictions: {xgb_predictions}")
print(f"K-S Test Statistic: {ks_stat}, P-Value: {p_value}")

# # Predict 2028 medals, golds, silver, bronze
# future_data = pd.DataFrame({
#     'Year': [2028], 
#     'Is_Host': [1], 
#     'Num_Athletes': [613],  # Placeholder value
#     'Female_Ratio': [0.463],  # Placeholder value
#     'Num_Sports': [47],    # Placeholder value
#     'Num_Events': [234],     # Placeholder value
#     'Total_Events': [329], # Placeholder value
#     'Total_Discipline': [48], # Placeholder value
#     'Total_Sports': [32]   # Placeholder value
# })
# rf_predictions_future = rf_model.predict(future_data).reshape(1, -1)
# future_data['RF_Predictions_Total'] = rf_predictions_future[:, 0]
# future_data['RF_Predictions_Gold'] = rf_predictions_future[:, 1]
# future_data['RF_Predictions_Silver'] = rf_predictions_future[:, 2]
# future_data['RF_Predictions_Bronze'] = rf_predictions_future[:, 3]

# # Ensure future_data has the same columns as X_train_rf
# future_data_rf = future_data[['Year', 'Is_Host', 'Num_Athletes', 'Female_Ratio', 'Num_Sports', 'Num_Events', 'Total_Events', 'Total_Discipline', 'Total_Sports',
#                               'RF_Predictions_Total', 'RF_Predictions_Gold', 'RF_Predictions_Silver', 'RF_Predictions_Bronze']]

# future_predictions = xgb_model.predict(future_data_rf)
# print(f"Predicted Medals for 2028: Total: {future_predictions[0][0]}, Gold: {future_predictions[0][1]}, Silver: {future_predictions[0][2]}, Bronze: {future_predictions[0][3]}")


# Predict medals for 2028 for all countries
future_data_all = data[(data['Year'] == 2024)  & (data['NOC'] != 'France') | ((data['Year'] == 2020) & (data['NOC'] == 'France')) | ((data['Year'] == 2016) & (data['NOC'] == 'Russia'))].copy()

future_data_all['Year'] = 2028


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
future_medal_totals_sorted.to_csv('./2025_Problem_C_Data/future_medal_totals_2028.csv', index=False)

# Sort by predicted total medals and get top 10 countries
top_10_future_countries = future_medal_totals_sorted.head(10)

# Plot the predicted medal counts for the top 10 countries in 2028 with horizontal stacked bars
fig, ax = plt.subplots(figsize=(12, 8))

# Plot stacked bars for each medal type
ax.barh(top_10_future_countries['NOC'], top_10_future_countries['Predicted_Gold'], color='gold', label='Gold')
ax.barh(top_10_future_countries['NOC'], top_10_future_countries['Predicted_Silver'], left=top_10_future_countries['Predicted_Gold'], color='silver', label='Silver')
ax.barh(top_10_future_countries['NOC'], top_10_future_countries['Predicted_Bronze'], left=top_10_future_countries['Predicted_Gold'] + top_10_future_countries['Predicted_Silver'], color='#cd7f32', label='Bronze')

# Add data labels
for i in range(len(top_10_future_countries)):
    ax.text(top_10_future_countries['Predicted_Gold'].iloc[i] / 2, i, int(top_10_future_countries['Predicted_Gold'].iloc[i]), va='center', ha='center', color='black')
    ax.text(top_10_future_countries['Predicted_Gold'].iloc[i] + top_10_future_countries['Predicted_Silver'].iloc[i] / 2, i, int(top_10_future_countries['Predicted_Silver'].iloc[i]), va='center', ha='center', color='black')
    ax.text(top_10_future_countries['Predicted_Gold'].iloc[i] + top_10_future_countries['Predicted_Silver'].iloc[i] + top_10_future_countries['Predicted_Bronze'].iloc[i] / 2, i, int(top_10_future_countries['Predicted_Bronze'].iloc[i]), va='center', ha='center', color='black')

plt.xlabel('Number of Medals')
plt.ylabel('Country')
plt.title('Top 10 Countries by Predicted Medals in 2028')
plt.legend(loc='upper right')
plt.gca().invert_yaxis()
plt.show()


# Calculate Mean Squared Error and R^2 for Random Forest
mse_rf = mean_squared_error(y_test, rf_predictions)
r2_rf = r2_score(y_test, rf_predictions)

# Calculate Mean Squared Error and R^2 for XGBoost
mse_xgb = mean_squared_error(y_test, xgb_predictions)
r2_xgb = r2_score(y_test, xgb_predictions)

# Print regression metrics
print(f"Random Forest MSE: {mse_rf}, R^2: {r2_rf}")
print(f"XGBoost MSE: {mse_xgb}, R^2: {r2_xgb}")

# Plot predictions vs actual values
plt.figure()
plt.scatter(y_test, rf_predictions, color='blue', label='Random Forest Predictions')
plt.scatter(y_test, xgb_predictions, color='red', label='XGBoost Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='gray', lw=2, linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predictions vs Actual Values')
plt.legend(loc='upper left')
plt.show()

# Plot correlation matrix without 'NOC' column
plt.figure(figsize=(12, 10))
correlation_matrix = data.drop(columns=['NOC']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Calculate Mean Squared Error and R^2 for each target variable
mse_total = mean_squared_error(y_test['Total'], xgb_predictions[:, 0])
r2_total = r2_score(y_test['Total'], xgb_predictions[:, 0])

mse_gold = mean_squared_error(y_test['Gold'], xgb_predictions[:, 1])
r2_gold = r2_score(y_test['Gold'], xgb_predictions[:, 1])

mse_silver = mean_squared_error(y_test['Silver'], xgb_predictions[:, 2])
r2_silver = r2_score(y_test['Silver'], xgb_predictions[:, 2])

mse_bronze = mean_squared_error(y_test['Bronze'], xgb_predictions[:, 3])
r2_bronze = r2_score(y_test['Bronze'], xgb_predictions[:, 3])

# Print regression metrics for each target variable
print(f"XGBoost Total Medals - MSE: {mse_total}, R^2: {r2_total}")
print(f"XGBoost Gold Medals - MSE: {mse_gold}, R^2: {r2_gold}")
print(f"XGBoost Silver Medals - MSE: {mse_silver}, R^2: {r2_silver}")
print(f"XGBoost Bronze Medals - MSE: {mse_bronze}, R^2: {r2_bronze}")
# Plot predictions vs actual values for each target variable
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Total Medals
axs[0, 0].scatter(y_test['Total'], xgb_predictions[:, 0], color='blue', label='Predicted Total Medals')
axs[0, 0].plot([y_test['Total'].min(), y_test['Total'].max()], [y_test['Total'].min(), y_test['Total'].max()], color='gray', lw=2, linestyle='--')
axs[0, 0].set_xlabel('Actual Total Medals')
axs[0, 0].set_ylabel('Predicted Total Medals')
axs[0, 0].set_title('Total Medals: Predictions vs Actual')
axs[0, 0].legend()

# Gold Medals
axs[0, 1].scatter(y_test['Gold'], xgb_predictions[:, 1], color='gold', label='Predicted Gold Medals')
axs[0, 1].plot([y_test['Gold'].min(), y_test['Gold'].max()], [y_test['Gold'].min(), y_test['Gold'].max()], color='gray', lw=2, linestyle='--')
axs[0, 1].set_xlabel('Actual Gold Medals')
axs[0, 1].set_ylabel('Predicted Gold Medals')
axs[0, 1].set_title('Gold Medals: Predictions vs Actual')
axs[0, 1].legend()

# Silver Medals
axs[1, 0].scatter(y_test['Silver'], xgb_predictions[:, 2], color='silver', label='Predicted Silver Medals')
axs[1, 0].plot([y_test['Silver'].min(), y_test['Silver'].max()], [y_test['Silver'].min(), y_test['Silver'].max()], color='gray', lw=2, linestyle='--')
axs[1, 0].set_xlabel('Actual Silver Medals')
axs[1, 0].set_ylabel('Predicted Silver Medals')
axs[1, 0].set_title('Silver Medals: Predictions vs Actual')
axs[1, 0].legend()

# Bronze Medals
axs[1, 1].scatter(y_test['Bronze'], xgb_predictions[:, 3], color='#cd7f32', label='Predicted Bronze Medals')
axs[1, 1].plot([y_test['Bronze'].min(), y_test['Bronze'].max()], [y_test['Bronze'].min(), y_test['Bronze'].max()], color='gray', lw=2, linestyle='--')
axs[1, 1].set_xlabel('Actual Bronze Medals')
axs[1, 1].set_ylabel('Predicted Bronze Medals')
axs[1, 1].set_title('Bronze Medals: Predictions vs Actual')
axs[1, 1].legend()

plt.tight_layout()
plt.show()