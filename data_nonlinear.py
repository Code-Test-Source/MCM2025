import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from scipy.stats import ks_2samp
import numpy as np


import chardet
from sklearn.metrics import mean_squared_error, r2_score

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
# Ensure the data is encoded in ISO-8859-1


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
}

country_codes = {
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
    'COD': 'Democratic Republic of the Congo',
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
    'LBA': 'Libya',
    'LBR': 'Liberia',
    'LCA': 'Saint Lucia',
    'LES': 'Lesotho',
    'LIE': 'Liechtenstein',
    'LTU': 'Lithuania',
    'LUX': 'Luxembourg',
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

# Remove ice sports and athletes playing ice sports
ice_sports = ['Figure Skating', 'Ice Hockey']
programs = programs[~programs['Sport'].isin(ice_sports)]
athletes = athletes[~athletes['Sport'].isin(ice_sports)]

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

# Merge athletes_agg and medal_counts
data = pd.merge(athletes_agg, medal_counts, on=['Year', 'NOC'], how='left')

data = data.fillna(0)

data.to_csv('./2025_Problem_C_Data/data.csv')

# Determine if the country is the host for each year
data['Is_Host'] = data.apply(lambda row: 1 if row['NOC'] in hosts[hosts['Year'] == row['Year']]['NOC'].values else 0, axis=1)


# Prepare features and target with additional variables
X = data[['Year', 'Is_Host', 'Num_Athletes', 'Female_Ratio', 'Num_Sports', 'Num_Events']]
y = data['Total']


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Create new variable matrix for XGBoost
X_train['RF_Predictions'] = rf_model.predict(X_train)
X_test['RF_Predictions'] = rf_predictions

# XGBoost model
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

# K-S Test
ks_stat, p_value = ks_2samp(y_test, xgb_predictions)

# Print results
print(f"Random Forest Predictions: {rf_predictions}")
print(f"XGBoost Predictions: {xgb_predictions}")
print(f"K-S Test Statistic: {ks_stat}, P-Value: {p_value}")

# Predict 2028 medals
future_data = pd.DataFrame({
    'Year': [2028], 
    'Is_Host': [1], 
    'Num_Athletes': [592],  # Placeholder value
    'Female_Ratio': [0.95],  # Placeholder value
    'Num_Sports': [28],    # Placeholder value
    'Num_Events': [5]     # Placeholder value
})
future_data['RF_Predictions'] = rf_model.predict(future_data)
future_predictions = xgb_model.predict(future_data)
print(f"Predicted Medals for 2028: {future_predictions}")

import matplotlib.pyplot as plt

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