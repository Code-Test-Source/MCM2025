import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv('./2025_Problem_C_Data/data.csv')


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
    'ROC': 'Russia',
}


data['NOC'] = data['NOC'].replace(country_mapping)

# Filter out the data for countries that won medals in 1896
data_1896_winners = data[(data['Year'] == 1896) & (data['Total'] > 0)]['NOC'].unique()
data = data[~data['NOC'].isin(data_1896_winners)]

# Create a feature set for each country
country_features = data.groupby('NOC').agg({
    'Year': ['count', 'min', 'max'],
    'Female_Ratio': 'mean',
    'Num_Athletes': 'sum',
    'Num_Sports': 'sum',
    'Num_Events': 'sum',
    'Total': 'sum'
}).reset_index()

country_features.columns = ['NOC', 'Participations', 'First_Year', 'Last_Year', 'Total_Athletes', 'Female_Ratio','Total_Sports', 'Total_Events', 'Total_Medals']

# Add a column for whether the country has won a medal
country_features['Has_Won'] = country_features['Total_Medals'] > 0

# Encode the NOC column
le = LabelEncoder()
country_features['NOC'] = le.fit_transform(country_features['NOC'])

# Split the data into training and testing sets
X = country_features.drop(columns=['Has_Won', 'Total_Medals'])
y = country_features['Has_Won']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10000, random_state=42)
clf.fit(X_train, y_train)

# Predict the probability of winning a medal for each country
country_features['Win_Probability'] = clf.predict_proba(X)[:, 1]

# Print the countries with the highest probability of winning a medal for the first time
first_time_winners = country_features[country_features['Has_Won'] == False].sort_values(by='Win_Probability', ascending=False)
# Decode the NOC column back to country names
first_time_winners['NOC'] = le.inverse_transform(first_time_winners['NOC'])
print(first_time_winners[['NOC', 'Win_Probability']].head(10))

# Save the prediction data to a CSV file
country_features.to_csv('./2025_Problem_C_Data/first_win_predictions.csv', index=False)

# Plot a pie chart for the top 10 countries with the highest probability of winning a medal for the first time
import matplotlib.pyplot as plt

top_10_countries = first_time_winners.head(10)
plt.figure(figsize=(10, 7))
plt.pie(top_10_countries['Win_Probability'], labels=top_10_countries['NOC'], autopct='%1.1f%%', startangle=140)
plt.title('Top 10 Countries with Highest Probability of Winning a Medal for the First Time')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Calculate the number of countries that have not won a medal
num_countries_no_medal = country_features[country_features['Has_Won'] == False].shape[0]
print(f'Number of countries that have not won a medal: {num_countries_no_medal}')

# Calculate the number of countries that have a probability of winning a medal greater than 0.3 and have not won a medal
num_countries_prob_above_3 = first_time_winners[first_time_winners['Win_Probability'] > 0.3].shape[0]
print(f'Number of countries with a probability of winning a medal greater than 0.3: {num_countries_prob_above_3}')

# Plot a histogram for the win probabilities of all countries
plt.figure(figsize=(12, 6))
plt.hist(country_features['Win_Probability'], bins=num_countries_no_medal, color='b', alpha=0.7, edgecolor='black')
plt.title('Distribution of Win Probabilities for All Countries')
plt.xlabel('Win Probability')
plt.ylabel('Number of Countries')
plt.show()
