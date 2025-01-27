from sklearn.preprocessing import LabelEncoder
data['NOC'] = data['NOC'].replace(country_mapping)
data_1896_winners = data[(data['Year'] == 1896) & (data['Total'] > 0)]['NOC'].unique()
data = data[~data['NOC'].isin(data_1896_winners)]
country_features = data.groupby('NOC').agg({
    'Year': ['count', 'min', 'max'],
    'Female_Ratio': 'mean',
    'Num_Athletes': 'sum',
    'Num_Sports': 'sum',
    'Num_Events': 'sum',
    'Total': 'sum'
}).reset_index()
country_features.columns = [features]
country_features['Has_Won'] = country_features['Total_Medals'] > 0
le = LabelEncoder()
country_features['NOC'] = le.fit_transform(country_features['NOC'])
X = country_features.drop(columns=['Has_Won', 'Total_Medals'])
y = country_features['Has_Won']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=10000, random_state=42)
clf.fit(X_train, y_train)
country_features['Win_Probability'] = clf.predict_proba(X)[:, 1]
first_time_winners = country_features[country_features['Has_Won'] == False].sort_values(by='Win_Probability', ascending=False)
first_time_winners['NOC'] = le.inverse_transform(first_time_winners['NOC'])
num_countries_no_medal = country_features[country_features['Has_Won'] == False].shape[0]
num_countries_prob_above_3 = first_time_winners[first_time_winners['Win_Probability'] > 0.3].shape[0]
