from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
us_women_gymnastics = selection
legendary_years = [1981, 1984, 1989, 1996, 2000, 2004, 2008, 2012, 2016, 2020]
us_women_gymnastics.loc[:, 'Legendary_Coach'] = us_women_gymnastics['Year'].apply(lambda x: 1 if x in legendary_years else 0)
medal_counts = us_women_gymnastics.groupby(['Year', 'Legendary_Coach'])['Medal'].value_counts().unstack(fill_value=0)
medal_counts['Total'] = medal_counts[['Gold', 'Silver', 'Bronze']].sum(axis=1)
medal_counts = medal_counts[['Total', 'Gold', 'Silver', 'Bronze', 'No medal']]
medal_counts = medal_counts.reset_index()
sports_list = programs_df['Sport'].unique()
all_medal_counts = pd.DataFrame()
for sport in sports_list:
    sport_data = df[df['Sport'] == sport]
    grouped_data = sport_data.groupby(['NOC', 'Sex'])    
    for (noc, sex), group in grouped_data:
        # count medals like previous medal_counts and keep Sports NOC Sex columns
        all_medal_counts = pd.concat([all_medal_counts, medal_counts0], ignore_index=True)
X = medal_counts[['Year', 'Total', 'Gold', 'Silver', 'Bronze', 'No medal']]
y = medal_counts['Legendary_Coach']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
params = {
    # Included the Main Chapter before.
}
model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_test, y_test)], 
          verbose=False)
grouped_all_medal_counts = all_medal_counts.groupby(['Sport', 'NOC', 'Sex'])
legendary_coach_results = pd.DataFrame()
for (sport, noc, sex), group in grouped_all_medal_counts:
    group['Legendary_Index'] = model.predict(group[['Year', 'Total', 'Gold', 'Silver', 'Bronze', 'No medal']])
    legendary_threshold = 0.68
    group['Predicted_Legendary_Coach'] = group['Legendary_Index'] > legendary_threshold
    result = group[['Year','Sport', 'NOC', 'Sex', 'Legendary_Index', 'Predicted_Legendary_Coach']]
    legendary_coach_results = pd.concat([legendary_coach_results, result], ignore_index=True)
