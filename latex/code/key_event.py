medal_data = data[data['Medal'] != 'No medal']
medal_counts = medal_data.groupby(['NOC', 'Sport', 'Event'])['Medal'].count().reset_index(name='Medal Count')
total_medals_per_country = medal_counts.groupby('NOC')['Medal Count'].sum().reset_index(name='Total Medals')
medal_data_with_totals = pd.merge(medal_counts, total_medals_per_country, on='NOC')
medal_data_with_totals['Medal Proportion'] = medal_data_with_totals['Medal Count'] / medal_data_with_totals['Total Medals']
threshold = 0.5
reliant_countries = medal_data_with_totals[medal_data_with_totals['Medal Proportion'] > threshold]
reliant_countries_top_sports = reliant_countries.groupby('NOC').apply(lambda x: x.nlargest(2, 'Medal Proportion')).reset_index(drop=True)


