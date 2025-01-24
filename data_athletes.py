import pandas as pd
import chardet

# 文件路径
athletes_file_path = r".\2025_Problem_C_Data\summerOly_athletes.csv"


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

athletes = read_csv(athletes_file_path)




# Replace 'Team' with 'Country' in athletes dataset
athletes.rename(columns={'Team': 'Country'}, inplace=True)

# Map old country names to current country names

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

athletes['NOC'] = athletes['NOC'].replace(noc_mapping)

# Remove ice sports and athletes playing ice sports
ice_sports = ['Figure Skating', 'Ice Hockey']
athletes = athletes[~athletes['Sport'].isin(ice_sports)]

# Calculate the number of participants per country per year
participants_per_country_year = athletes.groupby(['Year', 'NOC']).size().reset_index(name='Participants')
print(participants_per_country_year)

# Export participants per country per year to CSV
participants_per_country_year.to_csv(r'./participants_per_country_year.csv', index=False)

# Calculate the number of awards per athlete
awards_per_athlete = athletes.dropna(subset=['Medal']).groupby('Name')['Medal'].count().reset_index(name='Awards')
print(awards_per_athlete)

# Calculate the number of participations per athlete
participations_per_athlete = athletes.groupby('Name').size().reset_index(name='Participations')
print(participations_per_athlete)

# Calculate the number of each type of medal per athlete
medals_per_athlete = athletes.dropna(subset=['Medal']).groupby(['Name', 'Medal']).size().unstack(fill_value=0).reset_index()
print(medals_per_athlete)

# Merge awards, participations, and medals into a single DataFrame
athlete_summary = participations_per_athlete.merge(awards_per_athlete, on='Name', how='left').merge(medals_per_athlete, on='Name', how='left').fillna(0)
print(athlete_summary)

# Export athlete summary to CSV
athlete_summary.to_csv(r'./athlete_summary.csv', index=False)