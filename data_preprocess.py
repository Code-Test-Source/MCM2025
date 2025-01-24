import pandas as pd
import chardet

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
medals = read_csv(medals_file_path)
programs = read_csv(programs_file_path)

# 检查所有数据文件缺失值并以零填充
print(athletes.isnull().sum())
athletes.fillna(0, inplace=True)

print(hosts.isnull().sum())
hosts.fillna(0, inplace=True)

print(programs.isnull().sum())
programs.fillna(0, inplace=True)

print(medals.isnull().sum())
medals.fillna(0, inplace=True)

# 检查所有数据文件重复值并删除

print(athletes.duplicated().sum())
athletes.drop_duplicates(inplace=True)

print(hosts.duplicated().sum())
hosts.drop_duplicates(inplace=True)

print(programs.duplicated().sum())
programs.drop_duplicates(inplace=True)

print(medals.duplicated().sum())
medals.drop_duplicates(inplace=True)


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
medals['NOC'] = medals['NOC'].replace(country_mapping)

# Remove ice sports and athletes playing ice sports
ice_sports = ['Figure Skating', 'Ice Hockey']
programs = programs[~programs['Sport'].isin(ice_sports)]
athletes = athletes[~athletes['Sport'].isin(ice_sports)]

# Remove medals from the year 1906（其实他已经帮你去掉好了）
medals = medals[medals['Year'] != 1906]

# 计算历年来奖牌前10的国家及其奖牌数
top_15_countries = medals.groupby('NOC').sum().sort_values(by='Total', ascending=False).head(10)

# 打印前15的国家及其奖牌数
print(top_15_countries[['Gold', 'Silver', 'Bronze', 'Total']])
#通过Medal栏非No medal计算运动员奖牌总数
athletes['Total'] = athletes['Medal'] != 'No medal'
athletes['Gold'] = athletes['Medal'] == 'Gold'
athletes['Silver'] = athletes['Medal'] == 'Silver'
athletes['Bronze'] = athletes['Medal'] == 'Bronze'

# 计算获得奖牌数前15的运动员以及他们的金银铜牌数
top_15_athletes = athletes.groupby('Name').sum().sort_values(by='Total', ascending=False).head(15)

# 打印前10的运动员及其奖牌数
print(top_15_athletes[['Gold', 'Silver', 'Bronze', 'Total']])
