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

# 检查缺失值
print(medals.isnull().sum())

# 填充或删除缺失值
medals.fillna(0, inplace=True)  # 用0填充缺失值


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

athletes['Country'] = athletes['Country'].replace(country_mapping)
medals['NOC'] = medals['NOC'].replace(country_mapping)

# Remove ice sports and athletes playing ice sports
ice_sports = ['Figure Skating', 'Ice Hockey']
programs = programs[~programs['Sport'].isin(ice_sports)]
athletes = athletes[~athletes['Sport'].isin(ice_sports)]

# 计算历年来奖牌前10的国家及其奖牌数
top_10_countries = medals.groupby('NOC').sum().sort_values(by='Total', ascending=False).head(10)

# 打印前10的国家及其奖牌数
print(top_10_countries[['Gold', 'Silver', 'Bronze', 'Total']])
