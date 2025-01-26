import pandas as pd
import chardet
import seaborn as sns

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
top_10_countries = medals.groupby('NOC').sum().sort_values(by='Total', ascending=False).head(10)

# 打印前10的国家及其奖牌数
print(top_10_countries[['Gold', 'Silver', 'Bronze', 'Total']])

# 通过Medal栏非No medal计算运动员奖牌总数
athletes['Total'] = (athletes['Medal'] != 'No medal').astype(int)
athletes['Gold'] = (athletes['Medal'] == 'Gold').astype(int)
athletes['Silver'] = (athletes['Medal'] == 'Silver').astype(int)
athletes['Bronze'] = (athletes['Medal'] == 'Bronze').astype(int)

# 计算获得奖牌数前15的运动员以及他们的金银铜牌数
top_15_athletes = athletes.groupby('Name').sum().sort_values(by='Total', ascending=False).head(10)

# 打印前10的运动员及其奖牌数
print(top_15_athletes[['Gold', 'Silver', 'Bronze', 'Total']])

# 绘制前10国家奖牌数的表格
print("Top 10 Countries by Medal Count")
print(top_10_countries[['Gold', 'Silver', 'Bronze', 'Total']].to_string())

# 绘制前10运动员奖牌数的表格
print("Top 10 Athletes by Medal Count")
print(top_15_athletes[['Gold', 'Silver', 'Bronze', 'Total']].to_string())
import matplotlib.pyplot as plt
import matplotlib.table as tbl


# Function to create a table plot
def plot_table(data, title, filename):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axis('tight')
    ax.axis('off')
    table = tbl.table(ax, cellText=data.values, colLabels=data.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.25, 1.6)  # Increase the cell height

    # Set a nicer font
    for key, cell in table.get_celld().items():
        cell.set_fontsize(12)
        cell.set_text_props(ha='center', va='center')  # Center align text

    # Alternate row colors
    for i, key in enumerate(table.get_celld()):
        if key[0] == 0:
            table[key].set_fontsize(14)
        if key[0] % 2 == 0 and key[0] != 0:
            table[key].set_facecolor('#f0f0f0')

    # Center the table in the figure without adjusting the original size
    table.auto_set_column_width(col=list(range(len(data.columns))))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    plt.title(title, fontsize=18, pad=0)  # Increase title size and reduce padding
    plt.savefig(filename)
    plt.close()

# Plot top 10 countries by medal count
top_10_countries.reset_index(inplace=True)
plot_table(top_10_countries[['NOC', 'Gold', 'Silver', 'Bronze', 'Total']], "Top 10 Countries by Medal Count", "top_10_countries.png")

# Plot top 10 athletes by medal count
top_15_athletes.reset_index(inplace=True)
plot_table(top_15_athletes[['Name', 'Gold', 'Silver', 'Bronze', 'Total']], "Top 10 Athletes by Medal Count", "top_10_athletes.png")