import pandas as pd
import chardet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

# # 检查所有数据文件缺失值并以零填充
# print(athletes.isnull().sum())
# athletes.fillna(0, inplace=True)

# print(hosts.isnull().sum())
# hosts.fillna(0, inplace=True)

# print(programs.isnull().sum())
# programs.fillna(0, inplace=True)

# print(medals.isnull().sum())
# medals.fillna(0, inplace=True)

# # 检查所有数据文件重复值并删除

# print(athletes.duplicated().sum())
# athletes.drop_duplicates(inplace=True)

# print(hosts.duplicated().sum())
# hosts.drop_duplicates(inplace=True)

# print(programs.duplicated().sum())
# programs.drop_duplicates(inplace=True)

# print(medals.duplicated().sum())
# medals.drop_duplicates(inplace=True)


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

countries = medals['NOC'].unique()  # 获取所有国家
models = {}  # 用于存储每个国家的回归模型

# 遍历每个国家，构建回归模型
for country in countries:
    # 过滤出该国家的数据
    country_data = medals[medals['NOC'] == country]
    if len(country_data) < 2:
        print(f"Skipping country {country} due to insufficient data.")
        continue
    # 特征选择：年份作为自变量
    X = country_data[['Year']]
    y_gold = country_data['Gold']
    y_total = country_data['Total']
    
    # 数据集拆分
    X_train, X_test, y_train_gold, y_test_gold = train_test_split(X, y_gold, test_size=0.2, random_state=42)
    X_train_total, X_test_total, y_train_total, y_test_total = train_test_split(X, y_total, test_size=0.2, random_state=42)
    
    # 创建线性回归模型
    model_gold = LinearRegression()
    model_total = LinearRegression()
    
    # 训练模型
    model_gold.fit(X_train, y_train_gold)
    model_total.fit(X_train_total, y_train_total)
    
    # 预测
    y_pred_gold = model_gold.predict(X_test)
    y_pred_total = model_total.predict(X_test_total)
    
    # 评估模型：计算均方误差（MSE）
    mse_gold = mean_squared_error(y_test_gold, y_pred_gold)
    mse_total = mean_squared_error(y_test_total, y_pred_total)
    
    # 保存模型和MSE
    models[country] = {
        'model_gold': model_gold,
        'model_total': model_total,
        'mse_gold': mse_gold,
        'mse_total': mse_total
    }
    
    # 输出每个国家的评估结果
    print(f"Country: {country}")
    print(f"Gold Medal Model MSE: {mse_gold}")
    print(f"Total Medal Model MSE: {mse_total}")
    print('-' * 50)

# 在此之后，您可以选择查看某个国家的回归系数
# 比如，查看美国的金牌回归系数：
us_model_gold = models['United States']['model_gold']
print(f"United States Gold Medal Model Coefficients: {us_model_gold.coef_}")