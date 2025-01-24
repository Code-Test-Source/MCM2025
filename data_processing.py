import pandas as pd
import numpy as np
import chardet
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
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
'''
# 读取数据
data_dict = read_csv(file_dict_path)
athletes = read_csv(athletes_file_path)
hosts = read_csv(hosts_file_path)
medals = read_csv(medals_file_path)
programs = read_csv(programs_file_path)

# 数据类型转换
medals['Gold'] = medals['Gold'].astype(int)
medals['Silver'] = medals['Silver'].astype(int)
medals['Bronze'] = medals['Bronze'].astype(int)'''




# 1. 数据加载
file_path_1 = './2025_Problem_C_Data/participants_per_country_year.csv'
file_path_2 = './2025_Problem_C_Data/athlete_summary.csv'
file_path_3 = './2025_Problem_C_Data/summer_olympics_preprocessed.csv'

# 读取数据
participants_df = pd.read_csv(file_path_1)
athletes_summary_df = pd.read_csv(file_path_2)
olympics_df = pd.read_csv(file_path_3)

# 2. 数据预处理
# 计算每个国家每年参赛的运动员人数
athletes_per_country_year = olympics_df.groupby(['NOC', 'Year']).agg({'Name': 'nunique'}).reset_index()
athletes_per_country_year.rename(columns={'Name': 'Number_of_Athletes'}, inplace=True)

# 3. 合并数据
# 合并参赛运动员人数与奖牌数数据
df = olympics_df.groupby(['NOC', 'Year']).agg({'Gold': 'sum', 'Silver': 'sum', 'Bronze': 'sum'}).reset_index()
df = pd.merge(df, athletes_per_country_year, on=['NOC', 'Year'], how='left')

# 检查是否有缺失值
df.isnull().sum()

# 填充缺失值，或者根据需要进行处理
df.fillna(0, inplace=True)

# 4. PCA主成分分析
# 选择特征进行PCA，假设使用年份、运动员数、奖牌数等特征
features = ['Year', 'Number_of_Athletes', 'Gold', 'Silver', 'Bronze']

# 标准化数据
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# 应用PCA
pca = PCA(n_components=2)  # 降维至2个主成分
principal_components = pca.fit_transform(df_scaled)

# 将PCA结果添加到原数据中
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
df = pd.concat([df, df_pca], axis=1)

# 5. ARIMA模型预测
# 选择一个国家进行ARIMA建模，假设预测某个特定国家的奖牌数
country = "USA"  # 例如选择美国
country_data = df[df['NOC'] == country]

# 检查奖牌数的时间序列数据
medals = country_data[['Year', 'Gold', 'Silver', 'Bronze']].set_index('Year')

# 检查是否平稳
def check_stationarity(series):
    result = adfuller(series)
    return result[1] <= 0.05

# 这里以金牌为例进行ARIMA建模
if not check_stationarity(medals['Gold']):
    print("数据是非平稳的，需要差分处理")
    medals['Gold'] = medals['Gold'].diff().dropna()  # 一阶差分
    if not check_stationarity(medals['Gold']):
        print("差分后依然非平稳，可能需要进一步差分")

# 训练ARIMA模型
model = ARIMA(medals['Gold'], order=(5, 1, 0))  # 假设p=5, d=1, q=0
model_fit = model.fit()

# 预测未来奖牌数
forecast = model_fit.forecast(steps=4)  # 预测接下来的4年
print(f"未来4年金牌预测：{forecast}")

# 可视化预测结果
plt.plot(medals.index, medals['Gold'], label='历史金牌数')
plt.plot(range(medals.index[-1]+1, medals.index[-1]+5), forecast, label='预测金牌数', linestyle='--')
plt.xlabel('年份')
plt.ylabel('金牌数')
plt.legend()
plt.show()