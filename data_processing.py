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


# 数据类型转换
medals['Gold'] = medals['Gold'].astype(int)
medals['Silver'] = medals['Silver'].astype(int)
medals['Bronze'] = medals['Bronze'].astype(int)

# 特征工程：例如计算奖牌总数
medals['Total'] = medals['Gold'] + medals['Silver'] + medals['Bronze']

# 打印处理后的数据
print(medals.head())
