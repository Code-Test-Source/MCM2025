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

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 假设您已经成功加载并预处理了数据
# 特征选择：选择与金牌数和总奖牌数相关的列
# 假设我们使用 'Gold', 'Silver', 'Bronze' 等作为特征，可以根据需要选择其他特征
X = medals[['Gold', 'Silver', 'Bronze']]  # 例如使用金银铜牌数量作为特征
y_gold = medals['Gold']  # 预测金牌数
y_total = medals['Total']  # 预测总奖牌数

# 数据集拆分，80% 训练集，20% 测试集
X_train, X_test, y_train_gold, y_test_gold = train_test_split(X, y_gold, test_size=0.2, random_state=42)
X_train_total, X_test_total, y_train_total, y_test_total = train_test_split(X, y_total, test_size=0.2, random_state=42)

# 训练线性回归模型
regressor_gold = LinearRegression()
regressor_total = LinearRegression()

# 拟合模型
regressor_gold.fit(X_train, y_train_gold)
regressor_total.fit(X_train_total, y_train_total)

# 用测试集数据进行预测
y_pred_gold = regressor_gold.predict(X_test)
y_pred_total = regressor_total.predict(X_test_total)

# 评估模型：计算均方误差（MSE）
mse_gold = mean_squared_error(y_test_gold, y_pred_gold)
mse_total = mean_squared_error(y_test_total, y_pred_total)

print(f'Mean Squared Error (Gold): {mse_gold}')
print(f'Mean Squared Error (Total): {mse_total}')

# 打印模型系数和截距
print(f"Gold Medal Model Coefficients: {regressor_gold.coef_}")
print(f"Gold Medal Model Intercept: {regressor_gold.intercept_}")
print(f"Total Medal Model Coefficients: {regressor_total.coef_}")
print(f"Total Medal Model Intercept: {regressor_total.intercept_}")

# 输出预测结果
print("Predicted Gold Medals for test data:", y_pred_gold)
print("Actual Gold Medals for test data:", y_test_gold.values)
