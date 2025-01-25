import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 加载数据 (保持之前的数据加载和清洗代码不变)
athletes_df = pd.read_csv('summerOly_athletes.csv', encoding='utf-8')
medal_counts_df = pd.read_csv('summerOly_medal_counts.csv', encoding='utf-8')
medal_counts_df = medal_counts_df[~medal_counts_df['NOC'].isin(['Mixed team', 'ANZ'])]
medal_counts_df['Year'] = medal_counts_df['Year'].astype(int)
athletes_df['Year'] = athletes_df['Year'].astype(int)


# 特征工程 -  加入历史奖牌特征
def create_historical_features(df, years_ago_list=[4, 8, 12]): #  考虑过去 4, 8, 12 年的奥运会
    df_features = df.copy()
    for years_ago in years_ago_list:
        df_shifted = df.copy()
        df_shifted['Year'] = df_shifted['Year'] + years_ago #  年份向前平移
        df_shifted = df_shifted[['Year', 'NOC', 'Gold', 'Total']] #  只保留需要的列
        df_shifted.rename(columns={'Gold': f'Gold_Past_{years_ago}Y', 'Total': f'Total_Past_{years_ago}Y'}, inplace=True) #  重命名列名
        df_features = pd.merge(df_features, df_shifted, on=['Year', 'NOC'], how='left') #  左连接合并
    return df_features

medal_counts_featured_df = create_historical_features(medal_counts_df)

# 填充缺失值 (历史数据可能缺失，用 0 填充)
medal_counts_featured_df.fillna(0, inplace=True)

# 准备训练数据 (与之前的代码基本相同，但使用新的特征数据集)
features = ['Year', 'Gold_Past_4Y', 'Total_Past_4Y', 'Gold_Past_8Y', 'Total_Past_8Y', 'Gold_Past_12Y', 'Total_Past_12Y'] #  加入历史奖牌特征
target_gold = 'Gold'
target_total = 'Total'

data = medal_counts_featured_df[['Year', 'NOC', 'Gold', 'Total'] + features[1:]].copy() #  选择需要的列
data = pd.get_dummies(data, columns=['NOC']) # One-Hot 编码

train_data = data[data['Year'] < 2024].copy()
test_data = data[data['Year'] == 2024].copy()

X_train = train_data.drop([target_gold, target_total], axis=1)
y_gold_train = train_data[target_gold]
y_total_train = train_data[target_total]

X_test = test_data.drop([target_gold, target_total], axis=1)
y_gold_test = test_data[target_gold]
y_total_test = test_data[target_total]


print("\n特征工程后的训练特征 (X_train) 前几行:")
print(X_train.head())

# 模型训练 - 随机森林回归 (代码与之前相同)
model_gold = RandomForestRegressor(random_state=42)
model_total = RandomForestRegressor(random_state=42)

model_gold.fit(X_train, y_gold_train)
model_total.fit(X_train, y_total_train)

# 模型预测 (代码与之前相同)
y_gold_pred = model_gold.predict(X_test)
y_total_pred = model_total.predict(X_test)

# 模型评估 (代码与之前相同)
mse_gold = mean_squared_error(y_gold_test, y_gold_pred)
r2_gold = r2_score(y_gold_test, y_gold_pred)
mse_total = mean_squared_error(y_total_test, y_total_pred)
r2_total = r2_score(y_total_test, y_total_pred)

print("\n金牌数模型评估 (更新特征后):")
print(f"均方误差 (MSE): {mse_gold:.2f}")
print(f"R 平方值 (R^2): {r2_gold:.2f}")

print("\n总奖牌数模型评估 (更新特征后):")
print(f"均方误差 (MSE): {mse_total:.2f}")
print(f"R 平方值 (R^2): {r2_total:.2f}")

def bootstrap_predict(model, X_test, n_bootstrap=1000): #  Bootstrap 预测函数
    predictions = []
    for _ in range(n_bootstrap):
        # 从训练数据集中有放回地抽样
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_sample = X_train.iloc[indices]
        y_sample_gold = y_gold_train.iloc[indices]
        y_sample_total = y_total_train.iloc[indices]

        # 训练模型
        model_sample_gold = RandomForestRegressor(random_state=42) #  每次 bootstrap 训练一个新的模型
        model_sample_total = RandomForestRegressor(random_state=42)
        model_sample_gold.fit(X_sample, y_sample_gold)
        model_sample_total.fit(X_sample, y_sample_total)

        # 预测测试集
        y_pred_gold_bootstrap = model_sample_gold.predict(X_test)
        y_pred_total_bootstrap = model_sample_total.predict(X_test)
        predictions.append(np.column_stack((y_pred_gold_bootstrap, y_pred_total_bootstrap))) #  保存每次 bootstrap 的预测结果

    predictions = np.concatenate(predictions, axis=0) #  将所有 bootstrap 的预测结果合并
    return predictions


bootstrap_preds = bootstrap_predict(RandomForestRegressor(random_state=42), X_test) #  调用 bootstrap 预测函数

# 计算预测区间 (95% 置信区间)
lower_percentile = 2.5
upper_percentile = 97.5

y_gold_pred_lower = np.percentile(bootstrap_preds[:, 0], lower_percentile, axis=0) #  金牌数预测区间下界
y_gold_pred_upper = np.percentile(bootstrap_preds[:, 0], upper_percentile, axis=0) #  金牌数预测区间上界
y_total_pred_lower = np.percentile(bootstrap_preds[:, 1], lower_percentile, axis=0) #  总奖牌数预测区间下界
y_total_pred_upper = np.percentile(bootstrap_preds[:, 1], upper_percentile, axis=0) #  总奖牌数预测区间上界


# 查看带预测区间的结果
predictions_df_with_interval = pd.DataFrame({
    'Actual_Gold': y_gold_test,
    'Predicted_Gold': y_gold_pred.round(0),
    'Gold_Lower_95CI': y_gold_pred_lower.round(0),
    'Gold_Upper_95CI': y_gold_pred_upper.round(0),
    'Actual_Total': y_total_test,
    'Predicted_Total': y_total_pred.round(0),
    'Total_Lower_95CI': y_total_pred_lower.round(0),
    'Total_Upper_95CI': y_total_pred_upper.round(0)
}, index = test_data.filter(like='NOC_').columns.str.replace('NOC_', '')) # 使用 One-Hot 编码后的 NOC 列名作为索引，并去除前缀

print("\n2024 年奖牌预测结果 (部分国家) - 带 95% 预测区间:")
print(predictions_df_with_interval.head(10))
