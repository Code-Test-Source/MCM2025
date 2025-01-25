import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
participants_df = pd.read_csv("./2025_Problem_C_Data/participants_per_country_year.csv")
medals_df = pd.read_csv("./2025_Problem_C_Data/summerOly_medal_counts.csv")

# 数据预处理
medals_df['HasMedal'] = medals_df['Total'].notna()

# 计算每个国家首次获奖的年份
first_medals = medals_df[medals_df['HasMedal']].groupby('NOC').agg({
    'Year': 'min',
    'Total': 'first'
}).reset_index()
first_medals.columns = ['NOC', 'FirstMedalYear', 'FirstTotal']

# 过滤1896年后首次获奖的国家
first_medals = first_medals[first_medals['FirstMedalYear'] > 1896].copy()
print(f"Countries with first medals after 1896: {len(first_medals)}")

# 构建特征
def extract_features(df, noc, end_year):
    country_data = df[
        (df['NOC'] == noc) & 
        (df['Year'] <= end_year)
    ]
    if len(country_data) == 0:
        return None
        
    recent_data = country_data[country_data['Year'] >= end_year - 12]
    
    return {
        'avg_participants': country_data['Participants'].mean(),
        'recent_avg_participants': recent_data['Participants'].mean(),
        'max_participants': country_data['Participants'].max(),
        'total_participations': len(country_data),
        'participation_rate': len(country_data) / (end_year - country_data['Year'].min() + 1),
        'years_active': end_year - country_data['Year'].min()
    }

# 构建训练数据
training_features = []
training_labels = []

for _, row in first_medals.iterrows():
    features = extract_features(participants_df, row['NOC'], row['FirstMedalYear'] - 4)
    if features:
        training_features.append(features)
        training_labels.append(1)

print(f"Training samples created: {len(training_features)}")

# 转换为DataFrame
X_train = pd.DataFrame(training_features)
y_train = np.array(training_labels)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 训练模型
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# 准备预测数据
no_medal_nocs = set(participants_df['NOC']) - set(medals_df[medals_df['HasMedal']]['NOC'])
prediction_features = []
valid_nocs = []

for noc in no_medal_nocs:
    features = extract_features(participants_df, noc, 2024)
    if features:
        prediction_features.append(features)
        valid_nocs.append(noc)

# 预测
X_pred = pd.DataFrame(prediction_features)
X_pred_scaled = scaler.transform(X_pred)
probabilities = model.predict_proba(X_pred_scaled)[:, 1]

# 结果整理
results = pd.DataFrame({
    'NOC': valid_nocs,
    'Probability': probabilities
})
results = results.sort_values('Probability', ascending=False)

print("\nTop 10 countries most likely to win their first medal in 2028:")
print(results.head(10))

# 可视化
plt.figure(figsize=(15, 8))
plt.barh(results.head(20)['NOC'], results.head(20)['Probability'])
plt.title('Top 20 Countries Most Likely to Win Their First Medal in 2028')
plt.xlabel('Probability')
plt.ylabel('Country Code')
plt.tight_layout()
plt.show()

# 保存结果
results.to_csv('2028_medal_predictions.csv', index=False)