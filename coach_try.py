import pandas as pd
import chardet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score



# 读取CSV文件
df = pd.read_csv('./2025_Problem_C_Data/summerOly_athletes.csv')

# 筛选美国女子体操队的数据
us_women_gymnastics = df[(df['NOC'] == 'USA') & ((df['Sport'] == 'Gymnastics')) & (df['Sex'] == 'F')]

# 2020 2024的数据因为项目发生变化因此有些反常就不拿进去训练了

# 将筛选后的数据保存到另一个CSV文件
us_women_gymnastics.to_csv('./2025_Problem_C_Data/us_women_gymnastics.csv', index=False)

# 添加一个新列，标记是否有传奇教练执教
legendary_years = [1976, 1981, 1984, 1989, 1996, 2000, 2004, 2008, 2012, 2016, 2020]
us_women_gymnastics.loc[:, 'Legendary_Coach'] = us_women_gymnastics['Year'].apply(lambda x: 1 if x in legendary_years else 0)

# 统计每年传奇教练对奖牌数的影响
medal_counts = us_women_gymnastics.groupby(['Year', 'Legendary_Coach'])['Medal'].value_counts().unstack(fill_value=0)
medal_counts['Total'] = medal_counts[['Gold', 'Silver', 'Bronze']].sum(axis=1)
medal_counts = medal_counts[['Total', 'Gold', 'Silver', 'Bronze', 'No medal']]
medal_counts = medal_counts.reset_index()

# 保存medal_counts到CSV文件
medal_counts.to_csv('./2025_Problem_C_Data/medal_counts.csv', index=False)

print(medal_counts)



# 读取programs.csv文件
programs_df = pd.read_csv('./2025_Problem_C_Data/summerOly_programs.csv')

# 获取所有的Sports
sports_list = programs_df['Sport'].unique()

# 初始化一个空的DataFrame来存储所有的medal_counts
all_medal_counts = pd.DataFrame()

# 遍历每一个Sport
for sport in sports_list:
    # 筛选出当前Sport的数据
    sport_data = df[df['Sport'] == sport]
    
    # 按NOC和Sex分类
    grouped_data = sport_data.groupby(['NOC', 'Sex'])
    
    for (noc, sex), group in grouped_data:
        # 统计每年奖牌数
        medal_counts0 = group.groupby('Year')['Medal'].value_counts().unstack(fill_value=0)
        medal_counts0 = medal_counts0.reindex(columns=['Gold', 'Silver', 'Bronze', 'No medal'], fill_value=0)
        medal_counts0['Total'] = medal_counts0[['Gold', 'Silver', 'Bronze']].sum(axis=1)
        medal_counts0 = medal_counts0[['Total', 'Gold', 'Silver', 'Bronze', 'No medal']].reset_index()
        # 添加年份列
        medal_counts0 = medal_counts0.reset_index()
        
        # # 添加Sport, NOC, Sex列
        medal_counts0['Sport'] = sport
        medal_counts0['NOC'] = noc
        medal_counts0['Sex'] = sex
        
        # 将当前的medal_counts添加到all_medal_counts中
        all_medal_counts = pd.concat([all_medal_counts, medal_counts0], ignore_index=True)

# 保存所有的medal_counts到CSV文件
all_medal_counts.to_csv('./2025_Problem_C_Data/medal_all_counts.csv', index=False)

print(all_medal_counts)


# # medal_counts作为传奇指数（Legendary_Coach）的训练集,训练能够针对输入'Year'，'Total', 'Gold', 'Silver', 'Bronze', 'No medal'计算传奇指数的模型，示例输入[2020,13,8,4,1,8]

# 准备训练数据
X = medal_counts[['Year', 'Total', 'Gold', 'Silver', 'Bronze', 'No medal']]
y = medal_counts['Legendary_Coach']

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型

# 设置参数
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.01,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 1000,
    'random_state': 42,
    'early_stopping_rounds': 10
}

# 训练模型
model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_test, y_test)], 
          verbose=False)


# 预测

# 对于每个Sport, NOC, Sex分类做传奇教练指数计算与判断
grouped_all_medal_counts = all_medal_counts.groupby(['Sport', 'NOC', 'Sex'])

# 初始化一个空的DataFrame来存储结果
legendary_coach_results = pd.DataFrame()

for (sport, noc, sex), group in grouped_all_medal_counts:
    # 计算传奇教练指数
    group['Legendary_Index'] = model.predict(group[['Year', 'Total', 'Gold', 'Silver', 'Bronze', 'No medal']])
    
    # 如果legendary_index > 1则算作传奇教练数据
    legendary_threshold = 0.68
    
    # 判断是否为传奇教练
    group['Predicted_Legendary_Coach'] = group['Legendary_Index'] > legendary_threshold
    
    # 选择需要的列
    result = group[['Year','Sport', 'NOC', 'Sex', 'Legendary_Index', 'Predicted_Legendary_Coach']]
    
    # 将结果添加到legendary_coach_results中
    legendary_coach_results = pd.concat([legendary_coach_results, result], ignore_index=True)

# 保存结果到CSV文件
legendary_coach_results.to_csv('./2025_Problem_C_Data/legendary_coach_results.csv', index=False)

# 将结果为真的数据保存到CSV文件
legendary_coach_results_true = legendary_coach_results[legendary_coach_results['Predicted_Legendary_Coach'] == True]
legendary_coach_results_true.to_csv('./2025_Problem_C_Data/legendary_coach_results_true.csv', index=False)

print(legendary_coach_results_true)

