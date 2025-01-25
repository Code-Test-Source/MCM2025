import pandas as pd

# 加载数据
file_path = "./2025_Problem_C_Data/summerOly_athletes.csv"
data = pd.read_csv(file_path)

# 过滤掉没有奖牌的数据
medal_data = data[data['Medal'] != 'No medal']

# 统计每个国家、体育项目、赛事的获奖数量
medal_counts = medal_data.groupby(['NOC', 'Sport', 'Event'])['Medal'].count().reset_index(name='Medal Count')

# 统计每个国家的总奖牌数
total_medals_per_country = medal_counts.groupby('NOC')['Medal Count'].sum().reset_index(name='Total Medals')

# 合并数据，计算每个国家每个体育项目/赛事的获奖占比
medal_data_with_totals = pd.merge(medal_counts, total_medals_per_country, on='NOC')
medal_data_with_totals['Medal Proportion'] = medal_data_with_totals['Medal Count'] / medal_data_with_totals['Total Medals']

# 设定阈值，筛选依赖于一项或少数几项体育/赛事的国家
threshold = 0.5  # 国家依赖的奖牌占比超过此值
reliant_countries = medal_data_with_totals[medal_data_with_totals['Medal Proportion'] > threshold]

# 按国家分组，找出每个国家获奖最多的前2个体育/赛事
reliant_countries_top_sports = reliant_countries.groupby('NOC').apply(lambda x: x.nlargest(2, 'Medal Proportion')).reset_index(drop=True)

# 输出结果
print(reliant_countries_top_sports[['NOC', 'Sport', 'Event', 'Medal Proportion']])
import matplotlib.pyplot as plt

# 选择依赖一项或少数几项体育/赛事的国家数据
reliant_countries_top_sports = reliant_countries_top_sports[['NOC', 'Sport', 'Event', 'Medal Proportion']]

# 创建一个可视化的柱状图
plt.figure(figsize=(12, 8))
for i, country in enumerate(reliant_countries_top_sports['NOC'].unique()):
    country_data = reliant_countries_top_sports[reliant_countries_top_sports['NOC'] == country]
    plt.bar(country_data['Sport'] + ' - ' + country_data['Event'], 
            country_data['Medal Proportion'], 
            label=country, alpha=0.7)

plt.xticks(rotation=90)  # 旋转x轴标签，避免重叠
plt.xlabel('Sport - Event')
plt.ylabel('Medal Proportion')
plt.title('Countries Relying on One or Two Sports/Events for Most Medals')
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
