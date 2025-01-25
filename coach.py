import pandas as pd
import numpy as np

# 读取数据
medal_counts = pd.read_csv('summerOly_medal_counts.csv')
athletes = pd.read_csv('summerOly_athletes.csv')
hosts = pd.read_csv('summerOly_hosts.csv')

# 假设我们有一个特定数据框来记录教练及其教练的表现
# 为演示目的，这里用模拟数据
# coach_effect_data = pd.DataFrame({
#     'Country': ['USA', 'China', 'Russia'],
#     'Medals_with_great_coach': [100, 80, 70],
#     'Medals_without_great_coach': [70, 50, 30]
# })

# Calculate the effect of the great coach
def calculate_coach_effect(df):
    df['Effect'] = df['Medals_with_great_coach'] - df['Medals_without_great_coach']
    return df

# 选择三个国家进行分析
selected_countries = ['USA', 'China', 'Russia']  # 可以替换为任何要分析的国家
coach_effect_data = pd.DataFrame({
    'Country': selected_countries,
    'Medals_with_great_coach': [100, 80, 70],
    'Medals_without_great_coach': [70, 50, 30]
})

# 进行效应计算
effect_data = calculate_coach_effect(coach_effect_data)

# 输出效果数据
print(effect_data)

# 计算投资的潜在影响
# 这是一个示例，具体影响可能基于更复杂的模型
investment_effects = []
for index, row in effect_data.iterrows():
    potential_increase = row['Effect'] * 0.1  # 假设投资影响的10%
    investment_effects.append({
        'Country': row['Country'],
        'Potential_Increase': potential_increase
    })

investment_summary = pd.DataFrame(investment_effects)
print(investment_summary)