# ============================================================
# 01_eda_analysis.py - 探索性数据分析 (EDA)
# 目标：理解数据分布，找出影响客户认购的关键特征
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# --- 1. 读取数据 ---
print("正在读取数据...")
df = pd.read_csv('../data/train.csv')
print("列名检查：", df.columns.tolist())

# --- 2. 目标变量分布 ---
# 认购率约 13%，存在明显类别不平衡，后续建模需要处理
print("\n--- 客户认购比例 (Target Distribution) ---")
print(df['subscribe'].value_counts(normalize=True))

# --- 3. 职业 vs 认购率 ---
# 业务假设：学生和退休人群时间充裕、风险偏好稳健，转化率可能更高
plt.figure(figsize=(12, 6))
sns.countplot(x='job', hue='subscribe', data=df, palette='viridis')
plt.title('Job vs Subscription Rate')
plt.xlabel('Job Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../images/job_vs_subscription.png', dpi=150)
plt.close()
print("已保存：images/job_vs_subscription.png")

# --- 4. 通话时长 vs 认购率 ---
# 注意：duration 是事中变量，不能用于事前预测
# 通话越长说明客户意愿更强，但不能反推"打久一点就能提升转化"
# 该指标更适合作为意向识别信号，而非营销决策依据
plt.figure(figsize=(10, 6))
sns.boxplot(x='subscribe', y='duration', data=df)
plt.ylim(0, 2000)   # 去掉极端值干扰，聚焦主体分布
plt.title('Call Duration vs Subscription')
plt.xlabel('Subscribed')
plt.ylabel('Duration (seconds)')
plt.tight_layout()
plt.savefig('../images/duration_vs_subscription.png', dpi=150)
plt.close()
print("已保存：images/duration_vs_subscription.png")
