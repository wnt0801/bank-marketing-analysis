# ============================================================
# 02_feature_engineering.py - 特征工程
# 目标：将原始文本特征转换为模型可用的数值格式
# ============================================================

import pandas as pd

# --- 1. 读取数据 ---
print("正在读取数据...")
df = pd.read_csv('../data/train.csv')

# --- 2. 异常值检查：通话时长为 0 ---
# duration=0 表示电话未接通，对预测无实际意义，记录数量供参考
zero_duration_count = (df['duration'] == 0).sum()
print(f"通话时长为 0 的记录数：{zero_duration_count}（占比 {zero_duration_count/len(df):.2%}）")

# --- 3. 识别需要编码的文本列 ---
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
print(f"需要转换的文本列：{categorical_columns}")

# --- 4. One-Hot 编码 ---
# 将分类变量展开为 0/1 列，例如 job_student、job_retired 等
# drop_first=True 删除每组第一个哑变量，避免多重共线性
encode_cols = ['job', 'marital', 'education', 'default', 'housing',
               'loan', 'contact', 'month', 'day_of_week', 'poutcome']
df_encoded = pd.get_dummies(df, columns=encode_cols, drop_first=True)

# --- 5. 目标变量编码 ---
# 将 yes/no 转换为 1/0，删除原始文本列
df_encoded['target'] = df_encoded['subscribe'].map({'yes': 1, 'no': 0})
df_encoded = df_encoded.drop('subscribe', axis=1)

print(f"\n编码完成：列数 {df.shape[1]} → {df_encoded.shape[1]}")
print(df_encoded.head())

# --- 6. 保存处理结果 ---
# 保存为新文件，不覆盖原始数据
df_encoded.to_csv('../data/train_processed.csv', index=False)
print("已保存：data/train_processed.csv")
