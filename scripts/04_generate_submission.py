# ============================================================
# 04_generate_submission.py - 生成逻辑回归提交文件
# 目标：用全量训练集重新训练，对测试集生成预测结果
# ============================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- 1. 读取并处理训练数据 ---
print("1. 读取并处理训练集...")
train_df = pd.read_csv('../data/train.csv')

encode_cols = ['job', 'marital', 'education', 'default', 'housing',
               'loan', 'contact', 'month', 'day_of_week', 'poutcome']
train_encoded = pd.get_dummies(train_df, columns=encode_cols)
train_encoded['target'] = train_encoded['subscribe'].map({'yes': 1, 'no': 0})
X = train_encoded.drop(['target', 'subscribe'], axis=1)
y = train_encoded['target']

# --- 2. 全量训练 ---
# 提交阶段用全部训练数据，不保留验证集，最大化信息利用
print("2. 全量训练模型...")
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X, y)

# --- 3. 读取并处理测试集 ---
print("3. 读取测试集...")
test_df = pd.read_csv('../data/test.csv')
test_encoded = pd.get_dummies(test_df, columns=encode_cols)

# 列对齐：训练集中有的类别，测试集不一定都出现（如某月份无测试样本）
# reindex 以训练集列为标准，缺失列补 0，多余列丢弃
test_X = test_encoded.reindex(columns=X.columns, fill_value=0)
print(f"列数对齐：训练集 {X.shape[1]}，测试集 {test_X.shape[1]}（必须相等）")

# --- 4. 预测并生成提交文件 ---
print("4. 生成预测结果...")
pred_y = model.predict(test_X)

submission = pd.DataFrame()
submission['id'] = test_df['id']
submission['subscribe'] = np.where(pred_y == 1, 'yes', 'no')

submission.to_csv('../data/my_submission.csv', index=False)
print("提交文件已保存：data/my_submission.csv")
