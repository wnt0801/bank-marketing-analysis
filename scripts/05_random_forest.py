# ============================================================
# 05_random_forest.py - 随机森林模型（主模型）
# 目标：提升预测性能，分析真实特征重要性
# 注意：主动排除 duration（事中变量，存在数据泄露风险）
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# --- 1. 读取数据 ---
print("1. 读取并处理数据...")
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

# --- 2. 合并处理（统一特征工程，避免训练集/测试集列不一致）---
train_df['is_train'] = 1
test_df['is_train'] = 0
test_df['subscribe'] = 'no'   # 补占位列，仅用于拼接，不参与训练

full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

encode_cols = ['job', 'marital', 'education', 'default', 'housing',
               'loan', 'contact', 'month', 'day_of_week', 'poutcome']
full_encoded = pd.get_dummies(full_df, columns=encode_cols)

# --- 3. 拆分训练集与测试集 ---
# 主动排除 duration：该变量在通话结束后才能获得，不能用于事前预测客户
# 保留 duration 会造成数据泄露，导致模型在真实场景中失效
drop_cols = ['is_train', 'subscribe', 'duration', 'id']

train_final = full_encoded[full_encoded['is_train'] == 1].drop(drop_cols, axis=1)
test_final  = full_encoded[full_encoded['is_train'] == 0].drop(drop_cols, axis=1)
y = train_df['subscribe'].map({'yes': 1, 'no': 0})

print(f"训练集维度：{train_final.shape}，测试集维度：{test_final.shape}")

# --- 4. 切分验证集，评估真实模型性能 ---
X_train, X_val, y_train, y_val = train_test_split(
    train_final, y, test_size=0.2, random_state=42, stratify=y
)

# --- 5. 训练随机森林 ---
print("2. 训练随机森林...")
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# --- 6. 验证集评估 ---
y_val_pred = model.predict(X_val)
y_val_prob = model.predict_proba(X_val)[:, 1]

print("\n--- 验证集评估报告（已排除 duration）---")
print(f"AUC：{roc_auc_score(y_val, y_val_prob):.4f}")
print(classification_report(y_val, y_val_pred))

# --- 7. 特征重要性分析 ---
feature_imp = pd.DataFrame({
    'Feature': train_final.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

top_10 = feature_imp.head(10)
print("\n--- 随机森林 TOP 10 关键特征（不含 duration）---")
print(top_10.to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_10,
            hue='Feature', palette='magma', legend=False)
plt.title('Top 10 Feature Importances (Duration Excluded)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('../images/feature_importance_rf.png', dpi=150)
plt.close()
print("已保存：images/feature_importance_rf.png")

# --- 8. 用全量数据重新训练，生成最终提交文件 ---
print("\n3. 全量训练，生成提交文件...")
model.fit(train_final, y)
pred_y = model.predict(test_final)

submission = pd.DataFrame({
    'id': test_df['id'],
    'subscribe': pd.Series(pred_y).map({1: 'yes', 0: 'no'})
})
submission.to_csv('../data/submission_rf.csv', index=False)
print("提交文件已保存：data/submission_rf.csv")
