# ============================================================
# 03_model_training.py - 逻辑回归基线模型
# 目标：建立 baseline，验证特征有效性，输出可解释的系数
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# --- 1. 读取处理后的数据 ---
print("正在读取处理后的数据...")
df = pd.read_csv('../data/train_processed.csv')

# --- 2. 分离特征与目标变量 ---
y = df['target']
X = df.drop('target', axis=1)

# --- 3. 切分训练集与测试集 ---
# stratify=y：保证切分后两份数据的正负样本比例一致
# 在类别不平衡数据（认购率约 13%）中这一步很关键
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"训练集：{X_train.shape}，测试集：{X_test.shape}")

# --- 4. 训练逻辑回归模型 ---
# class_weight='balanced'：对少数类（认购=1）自动增加权重
# 不加这个参数，模型可能全部预测"不买"，准确率虚高但没有实际意义
model = LogisticRegression(max_iter=2000, class_weight='balanced')
print("正在训练模型...")
model.fit(X_train, y_train)

# --- 5. 模型评估 ---
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- 模型评估报告 ---")
print(f"准确率 (Accuracy)：{accuracy_score(y_test, y_pred):.4f}")
print(f"AUC（越接近 1 越好）：{roc_auc_score(y_test, y_prob):.4f}")
print("\n详细分类报告：")
print(classification_report(y_test, y_pred))

# --- 6. 特征系数分析 ---
# 逻辑回归系数越大，表示该特征对认购的正向影响越强
weights = pd.Series(model.coef_[0], index=X.columns)
print("--- 正向影响 TOP 5 特征 ---")
print(weights.sort_values(ascending=False).head(5))
print("\n--- 负向影响 TOP 5 特征 ---")
print(weights.sort_values(ascending=True).head(5))
