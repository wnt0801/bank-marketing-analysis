# 银行客户认购预测分析
# Bank Marketing Subscription Prediction

## 项目背景

某银行通过电话营销推广定期存款产品，样本量 22,500 人，整体认购率 **13.12%**。

核心业务问题：**哪类客户更可能认购？营销资源应如何分配？**

---

## 数据集

- 来源：天池公开赛数据集（基于 UCI Bank Marketing Dataset）
- 训练集：22,500 条，测试集：7,500 条
- 特征：21 个，包含客户属性（年龄、职业、婚姻）、宏观经济指标（就业变动率、贷款利率）、营销行为（联系次数、上次联系距今天数）

---

## 分析结论

**1. 高转化客群**
- 学生（32.98%）与退休人群（27.04%）认购率显著高于均值，适合精准营销
- 单身客户（16.44%）转化率高于已婚客户（10.71%），且规模足够大，是核心转化来源

**2. 低效投入群体**
- 蓝领客户规模最大（4,874 人），但认购率仅 7.55%，低于均值近一半
- 已婚 × 无贷款组合（10,710 人）认购率 10.25%，占用营销资源最多但转化最低

**3. 营销行为的反直觉发现**
- 联系次数增加至 4–5 次后转化率反而下降（9.41%），过度触达降低客户意愿
- `duration`（通话时长）与认购正相关，但属于**事中变量**——长通话是高意向的结果，而非原因，不能用于事前营销决策，建模时主动排除以避免数据泄露

**4. 宏观环境是最强信号**
- 随机森林特征重要性第一位是 `emp_var_rate`（就业变动率），说明宏观经济环境对客户金融决策影响大于个人特征

---

## 模型结果

| 模型 | AUC | 说明 |
|------|-----|------|
| Logistic Regression（baseline） | 0.7978 | `class_weight='balanced'` 处理类别不平衡 |
| Random Forest（主模型） | **0.8238** | 排除 `duration` 数据泄露，200 棵树 |

> AUC 含义：随机抽取一名认购客户和一名未认购客户，模型有 82.4% 的概率将认购客户排在前面。
>
> RF 在 AUC 上优于 LR baseline，但对少数类召回率较低（recall=0.13）。实际业务部署中需结合阈值调整或 `class_weight` 优化，平衡精准率与覆盖率。

---

## 项目结构

```
├── data/
│   ├── train.csv           # 训练集（原始）
│   └── test.csv            # 测试集（原始）
├── scripts/
│   ├── 01_eda_analysis.py          # 探索性数据分析
│   ├── 02_feature_engineering.py   # 特征工程 & One-Hot 编码
│   ├── 03_model_training.py        # 逻辑回归 baseline
│   ├── 04_generate_submission.py   # LR 提交文件生成
│   └── 05_random_forest.py         # 随机森林主模型 + 特征重要性
├── images/
│   ├── job_vs_subscription.png
│   ├── duration_vs_subscription.png
│   └── feature_importance_rf.png
└── README.md
```

---

## 运行方式

```bash
# 按顺序执行
python scripts/01_eda_analysis.py
python scripts/02_feature_engineering.py
python scripts/03_model_training.py
python scripts/05_random_forest.py
```

依赖：`pandas` `scikit-learn` `matplotlib` `seaborn`

---

## 技术栈

`Python` · `scikit-learn` · `pandas` · `matplotlib` · `seaborn`