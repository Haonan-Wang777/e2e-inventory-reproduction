# 公开子样本实证复现

当前脚本重新读取四个原始 CSV（处理组/对照组、实验前/后，各 500 行），独立重算统计量。结果并非论文完整样本结果，论文原表仅在 paper_vs_public_subset.csv 中作为参照。

## 主要结果

- 实验后平均总成本：E2E=884.090，对照=1985.148，公开子样本降幅=55.46%。独立样本双侧 t 检验 p=1.07e-30。
- 与作者 Field_experiments_summary.ipynb 历史保存输出比较：45/55 项匹配；其余为已确认的 DiD t/p 缓存差异，详见 notebook_output_verification.csv。直接执行原 notebook 当前源码与公开 CSV 得到与新脚本一致的结果，独立核验另存 source_equivalence.json；差异原因公开资料无法确定。
- 公开子样本 turnover 的 E2E 无截距 OLS 系数为 0.345794（p=0.7698），并未重现论文完整样本中该回归系数显著为负的结论。
- 候选对照 55,333 行，按作者代码为 500 个处理样本进行可重复使用的最近邻匹配，选中 424 个不同候选。图表展示匹配前后 propensity score 分布。

## 精确遵循作者 notebook 的口径

成本使用 h=1、b=9，只累计 test_inv[:-1]。缺货比例分子为上述区间库存小于零的点数，分母仍为完整库存路径 T。各条路径长度不统一，未自行改为日均成本。turnover 为 ave_inv/ave_demand，并在汇总和实验后回归中截断为非负。

回归包含 e2e、vlt、d 三列，无截距，采用普通标准误。DID 的点估计为处理组前后变化减去对照组前后变化，显著性则忠实重现原 notebook 对前后两个逐行组间差序列进行独立样本 t 检验的方式；未声称这是带个体固定效应或聚类标准误的面板 DID。

原 notebook DID 唯独对照组实验后 turnover 使用未截断值（post_exp_jd 存入 jd_turnover）。table4_public_subset_notebook_convention.csv 保留该行为，table4_public_subset_consistent_turnover.csv 则给出统一非负截断的敏感性结果。其他四项不变。

## 论文口径敏感性（独立附表）

论文现场成本单价 h=12、b=88，与 notebook 的 h=1、b=9 不同；这不是同一倍数缩放。sensitivity_group_summary_h12_b88.csv 与 sensitivity_post_comparison_h12_b88.csv 使用同一公开子样本及库存区间定义替换成本单价，供比较。论文回归公式含截距，而 notebook 未加截距；sensitivity_ols_with_intercept_h1_b9.csv 为 notebook 成本单价下加截距的对照回归。它们均为本次额外计算，未替代主结果，也未当作作者保存日志。

## 匹配演示限制

作者先用 LogisticRegression 估算 propensity score，但实际 NearestNeighbors 输入是未标准化的 ave_demand、vlt_num、e2e 三列，而不是 propensity score。e2e 指示列对所有候选距离贡献相同常数，不改变排名。未加入 caliper、未禁止同一候选重复使用。由于匿名索引不能可靠跨文件映射，重新匹配结果未被冒充为已发布 order_jd_post_500.csv 的身份重建；候选数据也没有完整库存路径，无法重新生成这些候选的实证轨迹。matching_propensity.png 采用直方图展示分布，不是原 seaborn KDE 图的像素级复刻。

## 运行

从项目根目录运行 `.venv/Scripts/python.exe scripts/reproduce_field.py`。所有作者文件保持原样；结果写入 results/field。verification.json 记录依赖版本、输入哈希、定义和核验状态。
