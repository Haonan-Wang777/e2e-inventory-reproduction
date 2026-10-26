"""Assemble a Chinese report from calculated result files, never cached outputs."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'


def table(frame, digits=3):
    def fmt(value, column):
        if isinstance(value, (float, int)):
            if 'p值' in str(column) and value != 0 and abs(value) < .001:
                return f'{value:.3g}'
            return f'{value:.{digits}f}'
        return str(value)
    rows = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
            '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    rows.extend('| ' + ' | '.join(fmt(v, c) for v, c in zip(row, frame.columns)) + ' |'
                for row in frame.itertuples(index=False, name=None))
    return '\n'.join(rows)


def main():
    offline = pd.read_csv(RESULTS / 'offline/cost_comparison.csv', index_col=0)
    offcheck = json.loads((RESULTS / 'offline/verification.json').read_text(encoding='utf-8'))
    field = pd.read_csv(RESULTS / 'field/table2_public_subset.csv')
    did = pd.read_csv(RESULTS / 'field/table4_public_subset_notebook_convention.csv')
    ols = pd.read_csv(RESULTS / 'field/table3_public_subset_all_coefficients.csv')
    cache = pd.read_csv(RESULTS / 'field/notebook_output_verification.csv')
    equivalence = json.loads((RESULTS / 'source_equivalence.json').read_text(encoding='utf-8'))
    provenance = json.loads((RESULTS / 'provenance.json').read_text(encoding='utf-8'))
    if not equivalence['passed'] or not offcheck['notebook_check_passed']:
        raise AssertionError('Cannot publish a report before source equivalence checks pass')
    ot = offline.T.reset_index(names='方法').rename(columns={'Total cost': '总成本', 'Holding cost': '持有成本', 'Stockout cost': '缺货成本'})
    ot['E2E相对该方法总成本下降(%)'] = [100*(1-offline.loc['Total cost','E2E_RNN']/v) for v in ot['总成本']]
    labels = dict(holding_cost='持有成本', stockout_cost='缺货成本', total_cost='总成本', turnover='周转指标(库存/需求)', stockout_ratio='缺货率')
    ft = field[['metric','treatment_mean','control_mean','relative_reduction_pct','p_value']].copy()
    ft.columns = ['指标','E2E组','对照组','相对下降(%)','双侧t检验p值']
    ft['指标'] = ft['指标'].map(labels)
    dt = did[['metric','treatment_pre','treatment_post','control_pre','control_post','did','p_value']].copy()
    dt.columns = ['指标','E2E前','E2E后','对照前','对照后','DiD','作者检验口径p值']
    dt['指标'] = dt['指标'].map(labels)
    failed = cache[~cache['match']]
    cache_note = (f'历史缓存 {int(cache["match"].sum())}/{len(cache)} 项匹配，{len(failed)} 项不匹配。'
                  '不匹配项来自 DiD 的 t 值和 p 值。独立执行原 notebook 的当前源码和公开 CSV，得到与本次脚本相同的结果；'
                  '公开资料不足以确认历史缓存差异的原因。缓存不作为改变算法或扩大容差的理由。')
    normal = pd.read_csv(RESULTS / 'offline/normal_formula_sensitivity.csv', index_col=0)
    e2e_ols_turnover = ols[(ols.metric=='turnover') & (ols.term=='e2e')].iloc[0]
    training = ''
    demo_file = RESULTS / 'model_demo/metrics.json'
    if demo_file.exists():
        demo = json.loads(demo_file.read_text(encoding='utf-8'))
        config = json.loads((RESULTS / 'model_demo/config.json').read_text(encoding='utf-8'))
        demo_cost = pd.DataFrame(demo['mean_hypothetical_interval_cost']).T.reset_index(names='方法')
        demo_cost = demo_cost.rename(columns={'holding':'持有成本','shortage':'缺货成本','total':'总成本'})
        training = f'''## 4. 可训练模型：机制演示，不是论文全量重训

根据图1和公开模块实现了需求 MQRNN、交货期 MLP、决策融合 MLP 和三项联合损失。保留默认 hidden=50、VLT层50/20、融合层100/100、dropout=0.2、SGD学习率0.001、momentum=0.85、batch=64、高斯初始化。保密的损失和特征由显式配置补全，不能据此恢复私有模型。

实际观察数据仅为5条公开销量序列、每条457天。交货期固定3天、到货间隔14天、初始库存由过去28天均值构造；这些条件都是演示假设。历史窗口60天、预测21天、分位数0.1/0.5/0.9；周特征为相对日序的正余弦。不同日期的场景独立生成库存，成本不是连续库存仿真的运营收益。

按时间60%/20%/20%划分，训练/验证/测试样本数分别为 {config['sample_counts']['train']}/{config['sample_counts']['validation']}/{config['sample_counts']['test']}。未来标签窗口不跨分区边界，缩放只拟合训练期。训练 {config['epochs']} 轮；依据验证损失选择第 {demo['best_validation_epoch']} 轮，不用测试集选模型。

- 定理1标签在 {demo['checks']['theorem_label_bruteforce']['cases']} 个随机小例上用凸成本折点穷举核验，最大目标差 {demo['checks']['theorem_label_bruteforce']['maximum_objective_gap']:.3g}。
- 梯度存在且有限，参数已更新，训练损失下降；保存/恢复模型预测最大差 {demo['checks']['checkpoint_max_prediction_difference']:.3g}。
- 测试最优订单标签 MAE={demo['test_order_MAE']:.3f}，RMSE={demo['test_order_RMSE']:.3f}。

以下仅为假设条件下的14天到货区间成本，不与论文图2比较：

{table(demo_cost)}

本演示模型未优于过去均值基准，因此不支持将这组小样本训练结果写成论文性能结论；它验证的是模型、标签和训练流程确实可运行。

![训练与验证曲线及演示成本](model_demo/learning_curve.png)

模型权重在 [best_validation.pt](model_demo/best_validation.pt)，配置与假设在 [config.json](model_demo/config.json)，全部训练记录在 [training_history.csv](model_demo/training_history.csv)。
'''
    else:
        training = '## 4. 模型训练\n\n本次仅汇总已完成的公开数据评估；没有训练结果时不声称训练完成。\n'
    report = f'''# 《A Practical End-to-End Inventory Management Model with Deep Learning》复现报告

**结论：已重算作者公开数据范围内的离线评估和现场统计，并独立核对当前作者源码；完整私有训练实验无法由公开包精确恢复。** 离线 E2E/GBM 的预测来自作者 CSV，本次没有将这些预生成预测冒充为重训所得。模型训练演示另列。

论文：Qi et al., Management Science 69(2), 759–773，DOI 10.1287/mnsc.2022.4564。来源为用户提供的论文和 ZIP，无外部替代数据。报告生成时间：{provenance['created_at']}（北京时间）。

## 1. 完成范围和验证

| 内容 | 本次完成 | 边界 |
| --- | --- | --- |
| 离线比较 | 重算7种方法、21个平均成本值；与缓存在六位小数舍入精度一致 | 公开CSV有3,000行、2,607个唯一SKU标识；不是完整24,333 SKU实验 |
| 现场实验 | 四组各500行的均值、t检验、OLS与DiD统计；匹配诊断 | 是公开子样本重算，未重复真实京东现场干预 |
| 模型实现 | 可训练MQRNN+VLT+融合模型、定理1标签、联合损失、权重保存 | 缺失特征、库存、交货期和损失细节有明确假设；仅机制演示 |
| 审计 | 作者文件逐字节校验、依赖版本、源码与数据哈希、独立源码等价检验 | 原论文表格仅作参照，不写入“本次计算”列 |

独立验证通过：7种方法×2,607组完整库存曲线及计费曲线、21个离线均值；现场12,000个逐行指标、10个实验后t/p、60个OLS系数/标准误/t/p、15个DiD点估计/t/p。详见 [source_equivalence.json](source_equivalence.json)。保留了 {provenance['preserved_author_file_count']} 个原始作者文件，校验记录在 [provenance.json](provenance.json)。

## 2. 离线结果：公开样本

按作者 notebook 的库存模拟与计费规则计算，h=1、b=9；成本是每个SKU组的累计成本均值，不是每天成本。表中相对下降按 `1-E2E成本/比较方法成本` 计算；OPT为事后参照。

{table(ot, 2)}

![公开数据离线成本](offline/offline_costs.png)

21项成本与作者保存的六位小数结果的最大绝对差为 {offcheck['max_absolute_error_to_six_decimal_notebook']:.3g}。E2E相对GBM总成本下降 {100*offcheck['e2e_total_cost_reduction']['gbm']:.2f}%。论文全量 E2E/OPT 成本分别为3766.52/3022.60，公开样本为3036.76/2572.96；数值不同是比较范围不同，不能称作完整数值精确复现。

必须保留的代码口径：初次到货前的库存不计入这里的成本；补货可能为负；到货时负库存被重置；SKU内部沿CSV顺序模拟，文件不含时间戳，无法另行验证排序。`vlt`字段在README与原日志中的语义不一致。以上均在忠实重算中保持，不直接等同于理论完全积压库存模型。

Normal基准的公开代码使用 `Z90=1.64`，其方差项也不同于论文式(7)。单独按式(7)替换正态公式、其余仿真不变时，Normal平均总成本从 {normal.loc['Total cost','Normal']:.2f} 变为 {normal.loc['Total cost','Normal_equation7']:.2f}。这只是公式敏感性，主表仍按作者代码。详见 [normal_formula_sensitivity.csv](offline/normal_formula_sensitivity.csv)。

## 3. 现场统计：公开500行子样本

下表按作者 notebook 使用 h=1、b=9；正文现场单价为h=12、b=88。二者并非相同比例缩放，已另存论文单价敏感性附表。库存路径长度不统一，成本计 `test_inv[:-1]`，缺货率分母仍用完整路径长度；周转为平均库存/平均需求，越小越好。

{table(ft, 6)}

![现场实验后成本](field/post_experiment_costs.png)

子样本也并非所有回归结论都重现：无截距OLS中，周转指标E2E系数为 {e2e_ols_turnover.coefficient:.4f}、p={e2e_ols_turnover.p_value:.4f}；论文完整样本系数为-1.16、p=.007，不能声称这里获得同样的显著改善。OLS和t检验按作者定义重算，不单独赋予新因果解释。

### DiD点估计及缓存审计

{table(dt, 6)}

{cache_note}

这里的显著性计算忠实原 notebook：按文件行顺序求处理减对照，再对前后两个差值序列做独立样本t检验；不是带个体固定效应、聚类标准误的面板DiD估计。原完整样本日志还存在三项前期变量复制错误，已在审计笔记记录；本次没有沿用这些错误来追求论文p值。

匹配程序先拟合logistic propensity score，但作者实际在未标准化的需求、交货期及组指示列上做有放回最近邻，而非按propensity score做匹配。保留该实现并输出 [匹配诊断图](field/matching_propensity.png)。候选匿名索引无法可靠对应现成控制组轨迹，因此匹配演示不冒充恢复控制组身份。

补充文件：[按论文单价的现场成本](field/sensitivity_post_comparison_h12_b88.csv)、[带截距OLS](field/sensitivity_ols_with_intercept_h1_b9.csv)、[缓存差异逐项记录](field/notebook_output_verification.csv)、[现场详细报告](field/field_report.md)。

{training}
## 5. 怎样运行和继续研究

当前文件夹已准备好本地Python环境。在PowerShell进入项目文件夹运行：

```powershell
./run_reproduction.ps1
```

只重算公开评估、不运行训练演示：

```powershell
./run_reproduction.ps1 -SkipDemo
```

移到其他电脑后，使用Python 3.12运行 `setup.ps1 -Python "你的python.exe绝对路径"` 安装固定版本依赖，再运行以上命令。已安装环境记录在provenance.json；入口将各阶段输出保存至results/logs，任何源码等价核验失败都会停止。

主要代码：`scripts/reproduce_offline.py`（离线）、`scripts/reproduce_field.py`（现场）、`scripts/train_demo.py`（训练演示）、`scripts/verify_source_equivalence.py`（原源码独立对照）。作者包原文件位于author_package，论文副本在paper/source_paper.pdf。

要做原论文全量训练与附录敏感性实验，仍需：完整132维训练/验证/测试特征及时间索引、匹配一致的销量/库存/订单/VLT表、保密预处理与标签规则、完整损失及辅助权重、分位设置和原随机种子/权重。可直接用这些真实数据替换演示的数据构造层，再按时间切分训练；不能将模拟条件所得曲线替换为论文复现结论。

研究细节见 [实验规格与原文参照](../paper/experiment_spec.md) 和 [模型代码审计](../paper/model_audit.md)。
'''
    (RESULTS / 'reproduction_report.md').write_text(report, encoding='utf-8')
    readme = '''# 论文复现项目

已重算公开数据的离线评估、现场统计，并补写可运行的模型训练演示。完整京东训练数据和部分模型细节未公开，演示不等同于论文全量重训。

- [复现报告：结果、限制和运行方法](results/reproduction_report.md)
- [离线成本表](results/offline/cost_comparison.csv)
- [现场实验表2](results/field/table2_public_subset.csv)
- [现场实验表3](results/field/table3_public_subset_all_coefficients.csv)
- [现场实验表4](results/field/table4_public_subset_notebook_convention.csv)
- [独立源码等价核验](results/source_equivalence.json)
- [论文实验规格](paper/experiment_spec.md)
- [模型审计](paper/model_audit.md)

## 运行

重新计算前请先按 [数据获取说明](DATA_ACCESS.md) 准备作者原始数据。本公开仓库不附带作者原文件或论文 PDF。

当前电脑直接在此文件夹运行 `./run_reproduction.ps1`。只做公开数据重算可用 `./run_reproduction.ps1 -SkipDemo`。

其他电脑先用 Python 3.12 执行 `setup.ps1 -Python "python.exe的绝对路径"`。直接Python入口为 `.venv/Scripts/python.exe run_reproduction.py`。

`author_package/` 保留作者原文件。`scripts/` 是本次新增可运行实现。`results/` 是本次计算输出，历史notebook缓存差异单独披露。`tmp/`、`.venv/` 可重新建立，不属于研究结论。
'''
    (ROOT / 'README.md').write_text(readme, encoding='utf-8')
    print('Created results/reproduction_report.md and README.md')


if __name__ == '__main__':
    main()
