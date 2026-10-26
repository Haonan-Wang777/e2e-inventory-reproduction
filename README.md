# 论文复现项目

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

当前电脑直接在此文件夹运行 `./run_reproduction.ps1`。只做公开数据重算可用 `./run_reproduction.ps1 -SkipDemo`。

其他电脑先用 Python 3.12 执行 `setup.ps1 -Python "python.exe的绝对路径"`。直接Python入口为 `.venv/Scripts/python.exe run_reproduction.py`。

`author_package/` 保留作者原文件。`scripts/` 是本次新增可运行实现。`results/` 是本次计算输出，历史notebook缓存差异单独披露。`tmp/`、`.venv/` 可重新建立，不属于研究结论。
