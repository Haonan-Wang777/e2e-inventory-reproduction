# 作者数据获取与目录结构

论文：Qi et al., *A Practical End-to-End Inventory Management Model with Deep Learning*, Management Science 69(2), 759–773。

原始来源：[论文及补充材料入口](https://doi.org/10.1287/mnsc.2022.4564)。请从出版方获取补充 ZIP；本仓库不重新分发作者原文件或论文全文。

1. 下载作者补充 ZIP（本次使用文件名 `mnsc.2022.4564(1).zip`）。
2. 解压后，找到 `replication_package_MS-FTBDA-20-01585`。
3. 将该目录中的内容复制到本仓库根目录下的 `author_package/`；不要再嵌套一层同名目录。
4. 用 Python 3.12 运行 `setup.ps1 -Python "你的python.exe绝对路径"`，然后运行 `run_reproduction.ps1`。

目录应为：

```text
README.md
scripts/
results/
paper/
author_package/
  df_test.csv
  control_demand_vlt.csv
  order_e2e_pre_500.csv
  order_e2e_post_500.csv
  order_jd_pre_500.csv
  order_jd_post_500.csv
  E2E_Section4.ipynb
  Field_experiments_summary.ipynb
  Propensity_score_matching.ipynb
  E2E-model-code/
    Data/rdc_sales_1320.csv
```

请保留其余作者文件，来源核验会按 `results/provenance.json` 的 SHA-256 清单检查。不同来源版本如校验不符，应先审阅差异，不要修改预期哈希来强行通过。

`scripts/provenance.py` 可通过环境变量 `E2E_AUTHOR_ZIP` 指定原始 ZIP 的绝对路径；不指定时，会利用保存的来源清单检查 `author_package/`。如需另外核验论文全文，可自行将论文保存为 `paper/source_paper.pdf`；该文件不影响数值评估，未提供时保留原报告中的论文哈希并标记本次未重查。

离线 E2E/GBM 的预测已经包含在作者 CSV 中，重新计算其成本不等于重新训练论文私有模型。`results/model_demo/` 是公开销量样例及显式假设条件下的独立训练演示，未超过历史均值基准。
