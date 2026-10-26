# 发布说明

本仓库补齐 `scripts/`、`results/`、`paper/` 的目录结构，以修复 README 中的相对链接。根目录原先的两份审计笔记移至 `paper/`。

数值结果、训练权重、离线与现场计算脚本保持原样。发布准备仅更新数据获取说明、README、忽略规则、安装入口的 Python 寻找方式，以及来源核验工具的可选外部路径。来源记录中的本机文件路径已移除。

`results/provenance.json` 保存原研究运行的来源与源代码哈希；发布辅助文件有所调整时，它不用于声称全部发布文件与原运行版本逐字节相同。`results/source_equivalence.json` 的数值源码等价核验对象 `reproduce_offline.py`、`reproduce_field.py` 未改动。当前公开文件的哈希见 `publication_manifest.json`。

原论文、补充数据、依赖环境和下载缓存未纳入仓库。请通过 [DATA_ACCESS.md](DATA_ACCESS.md) 获取原始材料。
