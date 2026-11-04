# 目录迁移

| 原位置 | 现位置 |
| --- | --- |
| `NKU_IR_HW3/app.py` | `app.py` |
| `NKU_IR_HW3/scripts/` | `scripts/` |
| `NKU_IR_HW3/requirements.txt` | `requirements.txt` |
| `NKU_IR_HW3/captions.json` | `data/captions.json` |
| `NKU_IR_HW3/index/` | `docs/archive/index/`，仅作历史快照 |
| 原设计 PDF | `docs/archive/design-report.pdf` |

重新构建的索引统一写入 `data/index/`。历史索引对应的图片未包含在仓库内，且路径文件含 Windows 分隔符，因此不会自动用作新目录的运行索引。课程来源与原始设计材料保留在归档中。
