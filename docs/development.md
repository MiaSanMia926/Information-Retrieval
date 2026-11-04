# 开发与数据约定

## 目录规则

可维护代码放在根入口与 `scripts/`，用户数据放在 `data/`，设计与使用文档放在 `docs/`。不为当前体量引入空的控制器、服务或部署目录。

`data/images/` 与 `data/index/` 仅保留目录占位文件，实际图片和生成索引由 `.gitignore` 排除。`data/captions.json` 是可维护的文本数据，继续纳入版本控制。

## 添加图片

1. 放入图片并使用有含义的文件名；页面 BM25 会使用文件名作为文本语料。
2. 可在 `data/captions.json` 中添加 `"文件名.jpg": "英文描述"`。
3. 重新运行 `python scripts/build_index.py`，不要手动打乱 `paths.txt` 的顺序。

模型必须与索引构建时一致。向量序号与路径行号一一对应；仅复制 `vector.index` 而不复制匹配的 `paths.txt` 会导致错误结果。

## 检查改动

```bash
python -m compileall -q app.py scripts
python scripts/build_index.py
python -m streamlit run app.py
```

手动验证文本查询、图片查询、三项开关和相关性反馈。构建与页面检查需要真实图片、模型及依赖，语法检查不等于功能验证。
