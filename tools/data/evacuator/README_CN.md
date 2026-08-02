# FLUED 语料撤离转换器

此工具将来源盘中的 `.parquet` 与 `.jsonl.gz` 流式转为统一的 `JSONL.zst`：

```json
{"text":"...","source":"fineweb_w","source_file":"fineweb/data/...parquet"}
```

设计约束：

- 仅提取可训练文本载荷：Parquet 的 `text/content`，或 JSONL 的 `text/content/conversations[].value`。
- 单个输入文件对应一个输出文件，先写 `.partial` 再原子改名；重启自动跳过已完成输出。
- `manifest.jsonl` 记录记录数、输入/输出字节数与输出 SHA-256。
- `zstd=3` 优先吞吐和容量；不会解压成巨型纯文本。

构建：

```powershell
cd E:\projects\FLUED\FLUED\tools\data\evacuator
cargo build --release
```

2026-07-27 撤离布局：

```text
W:\incoming  -> O:\FLUED_evacuation_20260727\fineweb_w
X:\incoming  -> N:\FLUED_evacuation_20260727\github_code_x
Y:\v11       -> O:\FLUED_evacuation_20260727\v11_y
Z:\v11       -> N:\FLUED_evacuation_20260727\v11_z
```

每条命令可中断、可重跑。正式迁移前先用一个小目录或单文件确认压缩倍率和记录格式。
