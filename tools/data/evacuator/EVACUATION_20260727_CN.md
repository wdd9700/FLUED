# W/X/Y/Z 语料撤离执行记录

## 范围

仅迁移已识别的语料原料目录，不迁移系统文件、应用、HF 缓存、历史检查点或下载日志：

| 来源 | 输入 | 格式 | 目标 |
|---|---|---|---|
| W | `W:\incoming` | 265 Parquet，约 417.29 GiB | `O:\FLUED_evacuation_20260727\fineweb_w` |
| X | `X:\incoming` | 1044 Parquet，约 279.78 GiB | `N:\FLUED_evacuation_20260727\github_code_x` |
| Y | `Y:\v11` | 40 jsonl.gz，约 330.19 GiB | `O:\FLUED_evacuation_20260727\v11_y_v3` |
| Z | `Z:\v11` | 59 jsonl.gz，约 82.11 GiB | `O:\FLUED_evacuation_20260727\v11_z_v3` |

## 输出与校验

- 统一输出为 `JSONL.zst`：`text`、`source`、`source_file`。
- 单输入文件对应单输出文件；`.partial` 只有完成后才会重命名为正式文件。
- 每个数据集根目录生成 `manifest.jsonl`，含输出 SHA-256、记录数和输入输出字节数。
- 输出无损于选定的文本字段，但刻意不携带网页 URL、Parquet 元数据、标签和下载器缓存；原始来源盘在校验前不删除。
- 试转换验证：网页 `574,780,318 -> 339,915,070` 字节、代码 `278,540,871 -> 177,539,203` 字节；zstd 结构检验通过。
- Y/Z 使用多成员 gzip 解码；损坏 gzip 尾部只停止该文件的后续成员并保留已验证可读前缀，警告写入 lane 日志。无已知文本字段的 JSON 会保留为规范 JSON，而不是空分片。异常超长 JSON 行按 8 MiB 原始片段保存，避免无界内存分配。旧的 `v11_y` / `v11_z` 及 v2 单成员输出均不作为撤离完成物。
- Y/Z 默认单 worker，避免大 gzip 成员并发解压造成主机内存压力；W/X 的 Parquet 阶段已完成，不受此限制影响。

## 启动

```powershell
$root = 'E:\projects\FLUED\FLUED\tools\data\evacuator'
Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',"$root\run_evacuation_20260727.ps1",'-Lane','O' -WindowStyle Hidden
Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',"$root\run_evacuation_20260727.ps1",'-Lane','N' -WindowStyle Hidden
```

两条 lane 分别串行写 O/N，避免同一目标盘并发写入；两盘之间并行。任何中断后重新执行同一命令即可跳过已完成文件。
