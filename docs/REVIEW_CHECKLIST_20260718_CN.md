# FLUED 审计清单（2026-07-18 用）

> 目的：消除信息差。按"先控制面、后实现面、再数据面"排序。
> 每项标注：是什么 / 审什么 / 我已知的弱点（自查声明）。
> 生成人：K3（AI 协作者）。本清单本身也应被审计。

---

## 0. Git 状态（先明确物理边界）

- 分支 `copilot/implement-stage-a-experiments`，本地领先 origin **13 个 commit，全部未 push**；
- 今天（2026-07-17）产生的 11 个 commit 是本审计的主体（`5badff1`..`f9af2f4`）；
- 审计期间如有任何修改建议，先记录后讨论，**不要急着 push**。

---

## 1. 控制面（最高优先级，决定"当前什么是真的"）

### 1.1 `docs/CURRENT_STATE.md`
- 是什么：单一事实源。默认值表、证据注册表 E1-E15、闸门注册表、待裁定队列 Q1-Q7、维护规则、changelog。
- 审什么：
  1. 证据表里每条 claim 的状态你是否认可（尤其 E7 memory"该条件下已验证"、E10"类级信号"、E11"准入线已证伪"的措辞）；
  2. 闸门表中两个"未过-归因待裁定"的表述是否公允（Q1 就在等你裁定）；
  3. 维护规则（新证据必须原地更新+changelog）是否可执行、是否会变成新的负担。
- 已知弱点：这是我今天创建的；它声明自己权威，你需要确认愿意把权威交给这个文件。

### 1.2 `configs/canonical_v35.json`
- 是什么：唯一推荐训练基线（独立 decoder/uniform/plain/无 emit/无 memory），train 脚本直接可消费。
- 审什么：逐字段对——特别是 `decoder_mode=legacy_independent`、`boundary_mode=uniform_budget`、
  `use_emit_controller=false`、`use_memory=false`、`max_eval_batches=32`、`data_path` 占位符（正式运行必须显式传）。
- 已知弱点：`usage_note` 里"实验分支"的边界是我划的（emit/memory/CBIU 不进基线），边界划分需你确认。

### 1.3 `tests/test_canonical_sync.py`
- 是什么：同步测试，CLI 默认值 == canonical == CURRENT_STATE 三处漂移即红灯。
- 审什么：CURATED_KEYS 20 个字段是否覆盖了你在意的项；marketers 检查是否太弱（目前是字符串存在性检查）。
- 已知弱点：第三项测试是"字符串存在性"级别的弱校验，防呆不防坏。

### 1.4 `README.md`、`AGENTS.md` 顶部指针
- 是什么：两个入口都指向 CURRENT_STATE.md。
- 审什么：同意这个指向即可。

---

## 2. 我今天写的设计与结果文档（重点审，里面有我的判断和 claim）

### 2.1 `docs/versions/v3.5/FLUED_V3_5_STAGED_FREEZE_AND_OFFLINE_UTILITY_CN.md`
- 是什么：v3.5 设计草案（分级冻结 L0-L5、离线 CBIU 数据集规范、统一预算分配器、
  L3.5 闭环闸门（你提的）、证据两级制（你提的）、memory 举证判据与失败记录、
  order 探针证据（§3.1）、8.1/8.2 原创性叙事分界、禁止事项 15-22）。
- 审什么：
  1. §3 架构裁决表（共享逆"判死"、boundary×emit"合并"、memory"隔离"）是否同意；
  2. §5.4 L3 验收阈值与 §5.4b 闭环闸门设计是否如你所想；
  3. §8 原创性叙事（8.1 vs 8.2）是否认可——这直接关系投稿定位；
  4. §11 禁止事项 15-22 是否遗漏你想禁的。
- 已知弱点：L3.5 的 rollout 协议只有原则没有实现规格（boundary merge-rebuild 仍未实现）。

### 2.2 `docs/versions/v3.5/FLUED_V3_5_PRESCALING_RESULTS_20260717_CN.md`
- 是什么：L0-L3 全部结果 + 归因链 + L3 勘误（27 参数因子模型）。
- 审什么：
  1. §1 闸门核对：order 0.70 与退火 -34pp 两处"未过"的归因解释你是否接受（这是 Q1）；
  2. §4 L2/L3 数据：类均值 0.439 vs MLP 0.095、74.4% 类间方差、可加因子模型追平全查找；
  3. §6 剩余问题优先级排序。
- 已知弱点：L1/L2/L3 全部单 seed（42）；ECE 是 z 分数伪概率口径；L2 只测 affected chunk。

### 2.3 `docs/versions/v3.4/FLUED_V3_4_FOLLOWUP_EXPERIMENTS_20260717_CN.md`
- 是什么：decoder 解耦 5 臂、CBIU 多种子 3×3、emit 容量课程 3 臂。
- 审什么：实验 A 三臂廉价解耦的设计是否公平（预热 2K/交替 500/缩放 0.3 的参数选择）；
  实验 B 的 legacy 对照是否够强（legacy 也给了 MLP-64）。

### 2.4 `FLUED_HANDOFF_20260717_CN.md`（我修订过的版本）
- 是什么：handoff + 我插入的 14 处"2026-07-17 复核修订"。
- 审什么：重点看我改的 4.1（plain lookup 默认）、4.8（decoder blocker）、5.3（memory 0.05）、
  7.0（CBIU V0 补入）、15.1（P0 改为 decoder+退火+CBIU 顺序）——这些是改写你原文的地方。

---

## 3. 我今天改的代码（实现细节，最需要逐行审）

### 3.1 `tools/train/v3_4/train_v34_pos_ar_probe.py`（改动最多）
- 改动点（建议 git diff 逐段看）：
  1. `decoder_warmup_steps/alternating_period/loss_scale` 三个解耦旋钮（loss 组装前的阶段逻辑，~line 1107）；
  2. `emit_warmup_steps` + `emit_budget_anneal_start/end/target` 退火（训练循环内每步设置 model.emit_budget_override）；
  3. `training_scope=backbone_only`（冻结 model 只训 backbone）；
  4. `--reset-backbone`（init checkpoint 时保留 backbone 新初始化）；
  5. **13 处 argparse 默认值改到 canonical**（影响所有不带 config 的运行）；
  6. evaluate() 的 warmup/budget 保存-恢复。
- 审什么：阶段逻辑的优先级（warmup 先于 alternating）、backbone_only 的 loss 分支、
  默认值改动是否影响了你在意的旧测试/旧脚本。

### 3.2 `flued/v34/rate_emit.py`
- 改动点：`ReadoutEmitController.forward` 新增 `budget_fraction` 分位数 top-k 路径。
- 审什么：top-k 按 soft 分数排序的语义（fallback 恒开、每 chunk 均匀 k——不随内容分配，这是已知简化）；soft[...,0] 处理的小重构。

### 3.3 `flued/v34/model.py`
- 改动点：`emit_warmup_active` 旁路；`emit_budget_override` 传入 controller；`retain_grad` 守卫（backbone_only 冻结时）。
- 审什么：旁路语义（warmup=全开，与 use_emit_controller=false 等效）。

### 3.4 新分析脚本（v3.5 三个，都是一次性但结论依赖它们）
- `tools/analysis/v3_5/build_l2_offline_utility_dataset.py`（30,304 条数据集的生产者）；
- `tools/analysis/v3_5/train_l3_budget_allocator.py`（0.439/0.095 的出处）；
- `tools/analysis/v3_5/probe_l0_latent_readability.py`（98.5% 的出处）。
- 审什么：L2 的 global_step 编排是否造成 slot/位置采样偏差；L3 的 train/test 切分（batch<24）；
  readability 探针的 4 类标签定义（阈值 0.15/0.30/0.30/0.20 是我拍的）。

### 3.5 `tools/analysis/v3_4/probe_v34_cbiu_action_calibration.py`
- 改动点：仅新增 `--eval-mask-seed`（多种子校准用）。

---

## 4. 我今天新建的实验配置（每个的 base 继承链要审）

| 文件 | 用途 | 关键覆盖 |
|---|---|---|
| `configs/v3_4/v34_decoder_decoupling_20k.json` | 实验 A 5 臂 | no-memory/plain/diag/20K |
| `configs/v3_4/v34_cbiu_multiseed_emit_only_3k.json` | 实验 B 6 臂 | round3 链继承，seed 42/123/999 |
| `configs/v3_4/v34_emit_capacity_curriculum_10k.json` | 实验 C 3 臂 | 阶跃 warmup 证伪 |
| `configs/v3_4/v34_memory_order_variants_20k.json` | memory 4 臂 | w0.05 + 位置三变体 |
| `configs/v3_5/v35_l0_codec_20k.json` | L0 两 seed | 独立 decoder/uniform/无 emit 无 memory |
| `configs/v3_5/v35_emit_anneal_20k.json` | e1/e2 | 阶跃 vs 分位数退火 |

- 审什么：base_matrix 继承链最终解析出的值（每臂归档里有 `resolved_input.json` 可对）。

---

## 5. 我今天产生的归档（L 盘原始数据，抽查）

| 目录 | 文件数 | 审什么 |
|---|---:|---|
| `L:\FLUED_archive\v34_decoder_decoupling_20k_20260717` | 63 | 5 臂 train_log/summary |
| `L:\FLUED_archive\v34_cbiu_multiseed_20260717` | 69 | 6 臂 + 18 个校准 JSON |
| `L:\FLUED_archive\v34_emit_capacity_curriculum_10k_20260717` | 48 | f0/f1/f2 轨迹 |
| `L:\FLUED_archive\v34_memory_order_variants_20k_20260717` | 56 | 4 臂 + 3 组干预 JSON |
| `L:\FLUED_archive\v35_l0_codec_20k_20260717` | 28 | L0 两 seed 全套 |
| `L:\FLUED_archive\v35_emit_anneal_20k_20260717` | 27 | e1/e2 全套 |
| `L:\FLUED_archive\v35_l1_backbone_3k_20260717` | 8 | L1 + 锚点 cbiu_v0.json |
| `L:\FLUED_archive\v35_l2_offline_utility_20260717` | 6 | 30,304 条 JSONL + L3 报告 |

- 每个归档含 SHA256 manifest（可用昨天的校验脚本复核）；
- 抽查建议：每归档抽 1 个 `train_log.jsonl` 尾部 + `summary.json` + `resolved_input.json` 即可，
  不必全看；重点是 L2 的 JSONL（随机抽 20 行看字段合理性）。

---

## 6. 历史文档（按需，不必重读）

- `FLUED_HANDOFF_20260717_CN.md`（你已读）；
- `docs/versions/v3.4/` 21 篇：若重读，按 `docs/versions/v3.4/README.md` 的顺序（我已更新）；
- `docs/research/` 谱系与证据审计（v3.2.1 的 0.1898 vs 0.1440 出处）；
- K 盘 `v34_cbiu_three_rounds_20260717`（118 文件，manifest 已验）。

---

## 7. 讨论议程建议（我预判的分歧点）

1. **Q1 裁定**：两处闸门（order 0.70、退火 -34pp）接受归因解释还是回炉重测？
2. **canonical 基线边界**：emit/memory/CBIU 全部排除在基线外，是否太保守？
3. **投稿叙事 8.1 vs 8.2**：memory 已搁置，是否接受"接口+协议+CBIU"的 8.1 定位？
4. **下一步执行序**：Q2（退火+因子分数）→ Q3（因子模型过闸）→ Q4 → Q6 → Q7（300M L0）；
5. **代码默认改动**（13 处 argparse）是否影响你手头任何在跑的旧流程；
6. 单 seed 层（L1/L2/L3）的 seed 补齐策略与算力预算。
