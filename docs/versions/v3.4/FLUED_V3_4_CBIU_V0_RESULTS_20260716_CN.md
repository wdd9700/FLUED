# FLUED v3.4 CBIU V0 离线验证结果

> 日期：2026-07-16  
> 状态：V0 通过“可计算、可分离”门槛；尚未证明在线训练有效  
> 主结果：`L:\FLUED_archive\v34_cbiu_v0_20260716`

## 1. 本轮回答的问题

本轮没有修改模型权重，只对冻结的 v3.4 20K 检查点执行配对干预，验证：

1. clean 重建、strict masked completion、visible preservation 能否形成有效 rich/null 锚点；
2. CBIU 是否能区分有价值和无价值的 extra readout；
3. 当前 emit probability 是否已经等价于真实边际价值；
4. memory 的直接作用和经 emit 产生的中介作用是否混在一起；
5. small AR 是否确实解决了 one-shot 路径的顺序/局部还原缺口。

边界合并尚未进入本轮，因为严格干预必须取消边界后从 chunk builder 开始重算。只改 confidence
或复用旧 chunk 属于伪干预。

## 2. 实现与验证

新增：

- `tools/analysis/v3_4/probe_v34_cbiu.py`
- `tests/test_v34_cbiu.py`

测试：

```text
61 passed
```

探针统一报告三项 `bits / target byte`，并使用 all-readout rich anchor 和 fallback-only null
anchor 归一化。所有实验的三个 anchor gap 均为正，没有出现无效维度。

## 3. m2 20K 检查点，16 批次主结果

检查点：

```text
L:\FLUED_archive\v34_attribution_matrices_20260716\memory_usage_20k\
m2_gate_usage_supervision_w005\latest.pt
```

| 干预 | clean 重建 BPB | mask 补全 BPB | 可见保持 BPB | readout/byte | rho | 保留质量效用 |
|---|---:|---:|---:|---:|---:|---:|
| rich：全部 readout | 3.67 | 5.41 | 4.19 | 0.58 | 0.00 | - |
| null：fallback-only | 5.61 | 5.83 | 5.80 | 0.04 | 1.00 | - |
| 当前 policy | 3.95 | 5.58 | 4.63 | 0.17 | 0.40 | - |
| memory zero，总效应 | 3.94 | 5.59 | 4.58 | 0.20 | 0.43 | +0.03 |
| memory stale，总效应 | 3.96 | 5.51 | 4.56 | 0.19 | 0.23 | **-0.17** |
| memory skip，总效应 | 3.94 | 5.59 | 4.58 | 0.20 | 0.43 | +0.03 |
| memory zero，固定 emit | 3.96 | 5.59 | 4.64 | 0.17 | 0.43 | +0.03 |
| memory stale，固定 emit | 3.96 | 5.59 | 4.63 | 0.17 | 0.42 | **+0.01** |
| memory skip，固定 emit | 3.96 | 5.59 | 4.64 | 0.17 | 0.43 | +0.03 |
| small AR skip，总效应 | 5.15 | 5.92 | 5.48 | 0.23 | 1.22 | **+0.82** |
| small AR skip，固定 emit | 4.91 | 5.89 | 5.39 | 0.17 | 1.13 | **+0.73** |

`保留质量效用 = rho(干预) - rho(policy)`。正值表示被删除组件有益；负值表示干预后质量更好。

## 4. Memory 的关键新结论

### 4.1 旧探针与 CBIU 曾出现相反结果

同一检查点、16 批次的旧评估协议为：

| 模式 | completion PPL |
|---|---:|
| normal | 43.437 |
| zero | 43.678 |
| shuffle chunk | 43.507 |
| stale batch | 43.603 |

旧协议认为 stale memory 有害；第一版 CBIU 总效应却显示 stale memory 明显改善。

### 4.2 原因不是随机波动，而是 emit 中介

stale memory 同时把实际 readout/byte 从约 `0.17` 提高到 `0.19`。允许 emit 自由响应时，模型用
更多 backbone latent 补偿了错误 memory，形成 `-0.17` 的表面改善。

固定正常路径的 hard emit 图后：

- zero/skip 使风险恶化约 `+0.03`；
- stale 使风险恶化约 `+0.01`；
- 三者不再显示错误 memory 更好。

因此当前最可靠结论是：

> m2 的正确 memory 具有弱正向直接作用；memory 对 emit 容量分配的中介效应远大于其内容
> 直接效应。旧的 memory on/off 结论混合了这两条路径。

这仍不足以证明 memory 已形成高质量语义记忆。其直接效用很小，并且没有完成 byte-anchor
顺序、实体追踪和 fresh-backbone 迁移验证。

## 5. Small AR 的结论

small AR 是本轮最强的正向组件：

- 允许 emit 联动时，保留质量效用约 `+0.82`；
- 固定 emit 后，直接效用仍约 `+0.73`；
- 严格 no-memory 对照上，16 批次保留效用约 `+1.22`。

这说明 small AR 的收益不是主要来自打开更多 readout，也不是 memory 的替代效应；它确实修复了
one-shot interpreter 在局部顺序与近似逆还原上的缺口。此前“RoPE + small AR 联合保留”的架构
判断得到更直接的反事实支持。

但这不支持固定十步交替更新。当前证据只证明执行 small AR 有价值，不证明某种优化节奏。

## 6. Emit 槽位价值与校准

m2 16 批次：

| extra slot | 平均 emit probability | 保留质量效用 | 符号 |
|---:|---:|---:|---|
| 1 | 0.435 | -0.0005 | 基本无价值 |
| 4 | 0.450 | +0.0006 | 基本无价值 |
| 8 | 0.554 | **+0.0941** | 有价值 |
| 12 | 0.448 | -0.0002 | 基本无价值 |
| 15 | 0.575 | **+0.1231** | 有价值 |

slot-level Brier 为 `0.2145`，阈值 0.5 的符号准确率为 `80%`。但跨四个检查点的 4-batch pilot：

| 检查点 | Brier | 符号准确率 |
|---|---:|---:|
| m0，usage weight 0 | 0.257 | 20% |
| m1，usage weight 0.02 | 0.215 | 60% |
| m2，usage weight 0.05 | 0.225 | 60% |
| no-memory control | 0.225 | 60% |

因此不能说当前 emit controller 已经学会“价值”。它只在部分训练配置中粗略识别高价值槽位，
并且低价值槽仍经常被错误预测。CBIU 替换现有固定 rate/cost emit target 有明确必要性。

## 7. CBIU V0 是否成功

### 已通过

1. 三风险 rich/null anchor 在所有被测检查点上均有效分离；
2. CBIU 能把 small AR、关键 readout 和近零价值 readout 明确分开；
3. total/direct effect 成功定位 memory→emit 的中介混淆；
4. 结果可用同一输入、mask、参数和硬执行图重复；
5. 工具没有修改模型权重，适合继续审计历史检查点。

### 尚未通过

1. emit 校准目前只有槽位级样本，缺少 per-chunk AUC/Brier/ECE；
2. boundary merge 反事实尚未实现；
3. memory 和 small AR 的真实延迟/FLOPs 成本尚未接入；
4. rich/null anchor 目前由同一模型的全 readout/fallback 构造，尚未做跨 checkpoint 冻结；
5. 没有 fresh-backbone、无教师语义干预和多种 mask 分布验证；
6. 没有证明在线 CBIU 能避免 3K 容量坍缩和 6K 边界梯度冲击。

结论口径：

> CBIU 已证明是可计算且能产生新归因结果的离线尺度；尚未证明是稳定的在线训练目标。

## 8. 下一步决策

1. 先补 per-chunk emit CBIU 与校准曲线；
2. 实现严格 boundary merge，从 chunk builder 重算并记录成对成本；
3. profiler 校准 readout、memory 和 small AR 的执行成本；
4. V1 只让 CBIU 接管 emit，固定 boundary/memory/AR，训练 5K；
5. 只有 V1 同时改善 `rho` 与真实 latent/byte，才进入 boundary+emit 联合控制。

