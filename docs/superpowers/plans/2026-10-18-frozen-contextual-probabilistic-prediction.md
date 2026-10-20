# 冻结表示上的上下文条件概率预测实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在冻结第三章 MTK-DG encoder 的前提下，加入一个不依赖具体概率输出头的上下文条件残差 adapter，并用按有效样本量向全局收缩的 validation-only 校准把预测分布连接到第五章风险决策。

**Architecture:** 从第三章 checkpoint 只加载并冻结 encoder，保留其多尺度表示、动态图诊断量、观测比例和 FD 工况 embedding。新增 `ContextualResidualAdapter` 在 `[B,C,d]` 表示上产生受支持度收缩的残差表示；概率 head 只是可替换的输出层。另加一个输出头无关的 reliability calibrator，在 validation 上根据推理时可得的上下文修正尺度/分位数并向全局关系收缩。训练使用完整 FD RUL 标签，MetroPT 保持多步多通道目标；三数据集各自训练和报告，不合并估计量。

**Tech Stack:** Python 3.9, PyTorch 2.5.1, NumPy, pytest, 现有 `mtkdg` bundle/checkpoint/artifact 约定。

**Design status:** 实现计划；尚未授权 GPU 矩阵。它是回到 `7aff882` 后形成的新候选方案；删失概率方案和历史结果继续保留为独立历史，不修改、不覆盖。

---

## 研究问题与贡献边界

第四章要回答：第三章冻结表示中的工况、预测难度和传感器可观测性信息，能否支持**上下文条件的表示修正和概率可靠性校准**？局部上下文样本不足时，向全局校准收缩是否比硬分组稳定？FD002、FD004、MetroPT-3 分开训练、分开拟合校准器和分开报告；不合并数据、参数或有效样本量。

不把以下内容称为创新：LogNormal、Gaussian、分位数回归、temperature scaling、isotonic regression、普通 split conformal，或把一个 MLP 直接接到冻结 `z`。这些只作为可替换 head、基线或校准器。主方法贡献是 `z` 之后的 context residual adapter 加支持度收缩策略；第五章负责将通过质量门的输出转成风险与代价。

### 固定计算定义

- Encoder 输出保持现有合同：`z [B,C,d]`、`z_self`、`z_cross`、`gate_weights [B,S]`、`fusion_alpha [B,C,d]`、`obs_ratio [B,C]`、FD 的 `cond_emb [B,d]`。
- Context adapter 的具体张量运算固定为共享通道函数：`h_c = LayerNorm([z_c; mean_c(z); max_c(z); r(x); e_c])`，`delta_c = W_delta GELU(W_h h_c)`，`a_c = sigmoid(W_gate h_c)`，`z_ctx,c = z_c + w_train(g_c) * a_c * delta_c`；`delta` 最后一层零初始化。FD 的 `e_c` 为 encoder channel embedding，MetroPT 同左；FD `cond_emb` 拼入 `r(x)`。禁止改变 encoder 输出维度或修改 encoder 内部。
- 表示统计 `r(x)` 只使用推理时可得量：`mean(z)`、`max(z)`、`std(z)`、`gate_weights`、`obs_ratio`、`mean(abs(fusion_alpha))`、FD `cond_emb`；不得读取目标、残差或 test 标签。离散上下文的拟合规则和特征标准化参数仅由 train 确定。
- difficulty score 定义为 train 拟合的 ridge probe 对绝对点误差的交叉拟合预测值；probe 特征仅为上述 `r(x)`，交叉拟合按独立 cluster 分折（FD 按 engine，MetroPT 按自然周）。以 train OOF score 的固定四分位点形成 4 档，之后 validation/test 只应用冻结边界。点误差只用于拟合训练阶段 probe，不输入概率模型或校准器。
- FD condition vocabulary 只由 train engine 的 rounded settings 拟合并保存；validation/test 的未见组合映射到 `condition_code=UNK`，其支持权重为 0。不得在全 train+val+test 上重新 factorize condition code。
- 独立 cluster 固定为 FD engine 与 MetroPT 自然周。对某上下文组 `g`，支持度 `n_eff(g)` 是该组包含至少一个合法样本的独立 cluster 数，不是窗口数；每个 cluster 在支持度统计中至多计 1。adapter 支持权重用 `w_train(g)=n_eff,train(g)/(n_eff,train(g)+kappa_train)`，其中 `kappa_train` 是 train 非空组支持度的中位数，截断到至少 1。validation 校准另行计算 `n_eff,val(g)` 和 `kappa_val`，禁止复用 train 计数或原始窗口数。
- 支持度键：FD 为 `(condition_code, difficulty_bin)`；MetroPT 为 `(channel_id, horizon_bucket, difficulty_bin)`。MetroPT horizon bucket 固定为 1–8、9–16、17–24（本项目 pred_len=24）；校准按输出通道和 horizon bucket 报告，风险阈值不是校准上下文。
- Adapter 计算 `z_ctx = z + w_train(g) * sigmoid(gate(h)) * delta(h)`，其中 `h` 由 `z`、可观测上下文和全局统计拼接得到；残差为零时严格退化为同一冻结 encoder 的 base readout。为防止 context 直接编码数据集身份，FD 与 MetroPT 各自训练独立 adapter，不共享参数。
- 主实验固定共同 quantile 接口：FD 输出 `[B,Q]`，MetroPT 输出 `[B,P,C,Q]`，`Q={0.05,0.10,...,0.90,0.95}`；每个数据集独立报告 pinball、由相邻分位数积分的 CRPS 近似、中央 50/80/90% coverage 与 width。输出头可替换，但替换属于独立预注册变更，不能用 test 选型。非负 RUL 通过正值变换约束，MetroPT 保持标准化预测后逆变换 raw unit。
- Validation 校准固定为分位残差加法校准：对每个预测分位 `tau`，在 validation 校准样本上估计 `c_global,tau = Quantile_tau(y-q_tau)`；局部组同理得 `c_local,g,tau`。预测校正量为 `c_g,tau = (1-w_val(g))*c_global,tau + w_val(g)*c_local,g,tau`，`w_val(g)=n_eff,val(g)/(n_eff,val(g)+kappa_val)`。校正后沿分位轴做单调投影；不得按 validation 校正前后分数选择方法、分组、`kappa` 或阈值。
- 为匹配测试推理分布，FD validation 校准只用每个 validation engine 的末端窗口；MetroPT 校准 origin 按时间每 `pred_len` 步抽取、目标区间不重叠，目标所属自然周作为 cluster ID。其余 validation 窗口只用于冻结 encoder 的独立训练诊断，不进入 calibrator。若某 split 可用独立 cluster 少于 5，报告该组 calibration unsupported，退回全局校准，不声称局部收益。

### 实现合同与研究边界

`FrozenEncoderBundle` 暴露 `forward(batch) -> EncoderOutput`、`fingerprint() -> str` 和 `train(mode=True)`（始终保持 encoder `eval()`）。`ContextMetadata` 暴露 `fit_train(features, cluster_id, context_key)`、`assign(features, context_key)` 和 `state_dict()`；`ContextualResidualAdapter.forward(encoder_output, metadata) -> AdapterOutput` 只返回 `z_ctx`、`delta`、`gate` 与支持权重。`QuantileReadout.forward(z_ctx, cond_emb, target_schema) -> ProbabilisticOutput` 不访问目标。`ShrunkQuantileCalibrator.fit(val_calibration, cluster_id, context_key)` 只接受 validation calibration view，`transform(output)` 不改变样本数量。

第三章的冻结 checkpoint 仍使用现有 capped-label 点预测协议；第四章不重训或修改第三章 encoder。第四章 FD 的 raw-label `Cpoint` 只作为同一冻结 encoder 上的公平标签协议对照，回答“完整标签本身带来多少变化”；概率主方法的增益必须相对于同 raw-label C0/C1 报告。若 raw-label Cpoint 改善而 C1/C2 不改善，结论只能归因于标签协议，不能宣称 adapter 或校准有效。

三套数据不构造一个共享数据集模型：FD002、FD004、MetroPT-3 各自拥有 scaler、condition vocabulary、difficulty probe、adapter、readout 和 calibrator；共享的是接口、预注册规则和统计指标。论文中把“跨数据集”表述为同一研究原则在三种任务上的复现，而不是参数共享或统一分布假设。

## 文件地图

- 修改 `mtkdg/data/cmapss.py`：支持显式 `training_target=raw|capped` 的完整 RUL train/val 标签和 target scaler 元数据，同时保持 capped 旧协议完全兼容；不要以 `None` 隐式表示 raw。
- 修改 `mtkdg/data/bundle.py`、`mtkdg/config.py`：增加概率实验的 `training_target=raw|capped` 记录，旧 point CLI 默认不变。
- 修改 `mtkdg/data/common.py`、`mtkdg/data/metropt.py`：为 MetroPT 窗口保留 origin 时间、完整目标区间和所属自然周；新增 validation calibration view，按 `pred_len` 取不重叠目标区间。
- 新建 `mtkdg/models/frozen_encoder.py`：从 audited point checkpoint 构建同构 encoder，只加载 `encoder.*` 权重并提供 bitwise fingerprint。
- 新建 `mtkdg/models/context_adapter.py`：实现无 head 依赖的 `ContextFeatureBuilder`、支持度表和 `ContextualResidualAdapter`。
- 修改 `mtkdg/models/prob_heads.py`：统一 `ProbabilisticOutput` 合同；保留现有 head，修正数值稳定性并增加 adapter 后的 forward 路径。
- 新建 `mtkdg/evaluation/calibration.py`：上述分位残差全局/局部/收缩校准、独立 cluster 支持数、可靠性指标与输出头适配。
- 新建 `mtkdg/training/probabilistic_runner.py`：固定 epoch 的 train-only adapter/readout 训练、validation-only calibration、一次性 test。
- 新建 `scripts/audit_frozen_checkpoints.py`：checkpoint、split、scaler、encoder state key 和 seed 审计。
- 新建 `scripts/run_contextual_probability.py`、`scripts/run_contextual_probability_matrix.sh`：显式协议参数和唯一 run 目录。
- 修改 `mtkdg/evaluation/artifacts.py` 或新建 `mtkdg/evaluation/probability_artifacts.py`：写出 encoder/adapter/calibrator provenance 和预测数组。
- 新建测试：`tests/test_raw_label_protocol.py`、`tests/test_frozen_encoder.py`、`tests/test_context_adapter.py`、`tests/test_calibration.py`、`tests/test_probabilistic_runner.py`、`tests/test_probability_artifacts.py`、`tests/test_checkpoint_audit.py`。
- 新建预注册文件 `refine-logs/CH4_CONTEXTUAL_PROBABILITY_PREREGISTRATION.md` 和结果文件 `refine-logs/CH4_CONTEXTUAL_PROBABILITY_RUN_LOG.md`。

## 核心执行伪代码

```python
# train-only metadata; cluster_id is engine or natural-week, never a window count
meta = ContextMetadata.fit_train(
    features=train_repr,
    point_error=train_point_error_raw,
    cluster_id=train_cluster_id,
    condition_code=train_condition_code,
    output_context=train_output_context,
    ridge_alpha=1.0,
    n_folds=5,
)

# train step; encoder is evaluated under no_grad and is absent from optimizer
with torch.no_grad():
    enc = frozen_encoder(batch.x, batch.t, batch.mask, batch.condition)
adapter_out = adapter(enc, meta.assign(batch_repr, batch_context))
prob = readout(adapter_out.z_ctx, enc.cond_emb, target_schema)
loss = pinball_loss(prob.quantiles, batch.y_raw_or_standardized, TAUS)
loss.backward()
optimizer.step()  # only adapter.parameters() + readout.parameters()

# validation-only calibration view
val_out = predict(model, val_calibration_view)
cal = ShrunkQuantileCalibrator.fit(
    val_out, val_targets, val_context, val_cluster_id,
    split="val_calibration", min_local_clusters=5,
)
val_calibrated = cal.transform(val_out)

# exactly once after all validation artifacts are closed
test_out = predict(model, test_view)
write_artifacts(..., evaluated_once=True, test_out=test_out)
```

每个阶段都先写失败测试，再实现最小接口、运行该阶段测试、提交一次。代码实现不能通过读取 test target、改变第三章 checkpoint、改变已注册 bin/kappa/head 或覆盖旧 run 来“修复”失败；失败结果进入 run log 和结果表。

---

## Task 1: 冻结完整 FD 标签协议

**Files:** `mtkdg/data/cmapss.py`, `mtkdg/data/bundle.py`, `mtkdg/config.py`, `tests/test_raw_label_protocol.py`

- [ ] 新增 `_label_for_training(raw_rul, training_target, training_cap)`：`training_target=raw` 返回完整 raw RUL；`training_target=capped` 要求正整数 `training_cap` 并返回 `min(raw_rul, cap)`；test 始终使用官方 raw RUL。
- [ ] 让 target scaler 在 raw 模式仅由 train-engine 的完整标签拟合；在 capped 模式保持旧行为；`target_schema` 明确写出 `training_target`、`training_cap` 和 test label source。
- [ ] 新 raw 模式中 `SplitArrays.y_raw` 保留完整 train/val RUL，`y` 使用只在 train engines 拟合的 raw target scaler 标准化；capped 模式保持现有 `y_raw` 与 `y_raw_uncapped` 语义，避免破坏第三章 checkpoint 协议。
- [ ] 在 `DatasetBundle.meta` 和 split manifest 中记录 `training_target=raw`、train/val uncapped label hash、target scaler identity；不把 raw val 标签用于第三章旧 checkpoint 选择。
- [ ] 保留现有 capped 测试并新增：raw train/val 中存在大于 125 的标签、raw scaler 不含 val/test、test target 不被 cap、capped 与 raw 两种 bundle identity 不相同。
- [ ] 运行 `pytest tests/test_raw_label_protocol.py tests/test_cmapss_adapter.py -q`；预期全部通过。
- [ ] 提交 `feat: add explicit full-label C-MAPSS protocol`。

## Task 2: 建立冻结 encoder loader

**Files:** `mtkdg/models/frozen_encoder.py`, `scripts/audit_frozen_checkpoints.py`, `tests/test_frozen_encoder.py`, `tests/test_checkpoint_audit.py`

- [ ] 从 point checkpoint 的 `extra.config` 重建 `MTKDGConfig` 和 dataset-specific encoder；只提取 `state_dict` 中 `encoder.` 前缀，缺 key、extra key 或 config 不一致时拒绝加载。
- [ ] 将所有 encoder 参数设为 `requires_grad=False`，调用 `eval()` 并在 wrapper 中阻止外层 `train()` 把 encoder 切回 train mode；forward 返回现有 encoder 全部诊断字段。
- [ ] 写 `encoder_fingerprint()`，按排序后的参数名、dtype、shape 和 CPU bytes 计算 SHA256；优化器只接收 adapter/readout 参数。
- [ ] 审计脚本检查 dataset、split identity、normalizer identity、model config、seed、checkpoint SHA256、validation-best metadata 和 encoder key fingerprint；不读取 test targets。
- [ ] 测试加载前后 encoder fingerprint 相同、optimizer step 后仍相同、encoder 无梯度、dropout/eval 状态固定、错误 dataset/split/key 被拒绝。
- [ ] 运行 `pytest tests/test_frozen_encoder.py tests/test_checkpoint_audit.py -q`；预期全部通过。
- [ ] 提交 `feat: add audited frozen encoder wrapper`。

## Task 3: 实现上下文条件残差 adapter

**Files:** `mtkdg/models/context_adapter.py`, `tests/test_context_adapter.py`

- [ ] 实现 `ContextFeatureBuilder`，从 encoder 输出和输入 condition 生成无标签特征；对 FD 使用 condition code/summary，对 MetroPT 使用 `obs_ratio`、scale gate、fusion summary 和 channel embedding。
- [ ] 以冻结第三章点模型在 train split 的 raw-unit 点误差作为 difficulty probe 目标。FD/MetroPT 分别按 engine/natural week 建固定 5 折；每折只用其余 4 折拟合 ridge `abs(error) ~ r(x)`（alpha=1），产生 train OOF difficulty score。特征标准化与 probe 全部只在每折训练 cluster 上拟合。
- [ ] 用 train OOF difficulty score 的固定四分位点形成 4 档；边界重复时用 `searchsorted(..., side="right")`，推理时夹到首/末 bin。保存各折 probe 参数、cluster ID、边界和 hash；validation/test 不重算边界。
- [ ] 训练集 fit `(context_key -> n_eff)` 支持表：一个 FD engine 或 MetroPT natural week 对每组最多计一次；`kappa_train=max(1, median(positive n_eff))`，未知组 `w=0`。fit API 显式要求 split=`train` 并接收 cluster IDs。
- [ ] 实现 `ContextualResidualAdapter(d_model, feature_dim)`：输出 `delta_z`、`gate` 和 `z_ctx`；严格按 `w_g` 缩放残差，未知/零支持组使用 `w=0`，使模型退化为 base representation。
- [ ] 增加 `permutation_context` 和 `disable_adapter` 诊断模式，供负对照和消融使用；两者不得改变 encoder 参数。
- [ ] 测试输入输出形状、未知组 fallback、零支持时 `z_ctx == z`、上下文置乱改变 adapter 但不改变 encoder、梯度只流向 adapter。
- [ ] 运行 `pytest tests/test_context_adapter.py -q`；预期全部通过。
- [ ] 提交 `feat: add support-shrunk contextual residual adapter`。

## Task 4: 统一概率输出与数值合同

**Files:** `mtkdg/models/prob_heads.py`, `tests/test_probabilistic_models.py`

- [ ] 定义 `ProbabilisticOutput`：包含 `family`、raw parameters、可选 quantiles、support、target unit 和 `head_version`；FD 与 MetroPT 头都转换到该合同。
- [ ] 主实验锁定 monotone quantile head，quantile levels 为 `{.05,.10,...,.90,.95}`；FD 输出 `[B,Q]`，MetroPT 输出 `[B,P,C,Q]`。Gaussian/LogNormal 只保留为合同适配测试或另行预注册的替换方案，不以 test 选择 head。
- [ ] quantile head 用单调参数化（中位数加正值 softplus 间距累积），FD raw RUL 输出经 softplus 保证非负；MetroPT 在标准化 target space 训练、raw unit 评价。所有 loss 和 score 保存其 target space。
- [ ] 让 head 接收 `z_ctx` 和上下文，但不把 adapter 逻辑复制到 head；base/head swap 能复现相同输入下的 shape contract。
- [ ] 测试 finite loss、梯度方向、正值支持、quantile 单调、raw-unit round trip，以及 FD 完整标签和 MetroPT 多步输出形状。
- [ ] 运行 `pytest tests/test_probabilistic_models.py -q`；预期全部通过。
- [ ] 提交 `feat: standardize probability output contracts`。

## Task 5: 实现 validation-only 全局/局部/收缩校准

**Files:** `mtkdg/evaluation/calibration.py`, `tests/test_calibration.py`

- [ ] 实现 `fit_global_calibrator(val_predictions, val_targets, output_context)`、`fit_local_calibrator(..., context_key, cluster_id)` 和 `fit_shrunk_calibrator(...)`；calibrator 接收的 split 标记必须为 `val_calibration`，否则报错。
- [ ] 对每个 tau 计算 validation residual `r_tau=y-q_tau`；global correction 为 residual 的 tau 分位数，local correction 在固定 context group 内同样估计。MetroPT weighted quantile 中每个 natural week 总权重为 1、周内样本均分；FD validation 每个 engine 一条末端窗口。
- [ ] 计算 validation `n_eff` 为有样本的独立 cluster 数，`kappa_val=max(1, median(positive group n_eff))`；`w_val=n_eff/(n_eff+kappa_val)`。`n_eff<5` 的局部组强制 `w_val=0`。校准完成后沿 quantile 轴作等距单调回归投影；所有 correction 记录 raw/normalized unit。
- [ ] 实现 pinball、CRPS 数值积分、50/80/90% coverage、width、PIT histogram、risk Brier 与 calibration slope；按 FD engine 或 MetroPT natural week 做 2000 次 cluster bootstrap，种子固定为 94017，报告 percentile 95% interval。窗口数不能冒充独立样本数。
- [ ] 测试 split leakage、全局退化、零事件组、未知组、完美校准、quantile monotonicity、bootstrap 单位和重复 fit 的确定性。
- [ ] 运行 `pytest tests/test_calibration.py -q`；预期全部通过。
- [ ] 提交 `feat: add validation-only contextual probability calibration`。

## Task 6: 实现 readout-only runner

**Files:** `mtkdg/training/probabilistic_runner.py`, `scripts/run_contextual_probability.py`, `tests/test_probabilistic_runner.py`

- [ ] 同一 runner 训练 C0（base probabilistic readout）和 C1（context adapter + 同 readout）；FD 另训练 Cpoint（相同 encoder、raw train labels、固定预算的 point readout），用于区分完整标签监督收益与概率/adapter 收益。对照使用同 seed、batch order、epoch、optimizer 和 scheduler。
- [ ] 使用预注册的固定 epochs、batch size、seed 和 scheduler；不做 validation early stopping 或 epoch 选择。epoch 数在 GPU 运行前锁定；若先做单 seed pilot，只可基于 train loss 稳定性判断，pilot 不运行 test。
- [ ] 只在正确匹配的 validation calibration view 上拟合 calibrator；保存 C0/C1 raw predictions、global/local/shrunk 校准值和校准前后 validation 输出；仅 `--evaluate-test` 显式执行一次 test。
- [ ] CLI 显式要求 dataset、seed、point checkpoint、training target、head family、context schema、output dir 和 data root；拒绝缺失 checkpoint、重复 run dir、未知 protocol version。
- [ ] 测试 synthetic FD/MetroPT batch 的 forward-loss-backward、adapter-only gradients、calibrator validation-only、checkpoint reload、test-once guard 和错误协议拒绝。
- [ ] 运行 `pytest tests/test_probabilistic_runner.py -q`；预期全部通过。
- [ ] 提交 `feat: add contextual probabilistic runner`。

## Task 7: 写出可审计 artifacts 与 checkpoint inventory

**Files:** `mtkdg/evaluation/probability_artifacts.py`, `scripts/audit_frozen_checkpoints.py`, `tests/test_probability_artifacts.py`

- [ ] 写出 `config.json`、`manifest.json`、`encoder_checkpoint_sha256`、`encoder_fingerprint`、`encoder_frozen`、adapter/calibrator versions、context schema、support counts、kappa、validation history、calibration parameters、validation/test arrays 和 `evaluated_once`。
- [ ] 保存 raw targets、normalized targets、base/adapter/calibrated parameters 或 quantiles、split identity、scaler identity、code SHA/dirty state；预测数组与 target 数量严格一致。
- [ ] 任何缺少 validation provenance、encoder fingerprint、training target 或 quality-gate status 的 artifact 都不能进入第五章。
- [ ] 测试 round trip、hash 变化、重复写出不覆盖旧 run、test marker、缺字段拒绝和 failed-gate refusal。
- [ ] 运行 `pytest tests/test_probability_artifacts.py -q`；预期全部通过。
- [ ] 提交 `feat: add auditable probability artifacts`。

## Task 8: 冻结预注册与最小矩阵

**Files:** `refine-logs/CH4_CONTEXTUAL_PROBABILITY_PREREGISTRATION.md`, `scripts/run_contextual_probability_matrix.sh`, `refine-logs/CH4_CONTEXTUAL_PROBABILITY_RUN_LOG.md`

- [ ] 写明第三章 checkpoint 不变、FD full-label protocol、MetroPT split、head family、context features、difficulty bins、kappa rule、fixed epochs、calibration fit、bootstrap units、一次性 test 和 failure policy。
- [ ] 写明对照：Cpoint（FD only, raw-label point readout）；C0 frozen encoder + base probabilistic readout；C1 frozen encoder + contextual residual adapter + same readout；C2 C1 + validation-only shrunk calibration（主方法）；C3 C1 with train-context permutation negative control。C2 校准阶段不增加训练 epochs。
- [ ] 建议正式 GPU 矩阵为 `FD002/FD004/MetroPT-3 × seeds 2026/2027/2028 × C0/C1/C3`（27 个训练 runs；C2 是每个 C1 run 的 validation-only calibrator artifact，不新增 GPU run；Cpoint 为 FD002/FD004 各 3 seeds，共 6 个训练 runs）。这是计划建议，run 集合、head、epochs 和失败重试规则必须在 GPU 前经用户授权并写入预注册变更日志。
- [ ] 质量门启动前冻结：encoder fingerprint 逐 run 完全不变；输出 finite、RUL quantiles 非负且分位单调；C1 对 C0 的 point projection RMSE 非劣界为相对恶化不超过 2%；C2 对 C1 的 CRPS 或宏平均 90% coverage error 至少一项改善，另一项恶化不超过 2%/2 percentage points；改善方向以预注册的 paired cluster-bootstrap 区间报告，不从 context group 中择优。门失败则保留负结果并停止对应数据集的第五章主张。
- [ ] 运行脚本只检查 inventory、唯一 run dir 和命令，不启动 GPU；提交预注册和空 run log。
- [ ] 提交 `docs: preregister contextual probability matrix`。

## Task 9: CPU end-to-end verification and launch gate

**Files:** `tests/test_contextual_probability_e2e.py`, `refine-logs/CH4_CONTEXTUAL_PROBABILITY_LAUNCH_GATE.md`

- [ ] 用合成 FD raw-label 和 MetroPT multi-horizon 数据完成 CPU 小批量：load checkpoint fixture、冻结 encoder、生成 OOF difficulty、训练 adapter/head、构造 FD terminal/MetroPT non-overlap validation views、校准、写 artifacts、reload、test-once。
- [ ] 运行聚焦测试：`pytest tests/test_raw_label_protocol.py tests/test_frozen_encoder.py tests/test_context_adapter.py tests/test_calibration.py tests/test_probabilistic_runner.py tests/test_probability_artifacts.py tests/test_contextual_probability_e2e.py -q`。
- [ ] 运行现有回归测试 `pytest -q`，记录完整/中断状态，不把用户中断的 pytest 说成通过。
- [ ] launch gate 只在测试、checkpoint audit、预注册、artifact schema 和 run authorization 全部具备后标记 PASS；否则明确 BLOCKED，不启动 GPU。
- [ ] 提交 `test: pass contextual probability CPU launch gate`。

## Task 10: GPU 矩阵、统计汇总与第五章接口

**Files:** `refine-logs/CH4_CONTEXTUAL_PROBABILITY_RUN_LOG.md`, `refine-logs/CH4_CONTEXTUAL_PROBABILITY_RESULTS.md`, `scripts/summarize_contextual_probability.py`, `docs/superpowers/plans/2026-10-18-ch5-risk-decision.md`

- [ ] 仅在 launch gate 和用户授权后运行正式矩阵；每次 run 记录命令、start/end、exit code、checkpoint hash、run dir、failure/retry，禁止静默重跑。
- [ ] test 仅执行一次；按 FD engine、MetroPT time block/natural week bootstrap，逐 seed 报告，不把 n=3 当传统显著性证据。
- [ ] 汇总 base/adapter/calibrated 的 raw-unit proper scores、coverage、width、conditional reliability、risk-coverage 和预注册代价；FD failure event 与 MetroPT engineering exceedance 分开。
- [ ] 只有通过第四章质量门的数据集输出才能被第五章读取；第五章不重训 encoder/adapter/head，不在 test 上选 calibrator、阈值或代价。
- [ ] 若 C2 无法优于 C0 或质量门失败，保留失败和机制诊断，收窄第四章结论为“当前表示与 regime 下未获得可靠条件校准”，不得改 head/bin 后重跑。
- [ ] 提交 `results: summarize contextual probability matrix`。

---

## 预期资源与停止规则

- 代码与 CPU 合同测试：约 1.5–2 天；checkpoint audit、artifact 和预注册：约 0.5 天。
- 小规模 GPU 诊断：每配置单 seed，先估计不超过 2 小时；超过则停止并记录，不进入正式矩阵。
- 正式矩阵建议为 27 个训练 runs 加 6 个 FD Cpoint runs；每个 C1 run 产生一个 C2 校准 artifact。GPU 总量和完成时间须以实现后的单 run dry-run 估计，不能先验承诺半天内完成。
- 不解冻 encoder，不以 test 结果改 context bins、kappa、head、risk threshold 或 run 集合；所有变更进入 append-only change log。

## 自审清单

- [ ] 完整 FD 标签与第三章 capped 结果有明确协议边界。
- [ ] Adapter 位于 encoder 之后，所有 encoder 参数冻结并有 fingerprint 测试。
- [ ] Context features 不读取标签，support counts 只由 train 计算，calibrator 只由 validation 拟合。
- [ ] 结果按 engine/time block 汇总，窗口不冒充独立样本。
- [ ] 输出头可替换，创新主张集中在 residual adapter、support shrinkage 和可靠性证据链。
- [ ] 旧删失/MC-Graph 方案和历史 artifacts 不删除、不覆盖。
