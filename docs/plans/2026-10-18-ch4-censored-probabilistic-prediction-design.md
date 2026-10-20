# 第四章与第五章方向设计：删失感知概率预测与风险决策

**日期：** 2026-10-18  
**设计基点：** `7aff88283b5d7ba72fd0bbc909d324cf9937906f`  
**状态：** 已批准，尚未进入实现与实验

## 1. 目标与章节边界

第四章研究冻结 MTK-DG 表示上的概率 readout，检验其能否在现有数据协议下刻画预测不确定性。FD002/FD004 使用 capped@125 训练标签研究右删失 RUL 分布；MetroPT-3 使用多步、多通道概率预测刻画异方差和 horizon 误差变化。

第五章作为纯后处理决策层，消费第四章冻结的预测分布，进行 validation-only 风险校准，并评估风险排序、预警质量和预先定义的代价情景。不新增 encoder、不重训表示，也不把模拟代价解释为真实维护收益。

第四章不再声称 availability-conditioned graph 的独立机制贡献。已完成的 MC-Graph 和 outage 实验原样保留为历史证据，可用于背景、限制或附录。FD 的删失似然只提供“真实 RUL 至少为 125”的信息；阈值以上分布形状仍依赖模型假设，不能宣称由删失数据本身识别。

## 2. 建议主实验

### 2.1 表示与训练

- 使用每个训练 seed 对应的第三章 validation-best checkpoint；固定 encoder 权重，只训练概率 readout。
- 沿用第三章 V1 数据切分、归一化、训练种子 `2026/2027/2028` 和一次性 test 约束。
- 每个 run 记录 encoder checkpoint 哈希、split/scaler 身份、概率头版本、训练配置和预测 artifact。
- 若缺少任何 seed 对应的第三章 checkpoint，则三 seed 配对主设计无法成立。此时将其记为启动阻塞项；不得复制其他 seed 权重或擅自重训第三章补齐。

### 2.2 数据集专用 readout

| 数据集         | 概率 readout              | 训练目标                                        | 验证选型                      |
| ----------- | ----------------------- | ------------------------------------------- | ------------------------- |
| FD002/FD004 | 正值 LogNormal RUL 分布     | 未删失标签用 LogNormal 密度；`y_capped=125` 用右删失生存概率 | validation censored NLL   |
| MetroPT-3   | 单调分位数头，分位水平 0.05 至 0.95 | 分位数 pinball loss                            | validation 加权 pinball/WIS |

FD 的删失状态必须从 capped 标签判定，LogNormal 的密度和生存概率以 raw-cycle 尺度定义。实现须使用数值稳定的对数生存概率和正值尺度参数化。MetroPT 的有限分位网格只支持 weighted pinball/WIS 口径；没有明确尾部插值与外推定义前，不将其称为完整分布 CRPS。

### 2.3 对照和评估

- 主参照为第三章同 seed 的冻结点预测 checkpoint，不为本章新增点预测重跑。
- 增加一个零训练的点预测残差分布基线，使概率分数可直接比较：以 validation 残差的经验分布包装第三章点预测；FD 按 engine 形成残差样本，MetroPT 按通道与 horizon 形成残差样本。基线仅用 validation 构造，test 不参与残差分布估计。MetroPT 的有限样本支持范围和 FD validation engine 数量须在实施前报告。
- FD 官方 raw test 报 raw-target NLL、50%/90% 区间覆盖率与区间宽度；`RUL > 125` 子组单列，解释为模型外推表现而非删失数据可识别保证。
- MetroPT test 报整体与逐通道 weighted pinball/WIS、50%/90% 覆盖率和区间宽度；逐月结果用于描述时间迁移，不据此调整模型或校准器。
- 逐 seed 列值并报告 mean ± sample std；FD bootstrap 单位为 engine，MetroPT bootstrap 单位为自然周。`n=3` 不支持传统显著性结论。
- 所有概率分数和区间指标均在原始目标尺度或明确说明的标准化尺度上计算；跨传感器 raw-unit 指标不合并成“均衡性”结论。

## 3. 第五章消费接口

- FD 风险事件为 `RUL < tau`，`tau` 与维护提前期扫描范围在实现前预注册；校准标签使用 validation 的 `y_raw_uncapped`。
- MetroPT 越限阈值由 train-only 每通道 p99.5 得到，只称工程越限代理，不称真实故障标签。
- 风险概率校准、决策阈值和代价比例全部只由 validation 确定；test 仅一次性计算 Brier、ECE/reliability、risk-coverage、hit/miss/false alarm、lead time 和预注册代价表。
- 第五章只消费通过第四章概率质量门的数据集输出；某数据集未通过时，停止该数据集的概率决策主张，不能在 test 上选校准器或改风险阈值。

## 4. 质量门与停止规则

1. GPU 训练前，用合成数据测试删失 NLL 的有限值、梯度方向与极端尾部稳定性，验证 quantile 单调性、反归一化、WIS、coverage 和宽度计算。
2. 训练采用三 seed 配对设计，不以单 seed 结果决定模型结构或超参。每个概率 run 单独保存 validation-best checkpoint，test 只评估一次。
3. 若某数据集验证概率分数/校准质量门失败，则停止该数据集的概率决策主张，并完整保留负结果。
4. 不解冻 encoder，不追加概率头，不调整 cap、分位网格或风险阈值后重跑。任何研究问题、run 集合、头或 bin 变更都必须另行授权，并在该分支的预注册变更日志中记录。
5. 当前 `mtkdg/models/prob_heads.py` 视为原型而非已验证实现；其损失、runner 接线、raw-unit 评估、artifact 和数值稳定性均需经过实现阶段审查。

## 5. 实施前置检查

- 核对三个 seed 的第三章 checkpoint 是否可用，并确认其模型结构、训练协议、数据身份与本设计基点兼容。
- 明确 validation censored NLL 和 MetroPT weighted pinball/WIS 的最佳 checkpoint 规则、patience 行为及概率质量门数值阈值，冻结后再训练。
- 明确 LogNormal 的 raw-cycle 密度定义、WIS 分位权重、区间端点算法、校准器类别、bootstrap 重采样细节及 artifact schema。
- 在上述定义冻结前，不启动训练、不查看新的 test 结果、不产生新的实验 run。

## 6. 与 7aff882 证据的关系

本设计使用 7aff882 中的第三章 V1 协议和冻结点预测结果作为参照。三 seed MC-Graph 结果显示 availability-conditioned graph 增量未达到预注册同向判据，因此该机制不作为本设计的核心贡献。概率 readout 是新研究阶段；它不改变、覆盖或重新解释原有 MC-Graph、outage 或鲁棒矩阵结果。
