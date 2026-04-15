# Commit Log

记录每次有意义的修改，保持代码变更可追溯。

---

## 2026-04-10: OPD 自定义 logging — 补充 math/deepscaler accuracy

### 背景

OPD (On-Policy Distillation) 的 `reward_func` 返回的是 teacher model 的原始 JSON response（包含 logprobs），而非标量 reward。这导致：

1. **Eval 崩溃**：`_log_eval_rollout_data` 中 `sum(rewards)` 对 dict 列表报 `TypeError`
2. **Train 无 accuracy**：`post_process_rewards` 将 reward 统一设为 `0.0`，wandb 上看不到模型能力变化

### 方案

通过 slime 已有的 custom log function 机制，新建 `slime/utils/opd_log.py`，在 log 阶段对 samples 补算 rule-based accuracy（`grade_answer_verl` + `get_deepscaler_rule_based_reward`），不改动 RM 调度链路。

### 改动文件

| 文件 | 改动 |
|---|---|
| `slime/utils/opd_log.py` | **新建**。两个 custom log 函数：`log_rollout_data`（train）、`log_eval_rollout_data`（eval）。计算 math_accuracy 和 deepscaler_accuracy 并写入 wandb |
| `slime/ray/rollout.py` | `_start_router` 中 `disable_health_check` 加 `hasattr` 保护，兼容旧版 sglang |
| `examples/on_policy_distillation/run-qwen3-8B-opd_dev.sh` | RM_ARGS 添加 `--custom-rollout-log-function-path` 和 `--custom-eval-rollout-log-function-path`；ROLLOUT_ARGS 添加 `--label-key label`（否则 sample.label 为 None） |

### wandb 指标

- Train: `rollout/math_accuracy`, `rollout/deepscaler_accuracy`
- Eval: `eval/<dataset>/math_accuracy`, `eval/<dataset>/deepscaler_accuracy`

### 注意事项

- `--label-key label` 必须指定，否则数据加载时不读 label 字段
- eval 的 `label_key` 通过 fallback 链自动继承 `--label-key`
- `deepscaler` 要求 response 有 `</think>` 结构才提取答案；`math` 直接从全文提取 `\boxed{}`
