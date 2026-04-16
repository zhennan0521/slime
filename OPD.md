# OPD (On-Policy Distillation) 开发笔记

## 项目概述

在 [slime](https://github.com/zhennan0521/slime) 框架上做 On-Policy Distillation（OPD），用 Qwen3-32B 作为 teacher，Qwen3-8B 作为 student，通过 GRPO + KL penalty 进行在线蒸馏。

分支：`opd_dev`

---

## 关键代码路径

### 资源调度

- `slime/ray/placement_group.py` — 创建 placement group，策略是 **PACK**，按 (node IP, GPU ID) 排序
  - 排序后列表：`[actor GPUs ... | rollout GPUs ...]`，由 `rollout_offset` 分割
  - Actor GPU 拿排序最前面的节点（IP 最小），rollout 拿剩下的
  - **Actor 不会跨节点**（PACK + 排序保证同节点 GPU 排在一起）

### Rollout 与 Reward

- `slime/rollout/sglang_rollout.py` — 默认 rollout function
  - `generate_and_rm()` → `async_rm()` → 根据 `rm_type` 或 `custom_rm_path` 算 reward
  - eval 时走 `eval_rollout_single_dataset()`，复用同一个 `generate_and_rm`
- `slime/rollout/on_policy_distillation.py` — OPD 专用
  - `reward_func`：调 teacher server 拿 logprobs，返回原始 JSON dict（不是标量 reward）
  - `post_process_rewards`：提取 teacher log-probs 存到 `sample.teacher_log_probs`，返回 `[0.0, ...]` 作为标量 reward
- `slime/rollout/rm_hub/__init__.py` — RM 调度
  - `custom_rm_path` 优先级高于 `rm_type`，eval 无法单独绕过

### Logging

- `slime/utils/logging_utils.py` — `log(args, metrics, step_key)` → `wandb.log(metrics)`
- `slime/ray/rollout.py`
  - `_log_rollout_data()` — train 每步 rollout 的 log，支持 `--custom-rollout-log-function-path`
  - `_log_eval_rollout_data()` — eval 的 log，支持 `--custom-eval-rollout-log-function-path`
  - 两者都会调用 `logging_utils.log` 打到 wandb
- **wandb step 注意**：`wandb.log()` 没传 step 参数，x 轴是调用次数不是 train step。看 accuracy 时 x 轴切成 `eval/step` 或 `rollout/step`

### Verify 函数（accuracy 计算）

- `slime/rollout/rm_hub/math_utils.py`
  - `grade_answer_verl(response, label)` — 从整个 response 提取 `\boxed{}`，mathd + sympy 判题
- `slime/rollout/rm_hub/deepscaler.py`
  - `get_deepscaler_rule_based_reward(response, label)` — 先按 `</think>` 分割，只看后半部分提取 `\boxed{}`
  - **需要 `</think>` 结构**，没有则直接返回 0

---

## 已解决的问题

### 1. sglang Router 兼容性（disable_health_check）

**现象**：`Router.__new__() got an unexpected keyword argument 'disable_health_check'`
**原因**：slime 代码设了 `router_args.disable_health_check = True`，但当前 sglang 版本不支持
**修复**：`slime/ray/rollout.py:945` 加 `hasattr` 保护

### 2. Eval 崩溃（sum(rewards) on dict）

**现象**：`TypeError: unsupported operand type(s) for +: 'int' and 'dict'`
**原因**：OPD 的 `reward_func` 返回 teacher server 的原始 JSON（dict），eval 默认 log 代码做 `sum(rewards)` 炸了
**修复**：通过 custom log function 机制，新建 `slime/utils/opd_log.py`，eval 时计算 math/deepscaler accuracy 替代

### 3. sample.label 为 None

**现象**：数据 jsonl 里有 label 字段，但 sample.label 是 None
**原因**：`--label-key` 默认 `None`，`data.py` 中 `label=data[label_key] if label_key is not None else None`
**修复**：脚本加 `--label-key label`，eval 通过 fallback 链自动继承

### 4. wandb API key 多节点问题

**现象**：Ray 把 actor 调度到了没有 wandb 凭据的节点
**原因**：Ray cluster 有多节点，placement group 在 head 节点，但 actor 可能被调度到其他节点
**修复**：在 ray job submit 的 env_vars 里传 `WANDB_API_KEY`，或所有节点提前 `wandb login`

---

## 我们新增的文件

### `slime/utils/opd_log.py`

OPD 专用 custom log functions：
- `log_rollout_data` — train rollout log，在默认指标基础上加 math/deepscaler accuracy
- `log_eval_rollout_data` — eval log，绕过 `sum(dict_rewards)` 崩溃，直接算 accuracy

通过脚本参数指定：
```bash
--custom-rollout-log-function-path slime.utils.opd_log.log_rollout_data
--custom-eval-rollout-log-function-path slime.utils.opd_log.log_eval_rollout_data
```

wandb 指标：`rollout/math_accuracy`, `rollout/deepscaler_accuracy`, `eval/<dataset>/math_accuracy`, `eval/<dataset>/deepscaler_accuracy`

---

## 资源分配方案

### 单节点（1 node x 8 GPUs）

```
--actor-num-nodes 1 --actor-num-gpus-per-node 2 --rollout-num-gpus 4
--tensor-model-parallel-size 2
```

GPU 7 给 teacher（`CUDA_VISIBLE_DEVICES=7`），剩余 7 张给 Ray（2 actor + 4 rollout + 1 ref）

### 多节点（4 nodes x 8 GPUs + 1 teacher node）

```
--actor-num-nodes 1 --actor-num-gpus-per-node 8 --rollout-num-gpus 24
--tensor-model-parallel-size 2
```

- Node 0（IP 最小）：8 GPU 全给 actor（TP=2, DP=4）
- Node 1-3：各 8 GPU 做 rollout（共 24 个 sglang engine）
- **第 5 个独立节点**：跑 teacher sglang server（tp=2，Qwen3-32B）
- 脚本中 `TEACHER_IP` 填第 5 节点 IP，不在训练节点本地启 teacher

---

## 待办

- [ ] 4 节点脚本（`run-qwen3-8B-opd_dev_4nodes.sh`）还没加 `--label-key label` 和两个 custom log path
- [ ] 跑通验证单节点 + 多节点
- [ ] 确认 wandb 上 math_accuracy / deepscaler_accuracy 指标正常
- [ ] 考虑 train 过程中 reward 均值是否需要额外记录（目前 OPD 恒为 0.0）

---

## 环境备忘

- 代理：`http://221.194.188.92:3128`
- GitHub repo：`https://github.com/zhennan0521/slime.git`（需 token 认证）
- wandb：`http://11.71.1.153:8080`（自建实例，entity=automl）
- Git user：`zhennanshen <1641225799@qq.com>`

