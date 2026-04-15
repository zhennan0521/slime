"""Custom logging functions for OPD (On-Policy Distillation).

Computes math and deepscaler accuracy on rollout/eval samples and logs to wandb.
OPD reward_func returns teacher logprobs (not scalar rewards), so the default
logging would crash on eval. These custom functions handle that and add accuracy metrics.

Usage in script:
    --custom-rollout-log-function-path slime.utils.opd_log.log_rollout_data
    --custom-eval-rollout-log-function-path slime.utils.opd_log.log_eval_rollout_data
"""

import logging
from typing import Any

from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
from slime.rollout.rm_hub.math_utils import grade_answer_verl
from slime.utils import logging_utils
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _compute_accuracy(samples: list[Sample]) -> dict[str, float]:
    """Compute math and deepscaler accuracy on samples that have labels."""
    math_scores = []
    deepscaler_scores = []

    for sample in samples:
        if sample.label is None:
            logger.error(f"Sample {sample.index} has no label")
            continue
        response = sample.response or ""
        label = sample.label

        math_scores.append(1.0 if grade_answer_verl(response, label) else 0.0)
        deepscaler_scores.append(float(get_deepscaler_rule_based_reward(response, label)))

    metrics = {}
    if math_scores:
        metrics["math_accuracy"] = sum(math_scores) / len(math_scores)
    if deepscaler_scores:
        metrics["deepscaler_accuracy"] = sum(deepscaler_scores) / len(deepscaler_scores)
    return metrics


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    """Custom train rollout log: default metrics + math/deepscaler accuracy."""
    from slime.ray.rollout import (
        compute_metrics_from_samples,
        compute_perf_metrics_from_samples,
        compute_rollout_step,
        dict_add_prefix,
    )

    if args.load_debug_rollout_data:
        return True

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")

    accuracy = _compute_accuracy(samples)
    for k, v in accuracy.items():
        log_dict[f"rollout/{k}"] = v

    logger.info(f"rollout {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")

    return True


def log_eval_rollout_data(rollout_id, args, data, extra_metrics):
    """Custom eval log: math/deepscaler accuracy instead of broken sum(dict_rewards)."""
    from slime.ray.rollout import (
        compute_metrics_from_samples,
        compute_rollout_step,
        dict_add_prefix,
    )

    log_dict = extra_metrics or {}

    for key in data.keys():
        samples = data[key].get("samples", [])
        if samples:
            accuracy = _compute_accuracy(samples)
            for ak, av in accuracy.items():
                log_dict[f"eval/{key}/{ak}"] = av
            log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), f"eval/{key}/")

        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)

    logger.info(f"eval {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    logging_utils.log(args, log_dict, step_key="eval/step")

    return True
