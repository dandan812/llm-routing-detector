from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

from .utils import canonical_json, quantile, sample_multinomial, softmax, wilson_interval


SCHEMA_VERSION = 2
SCORING_VERSION = "trusted-likelihood-v2"
MODELS = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")
TAU_GRID = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
TEMPERATURE_GRID = (0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0)
SMOOTHING = 0.5
MIN_WINDOWS = 3
MIN_VALID_RATE = 0.95
PASS_ERROR_UPPER_MAX = 0.05
PASS_COVERAGE_MIN = 0.90
PASS_WORST_WINDOW_COVERAGE_MIN = 0.80
ALERT_FALSE_RATE_UPPER_MAX = 0.01
SECOND_SHARE_MIN = 0.10


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, float(value)))


def _safe_log(value: float) -> float:
    return math.log(max(float(value), 1e-300))


def _distribution(counts: dict[str, int], categories: tuple[str, ...], alpha: float = SMOOTHING) -> dict[str, float]:
    total = sum(max(0, int(counts.get(category, 0))) for category in categories)
    denominator = total + alpha * len(categories)
    if denominator <= 0:
        return {category: 1.0 / len(categories) for category in categories}
    return {
        category: (max(0, int(counts.get(category, 0))) + alpha) / denominator
        for category in categories
    }


def js_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    categories = sorted(set(left) | set(right))
    if not categories:
        return 0.0
    middle = {category: 0.5 * (left.get(category, 0.0) + right.get(category, 0.0)) for category in categories}

    def kl(source: dict[str, float]) -> float:
        return sum(
            probability * math.log2(probability / middle[category])
            for category, probability in source.items()
            if probability > 0 and middle[category] > 0
        )

    return 0.5 * kl(left) + 0.5 * kl(right)


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _pairwise_mean(distributions: list[dict[str, float]]) -> float:
    values = [
        js_divergence(distributions[left], distributions[right])
        for left in range(len(distributions))
        for right in range(left + 1, len(distributions))
    ]
    return _mean(values)


def _window_map(profile: dict[str, Any], model: str) -> dict[str, dict[str, Any]]:
    return {
        str(key): value
        for key, value in profile.get("models", {}).get(model, {}).get("windows", {}).items()
    }


def _cell_categories(profile: dict[str, Any]) -> tuple[str, ...]:
    categories = {
        str(category)
        for model in MODELS
        for window in _window_map(profile, model).values()
        for category in (window.get("counts") or {})
    }
    categories.add("__OTHER__")
    categories.add("__INVALID_OUTPUT__")
    return tuple(sorted(categories))


def _aggregate_windows(profile: dict[str, Any], model: str, excluded_window: str | None = None) -> Counter[str]:
    counts: Counter[str] = Counter()
    for window_id, window in _window_map(profile, model).items():
        if excluded_window is not None and str(window_id) == str(excluded_window):
            continue
        counts.update({str(key): int(value) for key, value in (window.get("counts") or {}).items()})
    return counts


def fit_cell(profile: dict[str, Any], tau: float, *, excluded_window: str | None = None) -> dict[str, Any]:
    categories = _cell_categories(profile)
    model_distributions = {
        model: _distribution(dict(_aggregate_windows(profile, model, excluded_window)), categories)
        for model in MODELS
    }
    between = _pairwise_mean([model_distributions[model] for model in MODELS])
    within_values: list[float] = []
    for model in MODELS:
        windows = [
            _distribution({str(key): int(value) for key, value in (window.get("counts") or {}).items()}, categories)
            for window_id, window in _window_map(profile, model).items()
            if excluded_window is None or str(window_id) != str(excluded_window)
        ]
        if len(windows) >= 2:
            within_values.append(_pairwise_mean(windows))
    drift = _mean(within_values)
    weight = _clip((between - drift) / (between + float(tau))) if between + float(tau) > 0 else 0.0
    pooled = {
        category: _mean(model_distributions[model][category] for model in MODELS)
        for category in categories
    }
    used = {
        model: {
            category: weight * model_distributions[model][category] + (1.0 - weight) * pooled[category]
            for category in categories
        }
        for model in MODELS
    }
    return {
        "categories": list(categories),
        "model_distributions": model_distributions,
        "pooled_distribution": pooled,
        "between_model_jsd": between,
        "within_model_jsd": drift,
        "tau": float(tau),
        "weight": weight,
        "used_distributions": used,
    }


def _normalize_candidate_counts(counts: dict[str, int], categories: list[str]) -> dict[str, int]:
    allowed = set(categories)
    result = Counter({category: 0 for category in categories})
    for raw_category, raw_count in counts.items():
        category = str(raw_category)
        if category not in allowed:
            category = "__OTHER__"
        result[category] += int(raw_count)
    return dict(result)


def _cell_log_likelihood(counts: dict[str, int], distribution: dict[str, float]) -> float:
    return sum(int(count) * _safe_log(distribution.get(category, 0.0)) for category, count in counts.items())


def _family_scores(cells: dict[str, dict[str, Any]]) -> tuple[dict[str, float], dict[str, Any]]:
    families: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for key, cell in cells.items():
        if cell.get("isolated_ood"):
            continue
        families[str(cell["probe_id"])].append((key, cell))
    total_scores = {model: _safe_log(1.0 / 3.0) for model in MODELS}
    details: dict[str, Any] = {}
    for family, entries in sorted(families.items()):
        profile_weights = [max(0.0, float(cell["weight"])) for _key, cell in entries]
        weight_sum = sum(profile_weights)
        if weight_sum <= 0:
            continue
        family_weight = min(1.0, max(profile_weights))
        scores = {
            model: sum(
                profile_weight * float(cell["average_log_likelihood"][model])
                for profile_weight, (_key, cell) in zip(profile_weights, entries)
            ) / weight_sum
            for model in MODELS
        }
        for model in MODELS:
            total_scores[model] += family_weight * scores[model]
        details[family] = {
            "family_weight": family_weight,
            "profile_count": len(entries),
            "profile_keys": [key for key, _cell in entries],
            "model_contributions": {model: family_weight * scores[model] for model in MODELS},
        }
    return total_scores, details


def _mixture_fit(cells: dict[str, dict[str, Any]], family_details: dict[str, Any]) -> dict[str, Any]:
    observations: list[tuple[float, int, tuple[float, float, float]]] = []
    for family, details in family_details.items():
        entries = [cells[key] for key in details["profile_keys"]]
        profile_weight_sum = sum(max(0.0, float(cell["weight"])) for cell in entries)
        if profile_weight_sum <= 0:
            continue
        for cell in entries:
            n = max(1, int(cell["sample_count"]))
            profile_share = max(0.0, float(cell["weight"])) / profile_weight_sum
            coefficient = float(details["family_weight"]) * profile_share / n
            distributions = cell["used_distributions"]
            for category, count in cell["counts"].items():
                if count <= 0:
                    continue
                observations.append((
                    coefficient,
                    int(count),
                    tuple(float(distributions[model].get(category, 0.0)) for model in MODELS),
                ))

    def log_likelihood(q: list[float]) -> float:
        return sum(
            coefficient * count * _safe_log(sum(q[index] * probabilities[index] for index in range(3)))
            for coefficient, count, probabilities in observations
        )

    if not observations:
        return {
            "proportions": {},
            "identifiable_model_groups": [list(MODELS)],
            "best_mixture_log_likelihood": None,
            "best_pure_log_likelihood": None,
            "mixture_gain": 0.0,
            "second_share": 0.0,
            "identifiable": False,
        }
    starts = ([1 / 3, 1 / 3, 1 / 3], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8])
    best_q = list(starts[0])
    best_ll = float("-inf")
    for start in starts:
        q = list(start)
        for _ in range(500):
            responsibilities = [0.0, 0.0, 0.0]
            total_weight = 0.0
            for coefficient, count, probabilities in observations:
                denominator = sum(q[index] * probabilities[index] for index in range(3))
                if denominator <= 0:
                    continue
                weight = coefficient * count
                total_weight += weight
                for index in range(3):
                    responsibilities[index] += weight * q[index] * probabilities[index] / denominator
            if total_weight <= 0:
                break
            updated = [value / total_weight for value in responsibilities]
            if max(abs(updated[index] - q[index]) for index in range(3)) < 1e-10:
                q = updated
                break
            q = updated
        current_ll = log_likelihood(q)
        if current_ll > best_ll:
            best_ll = current_ll
            best_q = q
    pure_values = [log_likelihood([1.0 if index == target else 0.0 for index in range(3)]) for target in range(3)]
    best_pure = max(pure_values)
    distinguishable_groups: list[list[int]] = []
    for model_index in range(len(MODELS)):
        matched = False
        for group in distinguishable_groups:
            representative = group[0]
            identical = all(
                all(
                    abs(float(cell["used_distributions"][MODELS[model_index]].get(category, 0.0)) - float(cell["used_distributions"][MODELS[representative]].get(category, 0.0))) <= 1e-12
                    for category in cell["categories"]
                )
                for cell in cells.values()
                if cell.get("complete") and not cell.get("isolated_ood") and float(cell.get("weight", 0.0)) > 0
            )
            if identical:
                group.append(model_index)
                matched = True
                break
        if not matched:
            distinguishable_groups.append([model_index])
    grouped = [
        {
            "models": [MODELS[index] for index in group],
            "share": sum(best_q[index] for index in group),
        }
        for group in distinguishable_groups
    ]
    ordered_group_shares = sorted((item["share"] for item in grouped), reverse=True)
    return {
        "proportions": {
            "+".join(item["models"]): item["share"]
            for item in grouped
        },
        "identifiable_model_groups": [item["models"] for item in grouped],
        "best_mixture_log_likelihood": best_ll,
        "best_pure_log_likelihood": best_pure,
        "mixture_gain": max(0.0, best_ll - best_pure),
        "second_share": ordered_group_shares[1] if len(ordered_group_shares) > 1 else 0.0,
        "identifiable": all(len(item["models"]) == 1 for item in grouped),
    }


def _score_cells(
    fitted_cells: dict[str, dict[str, Any]],
    candidate_counts: dict[str, dict[str, int]],
    required_counts: dict[str, int],
    ood_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    ood_thresholds = ood_thresholds or {}
    for key, fitted in fitted_cells.items():
        required = int(required_counts.get(key, 0))
        raw_counts = candidate_counts.get(key, {})
        counts = _normalize_candidate_counts(raw_counts, fitted["categories"])
        sample_count = sum(counts.values())
        if sample_count < required:
            missing.append(key)
        average_ll = {
            model: _cell_log_likelihood(counts, fitted["used_distributions"][model]) / max(1, sample_count)
            for model in MODELS
        }
        best_local = max(average_ll.values())
        raw_threshold = ood_thresholds.get(key)
        threshold = float(raw_threshold) if raw_threshold is not None else None
        isolated = sample_count >= required > 0 and threshold is not None and best_local < threshold
        probe_id, profile = key.split("|", 1)
        cells[key] = {
            "probe_id": probe_id,
            "profile": profile,
            "counts": counts,
            "sample_count": sample_count,
            "required_samples": required,
            "complete": sample_count >= required,
            "weight": fitted["weight"],
            "between_model_jsd": fitted["between_model_jsd"],
            "within_model_jsd": fitted["within_model_jsd"],
            "tau": fitted["tau"],
            "categories": list(fitted["categories"]),
            "used_distributions": fitted["used_distributions"],
            "average_log_likelihood": average_ll,
            "best_local_fit": best_local,
            "ood_threshold": threshold,
            "isolated_ood": isolated,
            "ood_reason": "below_trusted_heldout_1pct" if isolated else None,
        }
    scores, family_details = _family_scores(cells)
    mixture = _mixture_fit(cells, family_details)
    return {
        "scores": scores,
        "families": family_details,
        "cells": cells,
        "mixture": mixture,
        "missing_cells": sorted(missing),
        "non_ood_family_count": len(family_details),
    }


def _runtime_signature(runtime_spec: dict[str, Any]) -> str:
    canonical = {
        "cells": {str(key): int(value) for key, value in sorted((runtime_spec.get("cells") or {}).items())},
        "contracts": {
            str(key): value
            for key, value in sorted((runtime_spec.get("contracts") or {}).items())
        },
    }
    return hashlib.sha256(canonical_json(canonical).encode("utf-8")).hexdigest()


def _expected_contract(key: str, probe_metadata: dict[str, Any]) -> dict[str, Any] | None:
    probe_id, profile = key.split("|", 1)
    metadata = probe_metadata.get(probe_id)
    if not isinstance(metadata, dict) or "+" not in profile:
        return None
    request_format, context_mode = profile.split("+", 1)
    return {
        "probe_id": probe_id,
        "profile": profile,
        "user_prompt_sha256": str(metadata.get("user_prompt_sha256") or ""),
        "developer_prompt_sha256": str(metadata.get("developer_prompt_sha256") or hashlib.sha256(b"").hexdigest()),
        "effort": str(metadata.get("effort") or "low"),
        "request_format": request_format,
        "context_mode": context_mode,
        "normalizer_hash": str(metadata.get("normalizer_hash") or ""),
    }


def _contract_failures(runtime_spec: dict[str, Any], probe_metadata: dict[str, Any]) -> list[str]:
    contracts = runtime_spec.get("contracts") or {}
    failures: list[str] = []
    for key in (runtime_spec.get("cells") or {}):
        expected = _expected_contract(str(key), probe_metadata)
        observed = contracts.get(key)
        if expected is None:
            failures.append(f"metadata_missing:{key}")
        elif observed is None:
            failures.append(f"runtime_contract_missing:{key}")
        elif canonical_json(observed) != canonical_json(expected):
            failures.append(f"runtime_contract_mismatch:{key}")
    return sorted(failures)


def _draw_counts(window: dict[str, Any], categories: list[str], n: int, rng: random.Random) -> dict[str, int]:
    source = {str(key): int(value) for key, value in (window.get("counts") or {}).items()}
    probabilities = []
    total = sum(source.values())
    for category in categories:
        probabilities.append(source.get(category, 0) / total if total > 0 else 1.0 / len(categories))
    sampled = sample_multinomial(n, probabilities, rng)
    return {category: sampled[index] for index, category in enumerate(categories) if sampled[index]}


def _calibration_metrics(records: list[dict[str, Any]], margin: float) -> dict[str, Any]:
    accepted = [record for record in records if record["margin"] >= margin]
    correct = [record for record in accepted if record["winner"] == record["true_model"]]
    errors = len(accepted) - len(correct)
    error_upper = wilson_interval(errors, len(accepted))[1] if accepted else 1.0
    coverage = len(correct) / len(records) if records else 0.0
    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_window[str(record["window"])].append(record)
    window_coverage = {
        window: sum(item["margin"] >= margin and item["winner"] == item["true_model"] for item in values) / len(values)
        for window, values in by_window.items()
    }
    window_error_rate: dict[str, float] = {}
    for window, values in by_window.items():
        accepted_window = [item for item in values if item["margin"] >= margin]
        window_error_rate[window] = (
            sum(item["winner"] != item["true_model"] for item in accepted_window) / len(accepted_window)
            if accepted_window else 1.0
        )
    return {
        "accepted": len(accepted),
        "correct": len(correct),
        "errors": errors,
        "error_rate": errors / len(accepted) if accepted else 1.0,
        "error_wilson95_upper": error_upper,
        "coverage_overall": coverage,
        "coverage_by_window": window_coverage,
        "coverage_worst_window": min(window_coverage.values()) if window_coverage else 0.0,
        "error_rate_by_window": window_error_rate,
        "error_rate_worst_window": max(window_error_rate.values()) if window_error_rate else 1.0,
    }


def _select_pass_margin(records: list[dict[str, Any]]) -> tuple[float | None, dict[str, Any]]:
    thresholds = sorted({1e-12, *(max(1e-12, float(record["margin"])) for record in records)})
    for threshold in thresholds:
        metrics = _calibration_metrics(records, threshold)
        if (
            metrics["error_wilson95_upper"] <= PASS_ERROR_UPPER_MAX
            and metrics["coverage_overall"] >= PASS_COVERAGE_MIN
            and metrics["coverage_worst_window"] >= PASS_WORST_WINDOW_COVERAGE_MIN
        ):
            return threshold, metrics
    return None, _calibration_metrics(records, float("inf"))


def _select_alert_margin(records: list[dict[str, Any]]) -> tuple[float | None, dict[str, Any]]:
    thresholds = sorted({1e-12, *(max(1e-12, float(record["margin"])) for record in records)})
    thresholds.append(math.nextafter(max(thresholds or [0.0]), math.inf))
    for threshold in thresholds:
        false_alerts = sum(record["winner"] != record["true_model"] and record["margin"] >= threshold for record in records)
        upper = wilson_interval(false_alerts, len(records))[1] if records else 1.0
        if upper <= ALERT_FALSE_RATE_UPPER_MAX:
            return threshold, {"false_alerts": false_alerts, "total": len(records), "wilson95_upper": upper}
    return None, {"false_alerts": len(records), "total": len(records), "wilson95_upper": 1.0}


def _select_mixture_threshold(records: list[dict[str, Any]]) -> tuple[float | None, dict[str, Any]]:
    candidates = sorted({0.0, *(float(record["mixture_gain"]) for record in records)})
    candidates.append(math.nextafter(max(candidates or [0.0]), math.inf))
    for threshold in candidates:
        false_alerts = sum(
            record["mixture_gain"] >= threshold and record["second_share"] >= SECOND_SHARE_MIN
            for record in records
        )
        upper = wilson_interval(false_alerts, len(records))[1] if records else 1.0
        if upper <= ALERT_FALSE_RATE_UPPER_MAX:
            return threshold, {"false_alerts": false_alerts, "total": len(records), "wilson95_upper": upper}
    return None, {"false_alerts": len(records), "total": len(records), "wilson95_upper": 1.0}


def _confusion(records: list[dict[str, Any]], margin: float) -> dict[str, dict[str, int]]:
    table = {model: {candidate: 0 for candidate in (*MODELS, "abstain")} for model in MODELS}
    for record in records:
        candidate = record["winner"] if record["margin"] >= margin else "abstain"
        table[record["true_model"]][candidate] += 1
    return table


def _fit_temperature(records: list[dict[str, Any]]) -> tuple[float, float]:
    best = (float("inf"), 1.0)
    for temperature in TEMPERATURE_GRID:
        loss = 0.0
        for record in records:
            probabilities = softmax({model: record["scores"][model] / temperature for model in MODELS})
            loss -= _safe_log(probabilities[record["true_model"]])
        loss /= max(1, len(records))
        if (loss, temperature) < best:
            best = (loss, temperature)
    return best[1], best[0]


def _tau_rank(candidate: dict[str, Any]) -> tuple[float, float, float]:
    metrics = candidate.get("pass_metrics") or {}
    return (
        float(metrics.get("error_rate_worst_window", 1.0)),
        -float(metrics.get("coverage_overall", 0.0)),
        float(candidate.get("tau", 0.0)),
    )


def _validate_raw_baseline(raw: dict[str, Any]) -> None:
    probes = raw.get("probes")
    if not isinstance(probes, dict) or not probes:
        raise ValueError("baseline probes are required")
    for probe_id, probe in probes.items():
        profiles = probe.get("profiles") or {}
        if not profiles:
            raise ValueError(f"baseline probe has no profiles: {probe_id}")
        for profile_name, profile in profiles.items():
            for model in MODELS:
                if len(_window_map(profile, model)) < MIN_WINDOWS:
                    raise ValueError(f"{probe_id}|{profile_name}|{model} requires at least {MIN_WINDOWS} windows")


def fit_baseline(
    raw: dict[str, Any],
    *,
    runtime_specs: list[dict[str, Any]],
    probe_metadata: dict[str, Any] | None = None,
    replay_draws_per_model_window: int = 50,
    seed: int = 0,
    baseline_id: str | None = None,
) -> dict[str, Any]:
    _validate_raw_baseline(raw)
    probe_metadata = deepcopy(probe_metadata or {})
    raw_cells = {
        f"{probe_id}|{profile_name}": profile
        for probe_id, probe in raw["probes"].items()
        for profile_name, profile in (probe.get("profiles") or {}).items()
    }
    cells_quality: dict[str, Any] = {}
    for key, profile in raw_cells.items():
        sample_counts = {
            model: {
                window_id: int(window.get("total", sum((window.get("counts") or {}).values())))
                for window_id, window in _window_map(profile, model).items()
            }
            for model in MODELS
        }
        valid_rates = {
            model: {
                window_id: int(window.get("valid", window.get("total", 0))) / max(1, int(window.get("total", 0)))
                for window_id, window in _window_map(profile, model).items()
            }
            for model in MODELS
        }
        cells_quality[key] = {
            "sample_counts": sample_counts,
            "valid_rates": valid_rates,
            "minimum_window_samples": min(value for model in sample_counts.values() for value in model.values()),
            "minimum_valid_rate": min(value for model in valid_rates.values() for value in model.values()),
            "window_count_by_model": {model: len(values) for model, values in sample_counts.items()},
        }

    calibrations: dict[str, Any] = {}
    selected_tau_by_signature: dict[str, float] = {}
    for runtime_spec in runtime_specs:
        signature = _runtime_signature(runtime_spec)
        required = {str(key): int(value) for key, value in (runtime_spec.get("cells") or {}).items()}
        missing = sorted(key for key in required if key not in raw_cells)
        quality_failures = _contract_failures(runtime_spec, probe_metadata) + [
            key for key, count in required.items()
            if key in cells_quality and (
                cells_quality[key]["minimum_window_samples"] < count
                or cells_quality[key]["minimum_valid_rate"] < MIN_VALID_RATE
                or min(cells_quality[key]["window_count_by_model"].values()) < MIN_WINDOWS
            )
        ]
        tau_candidates: list[dict[str, Any]] = []
        if not missing and not quality_failures:
            common_windows = sorted(set.intersection(*[
                set(_window_map(raw_cells[key], model))
                for key in required
                for model in MODELS
            ]))
            for tau in TAU_GRID:
                records: list[dict[str, Any]] = []
                cell_fit_scores: dict[str, list[float]] = defaultdict(list)
                rng = random.Random(seed ^ int(signature[:16], 16) ^ int(tau * 1_000_000))
                for window in common_windows:
                    fitted_cells = {key: fit_cell(raw_cells[key], tau, excluded_window=window) for key in required}
                    for true_model in MODELS:
                        for _draw in range(replay_draws_per_model_window):
                            candidate_counts = {
                                key: _draw_counts(
                                    _window_map(raw_cells[key], true_model)[window],
                                    fitted_cells[key]["categories"],
                                    required[key],
                                    rng,
                                )
                                for key in required
                            }
                            scored = _score_cells(fitted_cells, candidate_counts, required)
                            ordered = sorted(MODELS, key=lambda model: scored["scores"][model], reverse=True)
                            for key, cell in scored["cells"].items():
                                cell_fit_scores[key].append(max(cell["average_log_likelihood"].values()))
                            records.append({
                                "window": window,
                                "true_model": true_model,
                                "winner": ordered[0],
                                "margin": scored["scores"][ordered[0]] - scored["scores"][ordered[1]],
                                "scores": scored["scores"],
                                "mixture_gain": scored["mixture"]["mixture_gain"],
                                "second_share": scored["mixture"]["second_share"],
                            })
                pass_margin, pass_metrics = _select_pass_margin(records)
                alert_margin, alert_metrics = _select_alert_margin(records)
                mixture_threshold, mixture_metrics = _select_mixture_threshold(records)
                temperature, log_loss = _fit_temperature(records)
                formal = pass_margin is not None and alert_margin is not None and mixture_threshold is not None
                tau_candidates.append({
                    "tau": tau,
                    "formal_eligible": formal,
                    "pass_margin": pass_margin,
                    "alert_margin": alert_margin,
                    "mixture_gain_threshold": mixture_threshold,
                    "pass_metrics": pass_metrics,
                    "alert_metrics": alert_metrics,
                    "mixture_metrics": mixture_metrics,
                    "temperature": temperature,
                    "multiclass_log_loss": log_loss,
                    "records": records,
                    "ood_thresholds": {key: quantile(values, 0.01) for key, values in cell_fit_scores.items()},
                })
        eligible_tau = [candidate for candidate in tau_candidates if candidate["formal_eligible"]]
        if eligible_tau:
            chosen = min(
                eligible_tau,
                key=_tau_rank,
            )
        elif tau_candidates:
            chosen = min(
                tau_candidates,
                key=_tau_rank,
            )
        else:
            chosen = {
                "tau": 0.0,
                "formal_eligible": False,
                "pass_margin": None,
                "alert_margin": None,
                "mixture_gain_threshold": None,
                "pass_metrics": {},
                "alert_metrics": {},
                "mixture_metrics": {},
                "temperature": 1.0,
                "multiclass_log_loss": None,
                "records": [],
                "ood_thresholds": {},
            }
        selected_tau_by_signature[signature] = float(chosen["tau"])
        records = chosen.pop("records")
        chosen["confusion_matrix"] = _confusion(records, float(chosen["pass_margin"] or float("inf")))
        chosen["replay_count"] = len(records)
        chosen["runtime_name"] = str(runtime_spec.get("name") or "runtime")
        chosen["runtime_signature"] = signature
        chosen["required_samples"] = required
        chosen["exact_contracts"] = deepcopy(runtime_spec.get("contracts") or {})
        chosen["missing_cells"] = missing
        chosen["quality_failures"] = quality_failures
        chosen["formal_eligible"] = bool(chosen["formal_eligible"] and not missing and not quality_failures)
        calibrations[signature] = chosen

    tau_values = sorted(set(selected_tau_by_signature.values()))
    default_tau = tau_values[0] if len(tau_values) == 1 else 0.02
    fitted_cells = {key: fit_cell(profile, default_tau) for key, profile in raw_cells.items()}
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "scoring_version": SCORING_VERSION,
        "baseline_id": str(baseline_id or raw.get("baseline_id") or "gpt56-probability-baseline"),
        "smoothing_alpha": SMOOTHING,
        "tau_grid": list(TAU_GRID),
        "default_tau": default_tau,
        "models": list(MODELS),
        "probe_metadata": probe_metadata,
        "raw_counts": deepcopy(raw.get("probes") or {}),
        "cells_quality": cells_quality,
        "cells": fitted_cells,
        "calibrations": calibrations,
        "formal_eligible": any(value.get("formal_eligible") for value in calibrations.values()),
        "threshold_policy": {
            "min_windows": MIN_WINDOWS,
            "min_valid_rate": MIN_VALID_RATE,
            "pass_error_wilson95_upper_max": PASS_ERROR_UPPER_MAX,
            "pass_coverage_overall_min": PASS_COVERAGE_MIN,
            "pass_coverage_worst_window_min": PASS_WORST_WINDOW_COVERAGE_MIN,
            "alert_false_rate_wilson95_upper_max": ALERT_FALSE_RATE_UPPER_MAX,
            "second_share_min": SECOND_SHARE_MIN,
        },
        "resampling_policy": "one_multinomial_draw_from_each_heldout_empirical_window;no_dirichlet_candidate_or_baseline_resampling",
    }
    artifact["content_sha256"] = hashlib.sha256(canonical_json(artifact).encode("utf-8")).hexdigest()
    return artifact


def verify_baseline(artifact: dict[str, Any]) -> None:
    if int(artifact.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError("unsupported probability baseline schema")
    if artifact.get("scoring_version") != SCORING_VERSION:
        raise ValueError("unsupported scoring version")
    expected = str(artifact.get("content_sha256") or "")
    body = deepcopy(artifact)
    body.pop("content_sha256", None)
    observed = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    if expected != observed:
        raise ValueError("probability baseline content hash mismatch")


def load_baseline(path: str | Path) -> dict[str, Any]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_baseline(artifact)
    return artifact


class ProbabilityModel:
    def __init__(self, artifact: dict[str, Any]):
        verify_baseline(artifact)
        self.artifact = artifact

    def score(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        runtime_spec: dict[str, Any],
        claimed_model: str,
    ) -> dict[str, Any]:
        signature = _runtime_signature(runtime_spec)
        calibration = self.artifact.get("calibrations", {}).get(signature)
        required = {str(key): int(value) for key, value in (runtime_spec.get("cells") or {}).items()}
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        actual: dict[str, int] = Counter()
        for row in rows:
            if row.get("status") != "ok" or row.get("classification") != "category":
                continue
            key = f"{row.get('probe_id')}|{row.get('request_format')}+{row.get('context_mode')}"
            if key not in required:
                continue
            category = str(row.get("category") or "__INVALID_OUTPUT__")
            counts[key][category] += 1
            actual[key] += 1
        tau = float(calibration.get("tau", self.artifact.get("default_tau", 0.02))) if calibration else float(self.artifact.get("default_tau", 0.02))
        fitted_cells = {
            key: fit_cell(
                self.artifact["raw_counts"][key.split("|", 1)[0]]["profiles"][key.split("|", 1)[1]],
                tau,
            )
            for key in required
            if key.split("|", 1)[0] in self.artifact.get("raw_counts", {})
            and key.split("|", 1)[1] in self.artifact["raw_counts"][key.split("|", 1)[0]].get("profiles", {})
        }
        missing_baseline_cells = sorted(set(required) - set(fitted_cells))
        scored = _score_cells(
            fitted_cells,
            {key: dict(value) for key, value in counts.items()},
            required,
            calibration.get("ood_thresholds", {}) if calibration else {},
        )
        temperature = float(calibration.get("temperature", 1.0)) if calibration else 1.0
        probabilities = softmax({model: scored["scores"][model] / temperature for model in MODELS})
        ordered = sorted(MODELS, key=lambda model: scored["scores"][model], reverse=True)
        margin = scored["scores"][ordered[0]] - scored["scores"][ordered[1]]
        reasons: list[str] = []
        contract_failures = _contract_failures(runtime_spec, self.artifact.get("probe_metadata") or {})
        if calibration is None:
            reasons.append("no_exact_runtime_calibration")
        elif not calibration.get("formal_eligible"):
            reasons.append("baseline_calibration_gate_failed")
        reasons.extend(contract_failures)
        if missing_baseline_cells:
            reasons.append("baseline_cells_missing:" + ",".join(missing_baseline_cells))
        if scored["missing_cells"]:
            reasons.append("candidate_samples_incomplete:" + ",".join(scored["missing_cells"]))
        if scored["non_ood_family_count"] == 0:
            reasons.append("all_formal_families_ood")
        formal_ready = not reasons
        pass_margin = float(calibration["pass_margin"]) if calibration and calibration.get("pass_margin") is not None else float("inf")
        alert_margin = float(calibration["alert_margin"]) if calibration and calibration.get("alert_margin") is not None else float("inf")
        mixture_threshold = float(calibration["mixture_gain_threshold"]) if calibration and calibration.get("mixture_gain_threshold") is not None else float("inf")
        mixture_alert = bool(
            formal_ready
            and scored["mixture"]["mixture_gain"] >= mixture_threshold
            and scored["mixture"]["second_share"] >= SECOND_SHARE_MIN
        )
        pure_alert = bool(formal_ready and ordered[0] != claimed_model and margin >= alert_margin)
        probability_pass = bool(
            formal_ready and ordered[0] == claimed_model and margin >= pass_margin and not mixture_alert
        )
        if formal_ready and not (probability_pass or pure_alert or mixture_alert):
            reasons.append("calibrated_margin_or_mixture_threshold_not_reached")
        return {
            "schema_version": SCHEMA_VERSION,
            "scoring_version": SCORING_VERSION,
            "baseline_id": self.artifact["baseline_id"],
            "baseline_sha256": self.artifact["content_sha256"],
            "runtime_signature": signature,
            "runtime_name": str(runtime_spec.get("name") or "runtime"),
            "formal_eligible": formal_ready,
            "probability_pass": probability_pass,
            "pure_model_alert": pure_alert,
            "mixture_alert": mixture_alert,
            "evidence_insufficient": not (probability_pass or pure_alert or mixture_alert),
            "evidence_insufficient_reasons": reasons,
            "winner": ordered[0],
            "runner_up": ordered[1],
            "score_margin": margin,
            "pass_margin": pass_margin if math.isfinite(pass_margin) else None,
            "alert_margin": alert_margin if math.isfinite(alert_margin) else None,
            "conditional_relative_probability": probabilities,
            "temperature": temperature,
            "pure_scores": scored["scores"],
            "mixture": {
                **scored["mixture"],
                "mixture_gain_threshold": mixture_threshold if math.isfinite(mixture_threshold) else None,
                "second_share_threshold": SECOND_SHARE_MIN,
            },
            "family_contributions": scored["families"],
            "cell_details": scored["cells"],
            "actual_samples": dict(actual),
            "required_samples": required,
            "exact_contracts": deepcopy(runtime_spec.get("contracts") or {}),
            "contract_failures": contract_failures,
            "calibration": {
                key: calibration.get(key)
                for key in (
                    "pass_metrics", "alert_metrics", "mixture_metrics", "confusion_matrix",
                    "replay_count", "quality_failures", "missing_cells",
                )
            } if calibration else None,
            "probability_name_cn": "在已导入Sol/Terra/Luna可信基线下的条件相对匹配概率",
        }


__all__ = [
    "MODELS",
    "ProbabilityModel",
    "SCHEMA_VERSION",
    "SCORING_VERSION",
    "fit_baseline",
    "fit_cell",
    "js_divergence",
    "load_baseline",
    "verify_baseline",
]
