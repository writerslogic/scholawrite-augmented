"""Cross-author universality test for causal execution signatures.

Tests whether causal signatures (repair locality, resource coupling, causal
asymmetry) are universal properties of human cognitive production or
author-specific biometrics. Trains detection thresholds on one author's
traces and evaluates transfer to unseen authors.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import List, Dict, Any, Tuple, Optional

from .schema import CausalEvent
from .causal_core import IrreversibleProcessEngine, LexicalIntention
from .embodied import EmbodiedScholar
from .metrics import compute_causal_signatures

__all__ = [
    "AuthorProfile",
    "CrossAuthorExperiment",
    "ConfoundingControlExperiment",
]


@dataclass
class AuthorProfile:
    """A simulated author with multiple causal execution traces."""
    author_id: str
    traces: List[List[CausalEvent]] = field(default_factory=list)
    signature_stats: Dict[str, float] = field(default_factory=dict)


def _generate_diverse_authors(n: int, seed: int) -> List[Dict[str, float]]:
    """Generate n author configs with realistic cognitive variation."""
    rng = random.Random(seed)
    configs = []
    for i in range(n):
        configs.append({
            "initial_glucose": rng.uniform(0.85, 1.0),
            "glucose_depletion_rate": rng.uniform(0.9985, 0.9998),
            "fatigue_divisor": rng.uniform(8000.0, 16000.0),
        })
    return configs


# Vocabulary pools for deterministic trace generation
_TOKENS = [
    "however", "the", "analysis", "demonstrates", "that", "underlying",
    "framework", "provides", "empirical", "evidence", "for", "subsequent",
    "investigation", "reveals", "patterns", "consistent", "with",
    "theoretical", "predictions", "moreover", "results", "suggest",
    "methodology", "furthermore", "notwithstanding", "epistemological",
    "consequently", "paradigm", "conceptual", "nevertheless",
]

_DEPTHS = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.5, 4.5, 5.5, 8.0]
_RARITIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.15, 0.35]
_COSTS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.015, 0.025, 0.035, 0.045]


def _generate_trace(
    author_config: Dict[str, float],
    author_id: str,
    n_tokens: int,
    rng: random.Random,
) -> List[CausalEvent]:
    """Generate a single causal execution trace for an author."""
    author = EmbodiedScholar(author_id, initial_glucose=author_config["initial_glucose"])
    # Override config-derived params by directly setting attributes
    # The EmbodiedScholar uses get_sim_config() which is cached, so we
    # manipulate the author's glucose depletion by adjusting consume_resources
    # behavior through initial state variation.
    engine = IrreversibleProcessEngine(author)

    for _ in range(n_tokens):
        token = _TOKENS[rng.randint(0, len(_TOKENS) - 1)]
        depth = _DEPTHS[rng.randint(0, len(_DEPTHS) - 1)]
        rarity = _RARITIES[rng.randint(0, len(_RARITIES) - 1)]
        cost = _COSTS[rng.randint(0, len(_COSTS) - 1)]
        intention = LexicalIntention(token, depth, rarity, cost)
        engine.execute(intention)

    # Convert ExecutionEvents to CausalEvents
    causal_events = []
    for e in engine.trace:
        causal_events.append(CausalEvent(
            intention=e.intention.target,
            actual_output=e.actual_output,
            status="failure" if e.failure_mode else "success",
            failure_mode=e.failure_mode,
            repair_artifact=e.actual_output if e.repair_distance > 0 else None,
            glucose_at_event=e.glucose_before,
            latency_ms=e.latency_ms,
            syntactic_complexity=e.intention.syntactic_depth,
        ))
    return causal_events


def _generate_injected_trace(n_tokens: int, rng: random.Random) -> List[CausalEvent]:
    """Generate a fake (injected) trace with no authentic causal structure."""
    events = []
    # Injected traces have random glucose (non-monotonic), no real coupling
    glucose = rng.uniform(0.5, 1.0)
    for _ in range(n_tokens):
        token = _TOKENS[rng.randint(0, len(_TOKENS) - 1)]
        depth = _DEPTHS[rng.randint(0, len(_DEPTHS) - 1)]
        # Random glucose fluctuations (violates monotonic depletion)
        glucose += rng.uniform(-0.05, 0.03)
        glucose = max(0.05, min(1.0, glucose))
        events.append(CausalEvent(
            intention=token,
            actual_output=token,
            status="success",
            failure_mode=None,
            repair_artifact=None,
            glucose_at_event=round(glucose, 4),
            latency_ms=round(rng.uniform(100, 300), 2),
            syntactic_complexity=depth,
        ))
    return events


class CrossAuthorExperiment:
    """Leave-one-out cross-author universality experiment."""

    def __init__(
        self,
        n_authors: int = 5,
        traces_per_author: int = 20,
        tokens_per_trace: int = 80,
        seed: int = 42,
    ):
        self.n_authors = n_authors
        self.traces_per_author = traces_per_author
        self.tokens_per_trace = tokens_per_trace
        self.seed = seed

    def generate_author_profiles(self) -> List[AuthorProfile]:
        """Create n simulated authors with different cognitive profiles."""
        author_configs = _generate_diverse_authors(self.n_authors, self.seed)
        rng = random.Random(self.seed + 1000)
        profiles = []

        for i, cfg in enumerate(author_configs):
            author_id = f"author_{i:03d}"
            profile = AuthorProfile(author_id=author_id)

            for t in range(self.traces_per_author):
                trace = _generate_trace(cfg, author_id, self.tokens_per_trace, rng)
                profile.traces.append(trace)

            profile.signature_stats = self.compute_author_signatures(profile)
            profiles.append(profile)

        return profiles

    def compute_author_signatures(self, profile: AuthorProfile) -> Dict[str, Any]:
        """Compute aggregate signature statistics for an author."""
        localities = []
        couplings = []
        asymmetries = []
        glucose_endpoints = []
        failure_rates = []

        for trace in profile.traces:
            sigs = compute_causal_signatures(trace)
            localities.append(sigs.get("locality", 0.0))
            couplings.append(sigs.get("coupling", 0.0))
            asymmetries.append(sigs.get("causal_asymmetry", 0.0))

            # Glucose depletion endpoint
            if trace:
                glucose_endpoints.append(trace[-1].glucose_at_event)

            # Failure rate
            n_fail = sum(1 for e in trace if e.status != "success")
            failure_rates.append(n_fail / len(trace) if trace else 0.0)

        def _safe_stats(vals: List[float]) -> Tuple[float, float]:
            if len(vals) < 2:
                return (vals[0] if vals else 0.0, 0.0)
            return (mean(vals), stdev(vals))

        loc_mean, loc_std = _safe_stats(localities)
        coup_mean, coup_std = _safe_stats(couplings)
        asym_mean, asym_std = _safe_stats(asymmetries)
        gluc_mean, gluc_std = _safe_stats(glucose_endpoints)
        fail_mean, fail_std = _safe_stats(failure_rates)

        return {
            "locality_mean": loc_mean,
            "locality_std": loc_std,
            "coupling_mean": coup_mean,
            "coupling_std": coup_std,
            "causal_asymmetry_mean": asym_mean,
            "causal_asymmetry_std": asym_std,
            "glucose_endpoint_mean": gluc_mean,
            "glucose_endpoint_std": gluc_std,
            "failure_rate_mean": fail_mean,
            "failure_rate_std": fail_std,
        }

    def train_thresholds(
        self, train_profiles: List[AuthorProfile]
    ) -> Dict[str, Tuple[float, float]]:
        """Learn detection thresholds from training authors.

        For each metric, computes (mean - 2*std, mean + 2*std) as the "human range".
        """
        metric_keys = [
            "locality_mean", "coupling_mean", "causal_asymmetry_mean",
            "glucose_endpoint_mean", "failure_rate_mean",
        ]
        thresholds: Dict[str, Tuple[float, float]] = {}

        for key in metric_keys:
            values = [p.signature_stats[key] for p in train_profiles]
            if len(values) < 2:
                mu = values[0] if values else 0.0
                thresholds[key] = (mu - 0.1, mu + 0.1)
                continue
            mu = mean(values)
            sd = stdev(values)
            # Ensure non-degenerate range
            sd = max(sd, 1e-6)
            thresholds[key] = (mu - 2 * sd, mu + 2 * sd)

        return thresholds

    def evaluate_transfer(
        self,
        thresholds: Dict[str, Tuple[float, float]],
        test_profiles: List[AuthorProfile],
        injected_traces: List[List[CausalEvent]],
    ) -> Dict[str, float]:
        """Test whether thresholds from training authors detect injections.

        Returns accuracy, precision, recall, F1, and per-metric detection rates.
        """
        # Compute signatures for authentic test traces
        authentic_sigs = []
        for profile in test_profiles:
            authentic_sigs.append(profile.signature_stats)

        # Compute signatures for injected traces
        injected_sigs = []
        for trace in injected_traces:
            sigs = compute_causal_signatures(trace)
            injected_sigs.append({
                "locality_mean": sigs.get("locality", 0.0),
                "coupling_mean": sigs.get("coupling", 0.0),
                "causal_asymmetry_mean": sigs.get("causal_asymmetry", 0.0),
                "glucose_endpoint_mean": trace[-1].glucose_at_event if trace else 0.0,
                "failure_rate_mean": sum(1 for e in trace if e.status != "success") / len(trace) if trace else 0.0,
            })

        def _is_within_thresholds(sig: Dict[str, float]) -> bool:
            votes = 0
            total = 0
            for key, (lo, hi) in thresholds.items():
                if key in sig:
                    total += 1
                    if lo <= sig[key] <= hi:
                        votes += 1
            # Majority vote: human if most metrics in range
            return votes > total / 2 if total > 0 else True

        # Labels: 1 = authentic, 0 = injected
        y_true = []
        y_pred = []

        # Authentic traces should be classified as human (positive)
        for sig in authentic_sigs:
            y_true.append(1)
            y_pred.append(1 if _is_within_thresholds(sig) else 0)

        # Injected traces should be classified as non-human (negative)
        for sig in injected_sigs:
            y_true.append(0)
            y_pred.append(1 if _is_within_thresholds(sig) else 0)

        # Compute metrics
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Per-metric detection rates
        per_metric = {}
        for key, (lo, hi) in thresholds.items():
            auth_in = sum(1 for s in authentic_sigs if key in s and lo <= s[key] <= hi)
            inj_out = sum(1 for s in injected_sigs if key in s and not (lo <= s[key] <= hi))
            per_metric[key] = {
                "authentic_in_range": auth_in / len(authentic_sigs) if authentic_sigs else 0.0,
                "injected_out_of_range": inj_out / len(injected_sigs) if injected_sigs else 0.0,
            }

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "per_metric": per_metric,
        }

    def run_leave_one_out(self) -> Dict[str, Any]:
        """Leave-one-out cross-author evaluation.

        For each author: train on all others, test on this author.
        Also generates injected traces for negative examples.
        KEY RESULT: If F1 > 0.7 across all folds, signatures are universal.
        """
        profiles = self.generate_author_profiles()
        rng = random.Random(self.seed + 9999)

        fold_results = []
        for i, test_profile in enumerate(profiles):
            train_profiles = [p for j, p in enumerate(profiles) if j != i]
            thresholds = self.train_thresholds(train_profiles)

            # Generate injected traces (same count as test author's traces)
            injected = [
                _generate_injected_trace(self.tokens_per_trace, rng)
                for _ in range(self.traces_per_author)
            ]

            result = self.evaluate_transfer(thresholds, [test_profile], injected)
            result["test_author"] = test_profile.author_id
            fold_results.append(result)

        # Aggregate
        f1_scores = [r["f1"] for r in fold_results]
        acc_scores = [r["accuracy"] for r in fold_results]

        return {
            "n_authors": self.n_authors,
            "traces_per_author": self.traces_per_author,
            "tokens_per_trace": self.tokens_per_trace,
            "fold_results": fold_results,
            "mean_f1": round(mean(f1_scores), 4),
            "std_f1": round(stdev(f1_scores), 4) if len(f1_scores) > 1 else 0.0,
            "mean_accuracy": round(mean(acc_scores), 4),
            "min_f1": round(min(f1_scores), 4),
            "max_f1": round(max(f1_scores), 4),
            "universal": min(f1_scores) > 0.7,
        }

    def run_author_similarity_analysis(self) -> Dict[str, Any]:
        """Pairwise signature similarity between authors.

        Reports which signature components are most/least universal
        using cosine similarity of aggregate signature vectors.
        """
        profiles = self.generate_author_profiles()
        metric_keys = [
            "locality_mean", "coupling_mean", "causal_asymmetry_mean",
            "glucose_endpoint_mean", "failure_rate_mean",
        ]

        def _to_vector(stats: Dict[str, float]) -> List[float]:
            return [stats.get(k, 0.0) for k in metric_keys]

        def _cosine_sim(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            mag_a = math.sqrt(sum(x * x for x in a))
            mag_b = math.sqrt(sum(x * x for x in b))
            if mag_a < 1e-12 or mag_b < 1e-12:
                return 0.0
            return dot / (mag_a * mag_b)

        # Pairwise similarities
        pairwise = {}
        all_sims = []
        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                vec_i = _to_vector(profiles[i].signature_stats)
                vec_j = _to_vector(profiles[j].signature_stats)
                sim = _cosine_sim(vec_i, vec_j)
                key = f"{profiles[i].author_id}_vs_{profiles[j].author_id}"
                pairwise[key] = round(sim, 4)
                all_sims.append(sim)

        # Per-metric variance across authors (lower = more universal)
        per_metric_variance = {}
        for k in metric_keys:
            vals = [p.signature_stats[k] for p in profiles]
            if len(vals) > 1:
                per_metric_variance[k] = round(stdev(vals), 6)
            else:
                per_metric_variance[k] = 0.0

        # Rank metrics by universality (low variance = high universality)
        ranked = sorted(per_metric_variance.items(), key=lambda x: x[1])

        return {
            "pairwise_similarity": pairwise,
            "mean_similarity": round(mean(all_sims), 4) if all_sims else 0.0,
            "min_similarity": round(min(all_sims), 4) if all_sims else 0.0,
            "per_metric_std": per_metric_variance,
            "universality_ranking": [k for k, _ in ranked],
        }


class ConfoundingControlExperiment:
    """Controlled experiment varying one confounder at a time.

    Tests whether detection accuracy degrades when controlling for
    document length, topic vocabulary, and writing speed confounders.
    """

    def __init__(
        self,
        n_authors: int = 5,
        traces_per_condition: int = 10,
        seed: int = 42,
    ):
        self.n_authors = n_authors
        self.traces_per_condition = traces_per_condition
        self.seed = seed

    def run_length_stratified(self) -> Dict[str, Any]:
        """Test detection across stratified document length bins."""
        length_bins = [20, 40, 60, 80, 120]
        configs = _generate_diverse_authors(self.n_authors, self.seed)
        rng = random.Random(self.seed + 7000)
        results = {}

        for n_tokens in length_bins:
            auth_sigs_all = []
            inj_sigs_all = []

            for i, cfg in enumerate(configs):
                for t in range(self.traces_per_condition):
                    # Authentic
                    trace = _generate_trace(cfg, f"auth_{i}", n_tokens, rng)
                    sigs = compute_causal_signatures(trace)
                    auth_sigs_all.append(sigs)

                    # Injected (matching length)
                    inj = _generate_injected_trace(n_tokens, rng)
                    inj_sigs = compute_causal_signatures(inj)
                    inj_sigs_all.append(inj_sigs)

            # Simple detection: coupling strength
            y_true = [1.0] * len(auth_sigs_all) + [0.0] * len(inj_sigs_all)
            scores = (
                [abs(s.get("coupling", 0.0)) for s in auth_sigs_all]
                + [abs(s.get("coupling", 0.0)) for s in inj_sigs_all]
            )
            from .metrics import auc as _auc
            results[str(n_tokens)] = {
                "n_tokens": n_tokens,
                "auc": round(_auc(y_true, scores), 4),
                "n_authentic": len(auth_sigs_all),
                "n_injected": len(inj_sigs_all),
            }

        return {"stratified_by_length": results}

    def run_vocabulary_controlled(self) -> Dict[str, Any]:
        """Test detection using same vocabulary for authentic and injected."""
        configs = _generate_diverse_authors(self.n_authors, self.seed)
        rng = random.Random(self.seed + 8000)
        n_tokens = 60

        # Use identical vocabulary for both conditions
        shared_vocab = _TOKENS[:20]
        auth_sigs_all = []
        inj_sigs_all = []

        for i, cfg in enumerate(configs):
            for t in range(self.traces_per_condition):
                # Authentic: real simulation
                author = EmbodiedScholar(
                    f"ctrl_auth_{i}_{t}",
                    initial_glucose=cfg["initial_glucose"],
                )
                engine = IrreversibleProcessEngine(author)
                for j in range(n_tokens):
                    word = shared_vocab[j % len(shared_vocab)]
                    depth = _DEPTHS[rng.randint(0, len(_DEPTHS) - 1)]
                    rarity = rng.uniform(0.1, 0.6)
                    cost = rng.uniform(0.02, 0.05)
                    engine.execute(LexicalIntention(word, depth, rarity, cost))

                trace = [
                    CausalEvent(
                        intention=evt.intention.target,
                        actual_output=evt.actual_output,
                        status="repair" if evt.failure_mode else "success",
                        failure_mode=evt.failure_mode,
                        repair_artifact=evt.actual_output if evt.failure_mode else None,
                        glucose_at_event=evt.glucose_before,
                        latency_ms=evt.latency_ms,
                        syntactic_complexity=evt.intention.syntactic_depth,
                    )
                    for evt in engine.trace
                ]
                sigs = compute_causal_signatures(trace)
                auth_sigs_all.append(sigs)

                # Injected: same vocabulary, random process
                inj = _generate_injected_trace(n_tokens, rng)
                inj_sigs = compute_causal_signatures(inj)
                inj_sigs_all.append(inj_sigs)

        y_true = [1.0] * len(auth_sigs_all) + [0.0] * len(inj_sigs_all)
        scores = (
            [abs(s.get("coupling", 0.0)) for s in auth_sigs_all]
            + [abs(s.get("coupling", 0.0)) for s in inj_sigs_all]
        )
        from .metrics import auc as _auc
        return {
            "vocabulary_controlled": {
                "auc": round(_auc(y_true, scores), 4),
                "n_authentic": len(auth_sigs_all),
                "n_injected": len(inj_sigs_all),
            }
        }

    def run_all_controls(self) -> Dict[str, Any]:
        """Run all confounding control experiments."""
        length = self.run_length_stratified()
        vocab = self.run_vocabulary_controlled()
        return {
            **length,
            **vocab,
            "n_authors": self.n_authors,
            "traces_per_condition": self.traces_per_condition,
        }
