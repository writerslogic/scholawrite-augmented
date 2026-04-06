"""Optimization-based adversary for the impossibility boundary.

Instead of handcrafting forgery strategies, the GradientForger parameterizes
trace generation as a continuous optimization problem and uses numerical
gradient descent to minimize detection score. This is the strongest possible
empirical evidence for the impossibility theorem:

- If convergence is fast → forgery is computationally cheap, supporting impossibility.
- If convergence is slow → characterizes computational cost, connecting to VDF analogy.

The forger optimizes:
    L(params) = Σ_i w_i * (signal_i(trace(params)) - target_i)²
              + λ_mono * monotonicity_penalty(glucose)
              + λ_loc * locality_penalty(repairs)

where params = (glucose_deltas, failure_probs, complexity_values, latency_values).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

from .schema import CausalEvent
from .consciousness_signatures import compute_consciousness_signatures

__all__ = ["GradientForger", "ForgeryConvergenceResult"]


_FAILURE_MODES = ["lexical_starvation", "syntactic_collapse"]
_REPAIR_WORDS = ["framework", "concept", "element", "thus,", "so,"]
_ACADEMIC_WORDS = [
    "The", "methodology", "establishes", "a", "robust", "baseline",
    "for", "comparative", "analysis", "of", "emerging", "patterns",
    "in", "empirical", "research", "paradigm", "framework", "this",
    "study", "demonstrates", "that", "underlying", "assumptions",
    "remain", "foundational", "to", "subsequent", "investigations",
]


@dataclass
class ForgeryConvergenceResult:
    """Records how the optimizer converges toward human-like signal values."""
    iterations: int
    final_loss: float
    initial_loss: float
    loss_history: List[float] = field(default_factory=list)
    signal_history: List[Dict[str, float]] = field(default_factory=list)
    final_signals: Dict[str, float] = field(default_factory=dict)
    target_signals: Dict[str, float] = field(default_factory=dict)
    final_composite_auc: Optional[float] = None
    converged: bool = False
    convergence_iteration: Optional[int] = None


@dataclass
class _TraceParams:
    """Continuous parameters for trace generation."""
    glucose_deltas: List[float]   # per-token glucose depletion amount
    failure_probs: List[float]    # per-token probability of failure
    complexity: List[float]       # per-token syntactic complexity
    latency_base: List[float]     # per-token base latency


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ez = math.exp(x)
    return ez / (1.0 + ez)


def _params_to_trace(params: _TraceParams, seed: int = 42) -> List[CausalEvent]:
    """Convert continuous parameters to a CausalEvent trace."""
    rng = random.Random(seed)
    n = len(params.glucose_deltas)
    words = (_ACADEMIC_WORDS * ((n // len(_ACADEMIC_WORDS)) + 1))[:n]

    events: List[CausalEvent] = []
    glucose = 1.0

    for i in range(n):
        # Glucose decreases by delta (clamped positive for monotonicity)
        delta = max(0.0, params.glucose_deltas[i])
        glucose = max(0.05, glucose - delta)

        # Failure gated by sigmoid of failure_prob parameter
        fail_prob = _sigmoid(params.failure_probs[i])
        is_failure = rng.random() < fail_prob

        complexity = max(0.5, params.complexity[i])
        # Latency coupled to glucose (as in embodied model)
        latency = max(80.0, params.latency_base[i] * (1.12 - glucose) + 115.0)

        if is_failure:
            fm = rng.choice(_FAILURE_MODES)
            repair_word = rng.choice(_REPAIR_WORDS)
            events.append(CausalEvent(
                intention=words[i],
                actual_output=repair_word,
                status="repair",
                failure_mode=fm,
                repair_artifact=repair_word,
                glucose_at_event=round(glucose, 6),
                latency_ms=round(latency, 2),
                syntactic_complexity=round(complexity, 1),
            ))
        else:
            events.append(CausalEvent(
                intention=words[i],
                actual_output=words[i],
                status="success",
                failure_mode=None,
                repair_artifact=None,
                glucose_at_event=round(glucose, 6),
                latency_ms=round(latency, 2),
                syntactic_complexity=round(complexity, 1),
            ))

    return events


def _extract_signals(trace: List[CausalEvent]) -> Dict[str, float]:
    """Extract the 6 normalized consciousness signals."""
    if len(trace) < 4:
        return {k: 0.0 for k in [
            "causal_dag", "causal_concentration", "cross_channel_mi",
            "decay_type", "free_energy", "adaptation",
        ]}
    result = compute_consciousness_signatures(trace)
    return dict(result.signal_breakdown.get("normalized", {}))


def _compute_loss(
    signals: Dict[str, float],
    targets: Dict[str, float],
    weights: Dict[str, float],
    trace: List[CausalEvent],
    lambda_mono: float = 5.0,
    lambda_range: float = 10.0,
) -> float:
    """Composite loss: signal deviation + monotonicity penalty + range constraints.

    Range constraints prevent trivial convergence by penalizing signals
    that collapse to near-zero instead of matching the target distribution.
    Each signal must stay within [target * 0.7, target * 1.3] to
    avoid the optimizer finding degenerate minima.
    """
    # Signal deviation
    signal_loss = sum(
        weights.get(k, 1.0) * (signals.get(k, 0.0) - targets.get(k, 0.0)) ** 2
        for k in targets
    )

    # Range constraint penalty: each signal must be within [target*0.7, target*1.3]
    range_penalty = 0.0
    for k, target_val in targets.items():
        sig_val = signals.get(k, 0.0)
        lo = target_val * 0.7
        hi = target_val * 1.3
        if target_val > 0.05:  # only enforce for non-trivial targets
            if sig_val < lo:
                range_penalty += (lo - sig_val) ** 2
            elif sig_val > hi:
                range_penalty += (sig_val - hi) ** 2

    # Monotonicity penalty: penalize glucose increases
    glucoses = [e.glucose_at_event for e in trace]
    mono_violations = sum(
        max(0.0, glucoses[i + 1] - glucoses[i])
        for i in range(len(glucoses) - 1)
    )
    mono_penalty = lambda_mono * mono_violations

    return signal_loss + mono_penalty + lambda_range * range_penalty


class GradientForger:
    """Optimization-based adversary using numerical gradient descent.

    Parameterizes trace generation and optimizes to match human signal
    distribution. Uses finite differences for gradient estimation.
    """

    def __init__(
        self,
        n_events: int = 30,
        seed: int = 42,
        learning_rate: float = 0.01,
        epsilon: float = 1e-4,
        lambda_mono: float = 5.0,
    ):
        self.n_events = n_events
        self.seed = seed
        self.lr = learning_rate
        self.epsilon = epsilon
        self.lambda_mono = lambda_mono

    def _init_params(self) -> _TraceParams:
        """Initialize parameters near human-like values."""
        rng = random.Random(self.seed)
        n = self.n_events
        return _TraceParams(
            glucose_deltas=[rng.uniform(0.005, 0.015) for _ in range(n)],
            failure_probs=[rng.uniform(-1.0, 0.0) for _ in range(n)],  # ~27% failure rate
            complexity=[rng.uniform(2.0, 6.0) for _ in range(n)],
            latency_base=[rng.uniform(60.0, 120.0) for _ in range(n)],
        )

    def _params_to_flat(self, p: _TraceParams) -> List[float]:
        return p.glucose_deltas + p.failure_probs + p.complexity + p.latency_base

    def _flat_to_params(self, flat: List[float]) -> _TraceParams:
        n = self.n_events
        return _TraceParams(
            glucose_deltas=flat[0:n],
            failure_probs=flat[n:2*n],
            complexity=flat[2*n:3*n],
            latency_base=flat[3*n:4*n],
        )

    def compute_human_targets(
        self,
        n_traces: int = 20,
        seed_offset: int = 0,
    ) -> Dict[str, float]:
        """Compute mean signal values from authentic traces."""
        from .adversarial import _generate_authentic_trace

        all_signals: Dict[str, List[float]] = {}
        for i in range(n_traces):
            trace = _generate_authentic_trace(
                seed=self.seed + seed_offset + i,
                n_events=self.n_events,
            )
            signals = _extract_signals(trace)
            for k, v in signals.items():
                all_signals.setdefault(k, []).append(v)

        return {k: mean(v) for k, v in all_signals.items()}

    def forge(
        self,
        targets: Optional[Dict[str, float]] = None,
        max_iterations: int = 500,
        convergence_threshold: float = 0.01,
        signal_weights: Optional[Dict[str, float]] = None,
        record_every: int = 10,
    ) -> Tuple[List[CausalEvent], ForgeryConvergenceResult]:
        """Run gradient descent to forge a trace matching human targets.

        Returns (forged_trace, convergence_result).
        """
        if targets is None:
            targets = self.compute_human_targets()

        weights = signal_weights or {k: 1.0 for k in targets}

        params = self._init_params()
        flat = self._params_to_flat(params)
        dim = len(flat)

        # Initial evaluation
        trace = _params_to_trace(params, seed=self.seed)
        signals = _extract_signals(trace)
        loss = _compute_loss(signals, targets, weights, trace, self.lambda_mono)

        result = ForgeryConvergenceResult(
            iterations=0,
            initial_loss=loss,
            final_loss=loss,
            target_signals=dict(targets),
        )
        result.loss_history.append(loss)
        result.signal_history.append(dict(signals))

        for iteration in range(1, max_iterations + 1):
            iter_seed = self.seed + iteration
            # Numerical gradient via finite differences
            grad = [0.0] * dim
            for d in range(dim):
                flat_plus = list(flat)
                flat_plus[d] += self.epsilon

                params_plus = self._flat_to_params(flat_plus)
                trace_plus = _params_to_trace(params_plus, seed=iter_seed)
                signals_plus = _extract_signals(trace_plus)
                loss_plus = _compute_loss(
                    signals_plus, targets, weights, trace_plus, self.lambda_mono
                )

                grad[d] = (loss_plus - loss) / self.epsilon

            # Gradient descent step
            for d in range(dim):
                flat[d] -= self.lr * grad[d]

            # Clamp glucose deltas to be non-negative
            for d in range(self.n_events):
                flat[d] = max(0.0, flat[d])

            # Re-evaluate
            params = self._flat_to_params(flat)
            trace = _params_to_trace(params, seed=iter_seed)
            signals = _extract_signals(trace)
            loss = _compute_loss(signals, targets, weights, trace, self.lambda_mono)

            if iteration % record_every == 0 or iteration == 1:
                result.loss_history.append(loss)
                result.signal_history.append(dict(signals))

            # Check convergence (relative to initial loss)
            initial_loss = result.loss_history[0] if result.loss_history else 1.0
            effective_threshold = convergence_threshold * initial_loss if initial_loss > 0.1 else convergence_threshold
            if loss < effective_threshold and not result.converged:
                result.converged = True
                result.convergence_iteration = iteration

        result.iterations = max_iterations
        result.final_loss = loss
        result.final_signals = dict(signals)

        return trace, result

    def run_convergence_study(
        self,
        n_runs: int = 10,
        max_iterations: int = 300,
        targets: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Run multiple forgery attempts and report convergence statistics."""
        if targets is None:
            targets = self.compute_human_targets()

        convergence_results: List[ForgeryConvergenceResult] = []

        for run in range(n_runs):
            forger = GradientForger(
                n_events=self.n_events,
                seed=self.seed + run * 1000,
                learning_rate=self.lr,
                epsilon=self.epsilon,
                lambda_mono=self.lambda_mono,
            )
            _, conv = forger.forge(
                targets=targets,
                max_iterations=max_iterations,
                record_every=max(1, max_iterations // 20),
            )
            convergence_results.append(conv)

        # Aggregate
        final_losses = [c.final_loss for c in convergence_results]
        initial_losses = [c.initial_loss for c in convergence_results]
        converged = [c for c in convergence_results if c.converged]
        convergence_iters = [c.convergence_iteration for c in converged if c.convergence_iteration]

        # Per-signal final deviation from target
        signal_deviations: Dict[str, List[float]] = {}
        for c in convergence_results:
            for sig, val in c.final_signals.items():
                target_val = targets.get(sig, 0.0)
                signal_deviations.setdefault(sig, []).append(abs(val - target_val))

        return {
            "n_runs": n_runs,
            "max_iterations": max_iterations,
            "convergence_rate": len(converged) / n_runs,
            "mean_final_loss": round(mean(final_losses), 6),
            "mean_initial_loss": round(mean(initial_losses), 6),
            "loss_reduction_ratio": round(
                mean(final_losses) / max(mean(initial_losses), 1e-10), 4
            ),
            "mean_convergence_iter": round(mean(convergence_iters), 1) if convergence_iters else None,
            "per_signal_mean_deviation": {
                k: round(mean(v), 4) for k, v in signal_deviations.items()
            },
            "targets": targets,
            "interpretation": _interpret_convergence(
                convergence_rate=len(converged) / n_runs,
                mean_loss_reduction=mean(final_losses) / max(mean(initial_losses), 1e-10),
            ),
        }


class EvolutionaryForger:
    """CMA-ES-inspired evolutionary strategy for trace forgery.

    Uses population-based search instead of gradients, which handles
    non-smooth objectives better. This is the "smart blind adversary" —
    it doesn't need to know the embodied model, just evaluates signals.
    """

    def __init__(
        self,
        n_events: int = 30,
        seed: int = 42,
        population_size: int = 20,
        sigma: float = 0.1,
        lambda_mono: float = 5.0,
    ):
        self.n_events = n_events
        self.seed = seed
        self.pop_size = population_size
        self.sigma = sigma
        self.lambda_mono = lambda_mono

    def forge(
        self,
        targets: Dict[str, float],
        max_generations: int = 200,
        convergence_threshold: float = 0.01,
        signal_weights: Optional[Dict[str, float]] = None,
        record_every: int = 10,
    ) -> Tuple[List[CausalEvent], ForgeryConvergenceResult]:
        """Run evolutionary optimization to forge a trace."""
        rng = random.Random(self.seed)
        weights = signal_weights or {k: 1.0 for k in targets}
        n = self.n_events
        dim = 4 * n

        # Initialize mean vector
        mean_vec = (
            [rng.uniform(0.005, 0.015) for _ in range(n)]   # glucose_deltas
            + [rng.uniform(-1.0, 0.0) for _ in range(n)]     # failure_probs
            + [rng.uniform(2.0, 6.0) for _ in range(n)]      # complexity
            + [rng.uniform(60.0, 120.0) for _ in range(n)]   # latency_base
        )

        best_loss = float("inf")
        best_flat = list(mean_vec)

        result = ForgeryConvergenceResult(
            iterations=0,
            initial_loss=0.0,
            final_loss=0.0,
            target_signals=dict(targets),
        )

        for gen in range(max_generations):
            # Generate population
            population = []
            for _ in range(self.pop_size):
                candidate = [
                    mean_vec[d] + rng.gauss(0, self.sigma) for d in range(dim)
                ]
                # Clamp glucose deltas positive
                for d in range(n):
                    candidate[d] = max(0.0, candidate[d])
                population.append(candidate)

            # Evaluate
            fitness = []
            for candidate in population:
                params = _TraceParams(
                    glucose_deltas=candidate[0:n],
                    failure_probs=candidate[n:2*n],
                    complexity=candidate[2*n:3*n],
                    latency_base=candidate[3*n:4*n],
                )
                trace = _params_to_trace(params, seed=self.seed)
                signals = _extract_signals(trace)
                loss = _compute_loss(signals, targets, weights, trace, self.lambda_mono)
                fitness.append((loss, candidate, signals))

            fitness.sort(key=lambda x: x[0])

            # Update mean from top 50%
            elite_size = self.pop_size // 2
            mean_vec = [0.0] * dim
            for d in range(dim):
                mean_vec[d] = sum(
                    fitness[i][1][d] for i in range(elite_size)
                ) / elite_size

            # Track best
            if fitness[0][0] < best_loss:
                best_loss = fitness[0][0]
                best_flat = list(fitness[0][1])

            if gen == 0:
                result.initial_loss = fitness[0][0]

            if gen % record_every == 0 or gen == 0:
                result.loss_history.append(fitness[0][0])
                result.signal_history.append(dict(fitness[0][2]))

            if best_loss < convergence_threshold and not result.converged:
                result.converged = True
                result.convergence_iteration = gen

        # Final trace from best
        params = _TraceParams(
            glucose_deltas=best_flat[0:n],
            failure_probs=best_flat[n:2*n],
            complexity=best_flat[2*n:3*n],
            latency_base=best_flat[3*n:4*n],
        )
        trace = _params_to_trace(params, seed=self.seed)
        signals = _extract_signals(trace)

        result.iterations = max_generations
        result.final_loss = best_loss
        result.final_signals = dict(signals)

        return trace, result

    def run_convergence_study(
        self,
        n_runs: int = 10,
        max_generations: int = 200,
        targets: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Run multiple evolutionary forgery attempts."""
        if targets is None:
            gf = GradientForger(n_events=self.n_events, seed=self.seed)
            targets = gf.compute_human_targets()

        convergence_results: List[ForgeryConvergenceResult] = []
        for run in range(n_runs):
            forger = EvolutionaryForger(
                n_events=self.n_events,
                seed=self.seed + run * 1000,
                population_size=self.pop_size,
                sigma=self.sigma,
                lambda_mono=self.lambda_mono,
            )
            _, conv = forger.forge(
                targets=targets,
                max_generations=max_generations,
                record_every=max(1, max_generations // 20),
            )
            convergence_results.append(conv)

        final_losses = [c.final_loss for c in convergence_results]
        initial_losses = [c.initial_loss for c in convergence_results]
        converged = [c for c in convergence_results if c.converged]
        convergence_iters = [c.convergence_iteration for c in converged if c.convergence_iteration]

        signal_deviations: Dict[str, List[float]] = {}
        for c in convergence_results:
            for sig, val in c.final_signals.items():
                target_val = targets.get(sig, 0.0)
                signal_deviations.setdefault(sig, []).append(abs(val - target_val))

        conv_rate = len(converged) / n_runs
        loss_ratio = mean(final_losses) / max(mean(initial_losses), 1e-10)

        return {
            "method": "evolutionary",
            "n_runs": n_runs,
            "max_generations": max_generations,
            "population_size": self.pop_size,
            "convergence_rate": conv_rate,
            "mean_final_loss": round(mean(final_losses), 6),
            "mean_initial_loss": round(mean(initial_losses), 6),
            "loss_reduction_ratio": round(loss_ratio, 4),
            "mean_convergence_iter": round(mean(convergence_iters), 1) if convergence_iters else None,
            "per_signal_mean_deviation": {
                k: round(mean(v), 4) for k, v in signal_deviations.items()
            },
            "targets": targets,
            "interpretation": _interpret_convergence(conv_rate, loss_ratio),
        }


def _interpret_convergence(convergence_rate: float, mean_loss_reduction: float) -> str:
    """Interpret convergence results for the paper narrative."""
    if convergence_rate > 0.8 and mean_loss_reduction < 0.1:
        return (
            "FAST convergence: optimizer reliably finds human-matching traces. "
            "Supports impossibility theorem — forgery is computationally cheap."
        )
    elif convergence_rate > 0.5:
        return (
            "MODERATE convergence: optimizer finds partial matches. "
            "Signals offer some resistance but are ultimately defeatable."
        )
    elif mean_loss_reduction < 0.5:
        return (
            "SLOW convergence: optimizer reduces loss significantly but rarely converges fully. "
            "Signals have moderate computational hardness, consistent with VDF analogy."
        )
    else:
        return (
            "WEAK convergence: optimizer struggles to match human signals. "
            "Signals may have genuine computational hardness beyond the VDF analogy."
        )
