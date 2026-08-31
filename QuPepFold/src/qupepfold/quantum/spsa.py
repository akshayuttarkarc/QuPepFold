"""SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer.

Standard choice for variational quantum algorithms on noisy hardware
due to robustness to shot noise and low evaluation budget requirements.
"""

from typing import Callable, Optional, Tuple, List
import numpy as np


def spsa_optimize(
    cost_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    n_iterations: int = 50,
    a: float = 0.1,
    c: float = 0.1,
    A: Optional[float] = None,
    alpha: float = 0.602,
    gamma: float = 0.101,
    seed: int = 42,
    callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
) -> Tuple[np.ndarray, float, List[float]]:
    """Optimize using SPSA algorithm.
    
    SPSA uses two function evaluations per iteration to estimate
    the gradient via finite differences with random perturbations.
    
    Learning rate schedule: a_k = a / (A + k + 1)^alpha
    Perturbation schedule: c_k = c / (k + 1)^gamma
    
    Args:
        cost_fn: Function to minimize, takes parameter array, returns scalar.
        x0: Initial parameters.
        n_iterations: Number of iterations.
        a: Learning rate coefficient.
        c: Perturbation size coefficient.
        A: Stability constant (default: 10% of iterations).
        alpha: Learning rate decay exponent.
        gamma: Perturbation decay exponent.
        seed: Random seed for reproducibility.
        callback: Optional callback(iteration, params, cost).
        
    Returns:
        Tuple of (best_params, best_cost, cost_history).
        
    Example:
        >>> def cost(x):
        ...     return np.sum(x**2)
        >>> x_opt, cost_opt, _ = spsa_optimize(cost, np.ones(10))
    """
    rng = np.random.default_rng(seed)
    
    # Default stability constant
    if A is None:
        A = 0.1 * n_iterations
    
    x = x0.copy().astype(float)
    n_params = len(x)
    
    best_x = x.copy()
    best_cost = cost_fn(x)
    cost_history = [best_cost]
    
    for k in range(n_iterations):
        # Compute step sizes
        a_k = a / (A + k + 1) ** alpha
        c_k = c / (k + 1) ** gamma
        
        # Random perturbation direction (Bernoulli ±1)
        delta = rng.choice([-1.0, 1.0], size=n_params)
        
        # Perturbed evaluations (standard 2 evaluations per SPSA iteration)
        x_plus = x + c_k * delta
        x_minus = x - c_k * delta
        
        cost_plus = cost_fn(x_plus)
        cost_minus = cost_fn(x_minus)
        
        # Gradient estimate
        g_hat = (cost_plus - cost_minus) / (2 * c_k * delta)
        
        # Update parameters
        x = x - a_k * g_hat
        
        # Track iteration cost without redundant extra evaluation
        iter_cost = min(cost_plus, cost_minus)
        cost_history.append(iter_cost)
        
        if cost_plus < best_cost:
            best_cost = cost_plus
            best_x = x_plus.copy()
        if cost_minus < best_cost:
            best_cost = cost_minus
            best_x = x_minus.copy()
        
        # Callback
        if callback is not None:
            callback(k, x, iter_cost)
            
    # Final evaluation at converged parameters
    final_cost = cost_fn(x)
    cost_history.append(final_cost)
    if final_cost <= best_cost:
        best_cost = final_cost
        best_x = x.copy()
    
    return best_x, best_cost, cost_history


def spsa_with_averaging(
    cost_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    n_iterations: int = 50,
    avg_window: int = 10,
    **kwargs,
) -> Tuple[np.ndarray, float, List[float]]:
    """SPSA with parameter averaging for improved final estimate.
    
    Averages the last `avg_window` parameter vectors for smoother convergence.
    
    Args:
        cost_fn: Function to minimize.
        x0: Initial parameters.
        n_iterations: Number of iterations.
        avg_window: Number of final iterations to average.
        **kwargs: Additional arguments for spsa_optimize.
        
    Returns:
        Tuple of (averaged_params, final_cost, cost_history).
    """
    # Collect parameters over iterations
    param_history = []
    
    def collect_callback(k, x, cost):
        param_history.append(x.copy())
        if 'callback' in kwargs and kwargs['callback'] is not None:
            kwargs['callback'](k, x, cost)
    
    callbacks_kwargs = {**kwargs}
    callbacks_kwargs['callback'] = collect_callback
    
    _, best_cost, cost_history = spsa_optimize(
        cost_fn, x0, n_iterations, **callbacks_kwargs
    )
    
    # Average last window
    if len(param_history) >= avg_window:
        avg_params = np.mean(param_history[-avg_window:], axis=0)
    else:
        avg_params = np.mean(param_history, axis=0)
    
    avg_cost = cost_fn(avg_params)
    
    return avg_params, avg_cost, cost_history


# ── Additional optimizer implementations ─────────────────────────────────────

def cobyla_optimize(
    cost_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    n_iterations: int = 50,
    rhobeg: float = 0.5,
    **kwargs,
) -> Tuple[np.ndarray, float, List[float]]:
    """Optimize using COBYLA (Constrained Optimization BY Linear Approximations).

    Uses scipy.optimize.minimize with method='COBYLA'.
    Suitable for deterministic (statevector) cost functions.

    Args:
        cost_fn: Function to minimize.
        x0: Initial parameters.
        n_iterations: Maximum function evaluations.
        rhobeg: Initial trust-region radius.
        **kwargs: Ignored (for interface compatibility).

    Returns:
        Tuple of (best_params, best_cost, cost_history).
    """
    from scipy.optimize import minimize
    cost_history: List[float] = []

    def tracked_cost(x):
        val = float(cost_fn(x))
        cost_history.append(val)
        return val

    result = minimize(
        tracked_cost, x0,
        method="COBYLA",
        options={"maxiter": n_iterations * 2, "rhobeg": rhobeg},
    )
    return result.x, float(result.fun), cost_history


def nelder_mead_optimize(
    cost_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    n_iterations: int = 50,
    **kwargs,
) -> Tuple[np.ndarray, float, List[float]]:
    """Optimize using Nelder-Mead simplex algorithm.

    Uses scipy.optimize.minimize with method='Nelder-Mead'.
    Good for noisy cost landscapes with moderate parameter counts.

    Args:
        cost_fn: Function to minimize.
        x0: Initial parameters.
        n_iterations: Maximum iterations.
        **kwargs: Ignored.

    Returns:
        Tuple of (best_params, best_cost, cost_history).
    """
    from scipy.optimize import minimize
    cost_history: List[float] = []

    def tracked_cost(x):
        val = float(cost_fn(x))
        cost_history.append(val)
        return val

    result = minimize(
        tracked_cost, x0,
        method="Nelder-Mead",
        options={"maxiter": n_iterations * 5, "xatol": 1e-4, "fatol": 1e-4},
    )
    return result.x, float(result.fun), cost_history


def gradient_optimize(
    cost_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    n_iterations: int = 50,
    lr: float = 0.01,
    fd_step: float = 1e-3,
    seed: int = 42,
    **kwargs,
) -> Tuple[np.ndarray, float, List[float]]:
    """Gradient descent using finite-difference gradient estimates.

    Suitable for statevector simulators where cost_fn is deterministic.

    Args:
        cost_fn: Function to minimize.
        x0: Initial parameters.
        n_iterations: Number of gradient steps.
        lr: Learning rate.
        fd_step: Finite-difference step size.
        seed: Not used (for interface compatibility).
        **kwargs: Ignored.

    Returns:
        Tuple of (best_params, best_cost, cost_history).
    """
    x = x0.copy().astype(float)
    best_x = x.copy()
    best_cost = float(cost_fn(x))
    cost_history = [best_cost]

    for _ in range(n_iterations):
        # Finite-difference gradient
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += fd_step
            x_minus = x.copy()
            x_minus[i] -= fd_step
            grad[i] = (cost_fn(x_plus) - cost_fn(x_minus)) / (2 * fd_step)

        x = x - lr * grad
        cost = float(cost_fn(x))
        cost_history.append(cost)
        if cost < best_cost:
            best_cost = cost
            best_x = x.copy()

    return best_x, best_cost, cost_history


# ── Optimizer factory ─────────────────────────────────────────────────────────

_OPTIMIZER_MAP = {
    "spsa": spsa_optimize,
    "cobyla": cobyla_optimize,
    "nelder_mead": nelder_mead_optimize,
    "gradient": gradient_optimize,
}


def get_optimizer(name: str) -> Callable:
    """Return an optimizer function by name.

    All returned functions share the interface:
        fn(cost_fn, x0, n_iterations, ...) -> (best_params, best_cost, history)

    Args:
        name: One of "spsa", "cobyla", "nelder_mead", "gradient".

    Returns:
        Optimizer function.
    """
    fn = _OPTIMIZER_MAP.get(name)
    if fn is None:
        raise ValueError(
            f"Unknown optimizer '{name}'. Available: {list(_OPTIMIZER_MAP.keys())}"
        )
    return fn
