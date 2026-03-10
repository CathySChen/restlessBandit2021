"""
Hidden Markov Model for a 2-armed restless bandit task.

Identical in structure to hmm_models.py (3-armed version) with the following
differences:

  - 3 hidden states instead of 4:  1 explore + 2 exploit
  - 2 observations instead of 3:   choice 0 and choice 1
  - Parameter tying averages over 2 exploit states instead of 3

States
------
  0  Explore (ORE) — emits both choices with equal probability 1/2
  1  Exploit choice 0 — emits only choice 0
  2  Exploit choice 1 — emits only choice 1

Emission matrix (fixed, never updated by EM):
  b = [[1/2, 1/2],   # explore
       [1,   0  ],   # exploit choice 0
       [0,   1  ]]   # exploit choice 1

Transition matrix structure (enforced via initialisation):
  - Explore can transition to any state (row 0 unconstrained).
  - Each exploit state can only self-transition or return to explore;
    exploit-to-different-exploit entries are initialised to 0 and stay 0
    because those paths never accumulate forward probability.
  - Always starts in the explore state: initial_distribution = [1, 0, 0].

Parameter tying (applied ONCE after the full EM loop):
  - explore→exploit entries a[0,1] and a[0,2] are averaged and tied.
  - exploit self-transitions a[1,1] and a[2,2] are averaged and tied.
  - exploit→explore entries a[1,0] and a[2,0] are averaged and tied.
  - a[0,0] and off-diagonal exploit entries are left unchanged.

Numerical stability
-------------------
  All forward and backward passes use per-step normalisation.
  Log-likelihood is accumulated as sum(log(scale_t)).

Convergence
-----------
  Baum-Welch iterates until |ΔLL| < tol (default 1e-4) or max_iter is
  reached (default 1 000).

Reseed sampling
---------------
  Each row of the initial transition matrix draws an independent random
  value, giving better coverage across the 10 random restarts.
"""

import numpy as np
from typing import Optional, Tuple, List


# ── Utility helper ────────────────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> Tuple[np.ndarray, float]:
    """Normalise a non-negative vector to sum to 1.

    Returns ``(normalised, scale)`` where ``scale`` is the original sum.
    If the sum is zero the input is returned unchanged with scale 0.
    """
    s = float(v.sum())
    if s == 0:
        return v.copy(), 0.0
    return v / s, s


# ── Core algorithms ───────────────────────────────────────────────────────────

def forward(V: np.ndarray,
            a: np.ndarray,
            b: np.ndarray,
            initial_distribution: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Scaled forward algorithm for HMM inference.

    Parameters
    ----------
    V : np.ndarray, shape (T,)
        Observation (choice) sequence; integer values in {0, 1}.
    a : np.ndarray, shape (n_states, n_states)
        Transition matrix.  ``a[i,j]`` = P(next=j | curr=i).
    b : np.ndarray, shape (n_states, n_obs)
        Emission matrix.  ``b[i,o]`` = P(obs=o | state=i).
    initial_distribution : np.ndarray, shape (n_states,)
        Initial state distribution.

    Returns
    -------
    alpha : np.ndarray, shape (T, n_states)
        Scaled alpha matrix (each row sums to 1).
    scale : np.ndarray, shape (T,)
        Normalisation constant at each time step.
    log_likelihood : float
        ``sum(log(scale))``.  Returns ``-inf`` if any scale == 0.
    """
    T = V.shape[0]
    M = a.shape[0]

    alpha = np.zeros((T, M))
    scale = np.zeros(T)

    alpha[0, :] = initial_distribution * b[:, V[0]]
    alpha[0, :], scale[0] = _normalize(alpha[0, :])

    for t in range(1, T):
        m = alpha[t - 1, :] @ a
        alpha[t, :] = m * b[:, V[t]]
        alpha[t, :], scale[t] = _normalize(alpha[t, :])

    if np.any(scale == 0):
        ll = -np.inf
    else:
        ll = float(np.sum(np.log(scale)))

    return alpha, scale, ll


def backward(V: np.ndarray,
             a: np.ndarray,
             b: np.ndarray) -> np.ndarray:
    """Scaled backward algorithm for HMM inference.

    Parameters
    ----------
    V : np.ndarray, shape (T,)
        Observation (choice) sequence.
    a : np.ndarray, shape (n_states, n_states)
        Transition matrix.
    b : np.ndarray, shape (n_states, n_obs)
        Emission matrix.

    Returns
    -------
    np.ndarray, shape (T, n_states)
        Scaled beta matrix.
    """
    T = V.shape[0]
    M = a.shape[0]

    beta = np.zeros((T, M))
    beta[T - 1, :] = 1.0

    for t in range(T - 2, -1, -1):
        b_next = beta[t + 1, :] * b[:, V[t + 1]]
        beta[t, :] = a @ b_next
        beta[t, :], _ = _normalize(beta[t, :])

    return beta


def baum_welch(V: np.ndarray,
               a: np.ndarray,
               b: np.ndarray,
               initial_distribution: np.ndarray,
               max_iter: int = 1000,
               tol: float = 1e-4,
               parameter_tying: bool = True
               ) -> Tuple[np.ndarray, float, List[float]]:
    """Baum-Welch EM for a 3-state (2-armed) HMM.

    Runs until ``|ΔLL| < tol`` or ``max_iter`` is reached.
    Parameter tying is applied once after the full EM loop.

    Parameter tying (applied after the EM loop, rows re-normalised after):
      - ``a[0, 1:3]``  explore→exploit averaged and tied.
      - Exploit self-transitions ``a[1,1], a[2,2]`` averaged and tied.
      - Exploit→explore entries ``a[1,0], a[2,0]`` averaged and tied.

    Parameters
    ----------
    V : np.ndarray, shape (T,)
    a : np.ndarray, shape (3, 3)
    b : np.ndarray, shape (3, 2)
    initial_distribution : np.ndarray, shape (3,)
    max_iter : int
    tol : float
    parameter_tying : bool

    Returns
    -------
    a_fit : np.ndarray, shape (3, 3)
    log_likelihood : float
    ll_history : list of float
    """
    M = a.shape[0]
    T = len(V)

    old_ll:    float      = 0.0
    ll_history: List[float] = []

    for iteration in range(max_iter):

        # ── E step — forward ───────────────────────────────────────────────
        alpha, _, ll = forward(V, a, b, initial_distribution)
        ll_history.append(ll)

        if ll == -np.inf:
            break

        if iteration > 0 and abs(old_ll - ll) < tol:
            break
        old_ll = ll

        # ── E step — backward ──────────────────────────────────────────────
        beta = backward(V, a, b)

        # ── E step — xi ────────────────────────────────────────────────────
        exp_num_trans = np.zeros((M, M))
        for t in range(T - 1):
            b_next = beta[t + 1, :] * b[:, V[t + 1]]
            xi_t   = a * np.outer(alpha[t, :], b_next)
            xi_flat, _ = _normalize(xi_t.ravel())
            exp_num_trans += xi_flat.reshape(M, M)

        # ── M step ────────────────────────────────────────────────────────
        row_sums = exp_num_trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        a = exp_num_trans / row_sums

    # ── Parameter tying ONCE after the full EM loop ───────────────────────
    if parameter_tying:
        # Tie explore→exploit (2 entries)
        a[0, 1] = (a[0, 1] + a[0, 2]) / 2
        a[0, 2] = a[0, 1]

        # Tie exploit self-transitions (2 entries)
        a[1, 1] = (a[1, 1] + a[2, 2]) / 2
        a[2, 2] = a[1, 1]

        # Tie exploit→explore (2 entries)
        a[1, 0] = (a[1, 0] + a[2, 0]) / 2
        a[2, 0] = a[1, 0]

        # Re-normalise rows after tying
        row_sums = a.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        a = a / row_sums

    return a, ll, ll_history


def viterbi(V: np.ndarray,
            a: np.ndarray,
            b: np.ndarray,
            initial_distribution: np.ndarray) -> np.ndarray:
    """Viterbi algorithm — most likely hidden state sequence (log-space).

    Parameters
    ----------
    V : np.ndarray, shape (T,)
    a : np.ndarray, shape (n_states, n_states)
    b : np.ndarray, shape (n_states, n_obs)
    initial_distribution : np.ndarray, shape (n_states,)

    Returns
    -------
    np.ndarray, shape (T,), dtype int
    """
    T = V.shape[0]
    M = a.shape[0]

    log_a  = np.log(a  + 1e-300)
    log_b  = np.log(b  + 1e-300)
    log_pi = np.log(initial_distribution + 1e-300)

    omega = np.full((T, M), -np.inf)
    prev  = np.zeros((T - 1, M), dtype=int)

    omega[0, :] = log_pi + log_b[:, V[0]]

    for t in range(1, T):
        for j in range(M):
            scores         = omega[t - 1, :] + log_a[:, j] + log_b[j, V[t]]
            prev[t - 1, j] = int(np.argmax(scores))
            omega[t, j]    = float(np.max(scores))

    S        = np.zeros(T, dtype=int)
    S[T - 1] = int(np.argmax(omega[T - 1, :]))
    for t in range(T - 2, -1, -1):
        S[t] = prev[t, S[t + 1]]

    return S


def stationary_distribution_eigen(transition_matrix: np.ndarray) -> np.ndarray:
    """Stationary distribution via eigen-decomposition.

    Parameters
    ----------
    transition_matrix : np.ndarray, shape (n_states, n_states)

    Returns
    -------
    np.ndarray, shape (n_states,)
    """
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    v   = np.real(eigenvectors[:, idx])
    return v / v.sum()


# ── Top-level fitting function ────────────────────────────────────────────────

def fit_hmm_2arm(choices: np.ndarray,
                 n_states: int = 3,
                 n_obs: int = 2,
                 max_iter: int = 1000,
                 tol: float = 1e-4,
                 n_reseeds: int = 10,
                 parameter_tying: bool = True,
                 random_state: Optional[int] = None) -> dict:
    """Fit a 3-state HMM to a 2-armed restless bandit choice sequence.

    Identical to ``fit_hmm`` in ``hmm_models.py`` except:

    - Default ``n_states=3`` (1 explore + 2 exploit).
    - Default ``n_obs=2`` (two choices).
    - Parameter tying averages over 2 exploit states instead of 3.
    - Emission matrix:  b = [[0.5, 0.5], [1, 0], [0, 1]].

    All other behaviour is preserved:

    * Scaled forward-backward (no underflow).
    * Convergence criterion ``|ΔLL| < tol``.
    * Per-row independent initialisation.
    * Acceptance criterion: reject if ``a[0,0] ≤ 0.5 AND a[1,0] ≥ 0.5``.
    * Best LL among accepted seeds returned; fallback to last if none accepted.
    * ``convergence_histories`` returned for use with ``plot_convergence()``.

    Parameters
    ----------
    choices : np.ndarray, shape (T,)
        Integer choice sequence, values in {0, 1}.
    n_states : int
        Number of hidden states (default: 3 = 1 explore + 2 exploit).
    n_obs : int
        Number of distinct choices (default: 2).
    max_iter : int
        Maximum Baum-Welch iterations per seed (default: 1 000).
    tol : float
        Convergence tolerance on ``|ΔLL|`` (default: 1e-4).
    n_reseeds : int
        Maximum number of random restarts (default: 10).
    parameter_tying : bool
        Apply symmetric tying after EM (default: True).
    random_state : int or None
        Base random seed.

    Returns
    -------
    dict with keys:

    transition_matrix : np.ndarray, shape (n_states, n_states)
    state_labels : np.ndarray, shape (T,)
    state_label_names : list of str
    log_likelihood : float
    emission_matrix : np.ndarray, shape (n_states, n_obs)
    initial_distribution : np.ndarray, shape (n_states,)
    all_log_likelihoods : list of float
    convergence_histories : list of list of float
    best_seed_index : int
    n_accepted : int
    n_iters_per_seed : list of int

    Examples
    --------
    >>> choices = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0])
    >>> result = fit_hmm_2arm(choices, n_reseeds=5, random_state=42)
    >>> result['state_labels'].shape
    (10,)
    >>> result['transition_matrix'].shape
    (3, 3)
    """
    choices = np.asarray(choices, dtype=int)

    # Fixed emission matrix
    b = np.zeros((n_states, n_obs))
    b[0, :] = 1.0 / n_obs           # explore: uniform over both choices
    for k in range(1, n_states):
        b[k, k - 1] = 1.0           # exploit k: emits only choice k-1

    # Always start in explore state
    initial_distribution = np.zeros(n_states)
    initial_distribution[0] = 1.0

    STATE_NAMES = {0: 'explore'}
    for k in range(1, n_states):
        STATE_NAMES[k] = f'exploit_choice{k - 1}'

    accepted_matrices: list = []
    accepted_lls:      list = []
    all_histories:     list = []
    all_n_iters:       list = []
    last_a                  = None
    last_ll                 = -np.inf

    for reseed in range(n_reseeds):
        seed = (random_state + reseed) if random_state is not None else None
        rng  = np.random.default_rng(seed)

        # Per-row independent random initialisation
        rand_rows = rng.random(n_states)   # one value per row

        a_init = np.zeros((n_states, n_states))
        a_init[0, 0]  = 1.0 - rand_rows[0]
        a_init[0, 1:] = rand_rows[0] / (n_states - 1)
        for k in range(1, n_states):
            a_init[k, 0] = 1.0 - rand_rows[k]
            a_init[k, k] = rand_rows[k]

        a_fit, ll, ll_history = baum_welch(
            choices, a_init, b, initial_distribution,
            max_iter=max_iter,
            tol=tol,
            parameter_tying=parameter_tying,
        )
        all_histories.append(ll_history)
        all_n_iters.append(len(ll_history))

        last_a  = a_fit
        last_ll = ll

        # Acceptance criterion (same as 3-arm version)
        a00 = float(a_fit[0, 0])
        a10 = float(a_fit[1, 0])
        criteria_met = not (a00 <= 0.5 and a10 >= 0.5)

        if criteria_met:
            accepted_matrices.append(a_fit.copy())
            accepted_lls.append(ll)

    if not accepted_matrices:
        accepted_matrices = [last_a]
        accepted_lls      = [last_ll]

    best_idx_in_accepted = int(np.argmax(accepted_lls))
    best_a               = accepted_matrices[best_idx_in_accepted]
    best_ll              = accepted_lls[best_idx_in_accepted]

    # Map back to global seed index
    best_seed_global = 0
    for i, hist in enumerate(all_histories):
        if hist and abs(hist[-1] - best_ll) < 1e-8:
            best_seed_global = i
            break

    state_labels = viterbi(choices, best_a, b, initial_distribution)

    return {
        'transition_matrix':     best_a,
        'state_labels':          state_labels,
        'state_label_names':     [STATE_NAMES[s] for s in state_labels],
        'log_likelihood':        best_ll,
        'emission_matrix':       b,
        'initial_distribution':  initial_distribution,
        'all_log_likelihoods':   accepted_lls,
        'convergence_histories': all_histories,
        'best_seed_index':       best_seed_global,
        'n_accepted':            len(accepted_matrices),
        'n_iters_per_seed':      all_n_iters,
    }


# ── Multi-session fitting ─────────────────────────────────────────────────────

def baum_welch_multi(sessions: List[np.ndarray],
                     a: np.ndarray,
                     b: np.ndarray,
                     initial_distribution: np.ndarray,
                     max_iter: int = 1000,
                     tol: float = 1e-4,
                     parameter_tying: bool = True
                     ) -> Tuple[np.ndarray, float, List[float]]:
    """Baum-Welch EM over multiple independent sessions.

    Each session gets its own forward-backward pass starting from the explore
    state, so no artificial transition is ever counted across session
    boundaries.  The sufficient statistics (xi counts) are then pooled across
    all sessions before the M-step, estimating a single shared transition
    matrix for the subject.

    Total log-likelihood is the sum of per-session LLs, which is the correct
    joint log P(all sessions | model) under the independence assumption.

    Everything else — scaled forward-backward, convergence criterion, parameter
    tying — is identical to ``baum_welch``.

    Parameters
    ----------
    sessions : list of np.ndarray, each shape (T_s,)
        Choice sequences for each session.  Sessions may have different lengths.
        Values must be in {0, 1}.
    a : np.ndarray, shape (3, 3)
        Initial transition matrix.
    b : np.ndarray, shape (3, 2)
        Fixed emission matrix.
    initial_distribution : np.ndarray, shape (3,)
        Initial state distribution (always ``[1, 0, 0]``; applied fresh to
        every session).
    max_iter : int
        Maximum EM iterations (default: 1 000).
    tol : float
        Convergence tolerance on ``|Δ total LL|`` (default: 1e-4).
    parameter_tying : bool
        Apply symmetric tying after the EM loop (default: True).

    Returns
    -------
    a_fit : np.ndarray, shape (3, 3)
    total_log_likelihood : float
        Sum of log-likelihoods across all sessions.
    ll_history : list of float
        Total LL recorded at the start of each EM iteration.
    """
    M      = a.shape[0]
    old_ll = 0.0
    ll_history: List[float] = []

    for iteration in range(max_iter):

        # ── E step: pool xi counts across all sessions ─────────────────────
        exp_num_trans = np.zeros((M, M))
        total_ll      = 0.0

        for V in sessions:
            alpha, _, ll_s = forward(V, a, b, initial_distribution)

            if ll_s == -np.inf:
                total_ll = -np.inf
                break

            total_ll += ll_s
            beta = backward(V, a, b)

            T = len(V)
            for t in range(T - 1):
                b_next  = beta[t + 1, :] * b[:, V[t + 1]]
                xi_t    = a * np.outer(alpha[t, :], b_next)
                xi_flat, _ = _normalize(xi_t.ravel())
                exp_num_trans += xi_flat.reshape(M, M)

        ll_history.append(total_ll)

        if total_ll == -np.inf:
            break

        # ── Convergence check ──────────────────────────────────────────────
        if iteration > 0 and abs(old_ll - total_ll) < tol:
            break
        old_ll = total_ll

        # ── M step: row-normalise pooled sufficient statistics ─────────────
        row_sums = exp_num_trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        a = exp_num_trans / row_sums

    # ── Parameter tying ONCE after the full EM loop ────────────────────────
    if parameter_tying:
        a[0, 1] = (a[0, 1] + a[0, 2]) / 2
        a[0, 2] = a[0, 1]

        a[1, 1] = (a[1, 1] + a[2, 2]) / 2
        a[2, 2] = a[1, 1]

        a[1, 0] = (a[1, 0] + a[2, 0]) / 2
        a[2, 0] = a[1, 0]

        row_sums = a.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        a = a / row_sums

    return a, total_ll, ll_history


def fit_hmm_2arm_multi(sessions: List[np.ndarray],
                       n_states: int = 3,
                       n_obs: int = 2,
                       max_iter: int = 1000,
                       tol: float = 1e-4,
                       n_reseeds: int = 10,
                       parameter_tying: bool = True,
                       random_state: Optional[int] = None) -> dict:
    """Fit a single 3-state HMM to multiple sessions from one subject.

    Uses multi-sequence Baum-Welch: forward-backward is run independently on
    each session (so session boundaries are never crossed), then xi counts are
    pooled before the M-step.  This yields one transition matrix that reflects
    the subject's behaviour across all sessions without contaminating the
    estimate with spurious boundary transitions.

    Parameters
    ----------
    sessions : list of np.ndarray, each shape (T_s,)
        One array per session.  Each array is an integer choice sequence with
        values in {0, 1}.  Sessions may differ in length.
    n_states : int
        Number of hidden states (default: 3 = 1 explore + 2 exploit).
    n_obs : int
        Number of distinct choices (default: 2).
    max_iter : int
        Maximum Baum-Welch iterations per seed (default: 1 000).
    tol : float
        Convergence tolerance on ``|Δ total LL|`` (default: 1e-4).
    n_reseeds : int
        Maximum number of random restarts (default: 10).
    parameter_tying : bool
        Apply symmetric tying after EM (default: True).
    random_state : int or None
        Base random seed.

    Returns
    -------
    dict with keys:

    transition_matrix : np.ndarray, shape (n_states, n_states)
        Single shared transition matrix fitted across all sessions.
    state_labels : list of np.ndarray
        Viterbi-decoded state sequence for each session, in the same order
        as ``sessions``.
    state_label_names : list of list of str
        Human-readable state name per trial, per session.
    log_likelihood : float
        Total log-likelihood summed across all sessions.
    per_session_log_likelihoods : list of float
        Final log P(session_s | best model) for each session individually.
    emission_matrix : np.ndarray, shape (n_states, n_obs)
    initial_distribution : np.ndarray, shape (n_states,)
    all_log_likelihoods : list of float
        Total LL from every accepted reseed.
    convergence_histories : list of list of float
        Per-iteration total LL trajectory for every seed.
    best_seed_index : int
    n_accepted : int
    n_iters_per_seed : list of int
    n_sessions : int
        Number of sessions fitted.
    session_lengths : list of int
        Number of trials in each session.

    Examples
    --------
    >>> s1 = np.array([0, 0, 1, 1, 0, 0, 0, 1])
    >>> s2 = np.array([1, 1, 0, 0, 1, 1, 1, 0, 0])
    >>> result = fit_hmm_2arm_multi([s1, s2], random_state=0)
    >>> len(result['state_labels'])      # one array per session
    2
    >>> result['transition_matrix'].shape
    (3, 3)
    """
    sessions = [np.asarray(s, dtype=int) for s in sessions]

    # Fixed emission matrix
    b = np.zeros((n_states, n_obs))
    b[0, :] = 1.0 / n_obs
    for k in range(1, n_states):
        b[k, k - 1] = 1.0

    # Always start in explore at the beginning of each session
    initial_distribution = np.zeros(n_states)
    initial_distribution[0] = 1.0

    STATE_NAMES = {0: 'explore'}
    for k in range(1, n_states):
        STATE_NAMES[k] = f'exploit_choice{k - 1}'

    accepted_matrices: list = []
    accepted_lls:      list = []
    all_histories:     list = []
    all_n_iters:       list = []
    last_a                  = None
    last_ll                 = -np.inf

    for reseed in range(n_reseeds):
        seed = (random_state + reseed) if random_state is not None else None
        rng  = np.random.default_rng(seed)

        rand_rows = rng.random(n_states)

        a_init = np.zeros((n_states, n_states))
        a_init[0, 0]  = 1.0 - rand_rows[0]
        a_init[0, 1:] = rand_rows[0] / (n_states - 1)
        for k in range(1, n_states):
            a_init[k, 0] = 1.0 - rand_rows[k]
            a_init[k, k] = rand_rows[k]

        a_fit, ll, ll_history = baum_welch_multi(
            sessions, a_init, b, initial_distribution,
            max_iter=max_iter, tol=tol,
            parameter_tying=parameter_tying,
        )
        all_histories.append(ll_history)
        all_n_iters.append(len(ll_history))

        last_a  = a_fit
        last_ll = ll

        a00 = float(a_fit[0, 0])
        a10 = float(a_fit[1, 0])
        criteria_met = not (a00 <= 0.5 and a10 >= 0.5)

        if criteria_met:
            accepted_matrices.append(a_fit.copy())
            accepted_lls.append(ll)

    if not accepted_matrices:
        accepted_matrices = [last_a]
        accepted_lls      = [last_ll]

    best_idx_in_accepted = int(np.argmax(accepted_lls))
    best_a               = accepted_matrices[best_idx_in_accepted]
    best_ll              = accepted_lls[best_idx_in_accepted]

    best_seed_global = 0
    for i, hist in enumerate(all_histories):
        if hist and abs(hist[-1] - best_ll) < 1e-8:
            best_seed_global = i
            break

    # Viterbi decode each session independently with the best transition matrix
    state_labels_per_session    = []
    state_names_per_session     = []
    per_session_lls             = []

    for V in sessions:
        labels = viterbi(V, best_a, b, initial_distribution)
        state_labels_per_session.append(labels)
        state_names_per_session.append([STATE_NAMES[s] for s in labels])
        _, _, ll_s = forward(V, best_a, b, initial_distribution)
        per_session_lls.append(float(ll_s))

    return {
        'transition_matrix':           best_a,
        'state_labels':                state_labels_per_session,
        'state_label_names':           state_names_per_session,
        'log_likelihood':              best_ll,
        'per_session_log_likelihoods': per_session_lls,
        'emission_matrix':             b,
        'initial_distribution':        initial_distribution,
        'all_log_likelihoods':         accepted_lls,
        'convergence_histories':       all_histories,
        'best_seed_index':             best_seed_global,
        'n_accepted':                  len(accepted_matrices),
        'n_iters_per_seed':            all_n_iters,
        'n_sessions':                  len(sessions),
        'session_lengths':             [len(s) for s in sessions],
    }


# ── Convergence diagnostics ───────────────────────────────────────────────────

def plot_convergence(result: dict,
                     subject_id: str = '',
                     save_path: Optional[str] = None):
    """Plot LL convergence curves for all seeds from a ``fit_hmm_2arm`` result.

    Two-panel figure:
    - **Left** — per-iteration LL trajectory per seed (best seed in blue).
    - **Right** — bar chart of each seed's final LL (best seed in blue).

    Parameters
    ----------
    result : dict
        Return value of ``fit_hmm_2arm()``.
    subject_id : str
        Optional label appended to figure titles.
    save_path : str or None
        If provided, figure is saved to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    histories = result.get('convergence_histories', [])
    best_idx  = result.get('best_seed_index', 0)
    best_ll   = result.get('log_likelihood', None)

    final_lls = [hist[-1] if hist else -np.inf for hist in histories]
    n_seeds   = len(histories)
    title_sfx = f' — {subject_id}' if subject_id else ''

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: per-iteration LL
    ax = axes[0]
    for i, hist in enumerate(histories):
        if not hist:
            continue
        is_best = (i == best_idx)
        ax.plot(hist,
                color='steelblue' if is_best else 'lightgrey',
                linewidth=2.0     if is_best else 0.8,
                alpha=1.0         if is_best else 0.6,
                zorder=2          if is_best else 1,
                label=f'Seed {i} (best)' if is_best else f'Seed {i}')
    ax.set_xlabel('EM Iteration', fontsize=11)
    ax.set_ylabel('Log-Likelihood', fontsize=11)
    ax.set_title(f'LL Convergence per Seed{title_sfx}', fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)

    # Right: final LL per seed
    ax = axes[1]
    colors = ['steelblue' if i == best_idx else 'lightgrey' for i in range(n_seeds)]
    ax.bar(range(n_seeds), final_lls, color=colors, edgecolor='grey', linewidth=0.5)
    if best_ll is not None:
        ax.axhline(best_ll, color='steelblue', linestyle='--', linewidth=1.2,
                   label=f'Best LL = {best_ll:.2f}')
        ax.legend(fontsize=9)
    ax.set_xlabel('Seed Index', fontsize=11)
    ax.set_ylabel('Final Log-Likelihood', fontsize=11)
    ax.set_title(f'Final LL by Seed{title_sfx}', fontsize=12)
    ax.set_xticks(range(n_seeds))
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
