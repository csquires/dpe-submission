"""slippery gridworld MDP, value iteration, and softmax policy.

state indexing: s = i * L + j (row-major order).
actions: 0=up, 1=down, 2=left, 3=right.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import softmax


@dataclass
class MDP:
	"""tabular markov decision process.

	attributes:
		P: transition kernel [|S|, |A|, |S|] float64, row-stochastic in last dim.
		mu0: initial state distribution [|S|] float64, sums to 1.
		gamma: discount factor, float in [0, 1).
		L: grid side length (gridworld-specific metadata).
		n_states: total states = L * L.
		n_actions: 4 (up, down, left, right).
	"""
	P: np.ndarray
	mu0: np.ndarray
	gamma: float
	L: int
	n_states: int
	n_actions: int


def build_gridworld(
	L: int,
	p_slip: float,
	terminals: list[tuple[int, int]],
	gamma: float,
	mu0_kind: str = "uniform",
	mu0_centers: list[tuple[int, int]] | None = None,
) -> MDP:
	"""builds a slippery gridworld MDP on an L×L grid with absorbing terminal states.

	state indexing: s = i * L + j (row-major).
	actions: 0=up, 1=down, 2=left, 3=right.
	transitions: deterministic step T(s, a) clamps at walls; apply slip kernel.

	slip kernel: P(s'|s, a) = (1 - p_slip) * 1[s' = T(s,a)] + (p_slip/4) * (1 on any action from state s).
	- if p_slip=0, deterministic; if p_slip=1, uniformly random.

	terminal states: P[s, a, s] = 1 for all a if s ∈ terminals; self-loop with zero reward.

	initial distribution mu0:
		- "uniform" (default): uniform over non-terminal cells.
		- "distance_quadratic": mu0(s) ∝ Σ_c ||s − c||_2^2 for c in mu0_centers.
		  zero on terminals; renormalize. for two centers at opposite corners,
		  this concentrates mass on the off-diagonal corners (max sum of squared
		  distances) and is minimized at the midpoint between centers.

	args:
		L: grid side length.
		p_slip: slip probability in [0, 1].
		terminals: list of (i, j) tuples marking absorbing states.
		gamma: discount factor in [0, 1).
		mu0_kind: initial-distribution scheme; one of "uniform", "distance_quadratic".
		mu0_centers: required if mu0_kind == "distance_quadratic"; list of (i, j) cells.

	returns:
		MDP dataclass with P, mu0, gamma, L, n_states=L*L, n_actions=4.

	raises:
		ValueError: if L < 1, all cells terminal, p_slip out of range, or mu0
			args are inconsistent.
	"""
	# validation
	if L < 1:
		raise ValueError("L must be ≥ 1")
	if p_slip < 0 or p_slip > 1:
		raise ValueError("p_slip must be in [0, 1]")

	S = L * L
	A = 4

	# build deterministic transition table T_table[s, a] = next state under action a
	ss, aa = np.meshgrid(np.arange(S), np.arange(A), indexing='ij')
	ii, jj = ss // L, ss % L
	di = np.array([-1, 1, 0, 0])
	dj = np.array([0, 0, -1, 1])
	ii_next = np.clip(ii + di[aa], 0, L - 1)
	jj_next = np.clip(jj + dj[aa], 0, L - 1)
	T_table = ii_next * L + jj_next  # [S, A] int indices of next states

	# build slip kernel
	P = np.zeros((S, A, S), dtype=np.float64)

	# deterministic component: (1 - p_slip) mass on intended action
	P_det = np.zeros((S, A, S), dtype=np.float64)
	P_det[np.arange(S)[:, None], np.arange(A), T_table] = 1 - p_slip

	# slip component: independent of intended action `a`. for each state s, the slip
	# contribution at s' is (p_slip/4) * #{a' : T(s,a') == s'}. compute once per (s,s')
	# then broadcast across the action axis.
	slip_per_s = np.zeros((S, S), dtype=np.float64)
	for a_prime in range(4):
		np.add.at(slip_per_s, (np.arange(S), T_table[:, a_prime]), p_slip / 4)
	P_slip = np.broadcast_to(slip_per_s[:, None, :], (S, A, S)).copy()

	P = P_det + P_slip

	# handle terminal states: self-loop with certainty
	terminal_idxs = {i * L + j for (i, j) in terminals}
	if len(terminal_idxs) == S:
		raise ValueError("empty non-terminal set")

	for s_term in terminal_idxs:
		P[s_term, :, :] = 0.0
		P[s_term, :, s_term] = 1.0

	# build initial distribution
	if mu0_kind == "uniform":
		mu0 = np.ones(S, dtype=np.float64)
	elif mu0_kind == "distance_quadratic":
		if not mu0_centers:
			raise ValueError("mu0_kind='distance_quadratic' requires non-empty mu0_centers")
		ii_grid, jj_grid = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
		w = np.zeros((L, L), dtype=np.float64)
		for (ci, cj) in mu0_centers:
			w += (ii_grid - ci) ** 2 + (jj_grid - cj) ** 2
		mu0 = w.reshape(S)
	else:
		raise ValueError(f"unknown mu0_kind: {mu0_kind}")

	for s_term in terminal_idxs:
		mu0[s_term] = 0.0
	total = mu0.sum()
	if total <= 0:
		raise ValueError("mu0 has zero mass after zeroing terminals")
	mu0 /= total

	return MDP(P=P, mu0=mu0, gamma=gamma, L=L, n_states=S, n_actions=A)


def reward_to_goal(
	L: int,
	goal: tuple[int, int],
	terminals: list[tuple[int, int]]
) -> np.ndarray:
	"""constructs a reward matrix r[|S|, |A|] where r[s, a] = 1 if transition T(s, a) reaches goal, else 0.

	reward is earned upon entry to goal, not at goal.
	terminal cells always give zero reward (absorbing dynamics, no further movement possible).

	args:
		L: grid side length.
		goal: (i, j) tuple marking goal cell.
		terminals: list of (i, j) tuples marking absorbing states.

	returns:
		r [L*L, 4] float64 array; r[s, a] = 1 if action a from state s reaches goal, else 0.
	"""
	S = L * L
	A = 4

	# build deterministic transition table (same as in build_gridworld)
	ss, aa = np.meshgrid(np.arange(S), np.arange(A), indexing='ij')
	ii, jj = ss // L, ss % L
	di = np.array([-1, 1, 0, 0])
	dj = np.array([0, 0, -1, 1])
	ii_next = np.clip(ii + di[aa], 0, L - 1)
	jj_next = np.clip(jj + dj[aa], 0, L - 1)
	T_table = ii_next * L + jj_next

	# compute goal state index (clamped if out of bounds)
	goal_i = max(0, min(L - 1, goal[0]))
	goal_j = max(0, min(L - 1, goal[1]))
	goal_state = goal_i * L + goal_j

	# initialize reward: 1 at transitions reaching goal, else 0
	r = np.zeros((S, A), dtype=np.float64)
	r[T_table == goal_state] = 1.0

	# zero out terminals: they never give reward
	terminal_idxs = {i * L + j for (i, j) in terminals}
	for s_term in terminal_idxs:
		r[s_term, :] = 0.0

	return r


def shaped_reward_to_goal(
	L: int,
	goal: tuple[int, int],
	terminals: list[tuple[int, int]],
	kind: str = "sparse",
	sigma: float = 1.0,
) -> np.ndarray:
	"""build a reward matrix r[|S|, |A|] with either sparse or dense shape.

	"sparse": r(s, a) = 1 if T(s, a) == goal else 0 (delegates to reward_to_goal).
	"gaussian": r(s, a) = exp(-||T(s, a) - goal||_2^2 / sigma^2). dense gradient
	  toward goal across all cells; sigma controls bandwidth.

	terminal cells get zeroed (absorbing dynamics, no further reward).

	args:
		L: grid side length.
		goal: (i, j) tuple marking goal cell.
		terminals: list of (i, j) tuples marking absorbing states.
		kind: "sparse" or "gaussian".
		sigma: bandwidth for gaussian shaping; ignored for sparse.

	returns:
		r [L*L, 4] float64.
	"""
	if kind == "sparse":
		return reward_to_goal(L, goal, terminals)
	if kind != "gaussian":
		raise ValueError(f"unknown reward kind: {kind}")
	if sigma <= 0:
		raise ValueError(f"sigma must be > 0, got {sigma}")

	S = L * L
	A = 4
	ss, aa = np.meshgrid(np.arange(S), np.arange(A), indexing='ij')
	ii, jj = ss // L, ss % L
	di = np.array([-1, 1, 0, 0])
	dj = np.array([0, 0, -1, 1])
	ii_next = np.clip(ii + di[aa], 0, L - 1)
	jj_next = np.clip(jj + dj[aa], 0, L - 1)
	dist2 = (ii_next - goal[0]) ** 2 + (jj_next - goal[1]) ** 2
	r = np.exp(-dist2 / sigma ** 2).astype(np.float64)

	terminal_idxs = {i * L + j for (i, j) in terminals}
	for s_term in terminal_idxs:
		r[s_term, :] = 0.0

	return r


def value_iteration(
	P: np.ndarray,
	r: np.ndarray,
	gamma: float,
	tol: float = 1e-8,
	max_iter: int = 5000
) -> np.ndarray:
	"""solves the discounted MDP via tabular value iteration (Bellman backup on Q-values).

	standard Bellman: Q_{k+1}(s, a) = r(s, a) + γ * Σ_s' P(s'|s,a) max_a' Q_k(s', a').
	iterate until ||Q_{k+1} - Q_k||_∞ < tol or max_iter exhausted.

	args:
		P: transition kernel [S, A, S] float64, row-stochastic in last dim.
		r: reward matrix [S, A] float64.
		gamma: discount factor in [0, 1).
		tol: convergence tolerance (default 1e-8); must be > 0.
		max_iter: maximum iterations (default 5000).

	returns:
		Q [S, A] float64 array; optimal action-values.

	raises:
		ValueError: if tol <= 0.
		RuntimeError: if VI does not converge within max_iter.
	"""
	if tol <= 0:
		raise ValueError("tol must be > 0")

	S, A = P.shape[0], P.shape[1]
	Q = np.zeros((S, A), dtype=np.float64)

	for k in range(max_iter):
		# compute optimal state values
		V = np.max(Q, axis=1)  # [S]

		# bellman backup: Q[s,a] = r[s,a] + gamma * sum_{s'} P[s,a,s'] * V[s']
		Q_next = r + gamma * np.einsum('sat,t->sa', P, V)

		# check convergence
		residual = np.max(np.abs(Q_next - Q))
		if residual < tol:
			return Q_next

		Q = Q_next

	# did not converge
	raise RuntimeError(f"VI did not converge after {max_iter} iterations; final residual = {residual}")


def softmax_policy(
	Q: np.ndarray,
	tau: float
) -> np.ndarray:
	"""converts Q-values to a stochastic policy via softmax temperature.

	π(a|s) = exp(Q(s,a) / τ) / Σ_{a'} exp(Q(s,a') / τ).
	low tau (→ 0): sharp, nearly deterministic (argmax).
	high tau (→ ∞): nearly uniform.

	uses scipy.special.softmax for numerical stability (subtract-max internally).

	args:
		Q: action-values [S, A] float64.
		tau: temperature parameter; must be > 0.

	returns:
		pi [S, A] float64 array; row-stochastic policy (rows sum to 1).

	raises:
		ValueError: if tau <= 0.
	"""
	if tau <= 0:
		raise ValueError(f"tau must be > 0, got {tau}")

	# softmax applies max-subtract internally for stability
	pi = softmax(Q / tau, axis=1)

	return pi
