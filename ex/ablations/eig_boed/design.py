"""map optuna's 2 unconstrained floats [-1,1]^2 to a unit vector on the upper
hemisphere of S^2 via shirley-chiu concentric square-to-disk composed with
inverse lambert azimuthal equal-area. bijective, area-preserving, no rejection/
singularity/wrap-around; quotients antipodal symmetry. this is d=3 only;
a different chart is needed for higher d."""

import numpy as np
from scipy.stats import qmc
import optuna

SQRT2 = np.sqrt(2.0)


def _sq_to_disk(a, b):
	"""shirley-chiu concentric map: [-1,1]^2 -> unit disk, area-preserving.

	scatters the square uniformly to the disk via concentric annuli.
	inputs: python floats; outputs: tuple of floats (u, v).
	"""
	if a == 0.0 and b == 0.0:
		return 0.0, 0.0

	if a*a > b*b:
		r = a
		phi = (np.pi / 4.0) * (b / a)
	else:
		r = b
		phi = (np.pi / 2.0) - (np.pi / 4.0) * (a / b)

	u = r * np.cos(phi)
	v = r * np.sin(phi)
	return u, v


def _disk_to_hemi(u, v):
	"""inverse lambert azimuthal equal-area: unit disk -> upper hemisphere of S^2.

	scales the unit disk to lambert radius sqrt(2) (rho=sqrt(2) is the equator).
	inputs: floats u, v in [-1,1]; output: np.ndarray (3,) on upper hemisphere.
	"""
	X = u * SQRT2
	Y = v * SQRT2
	rho2 = X*X + Y*Y
	s = np.sqrt(max(0.0, 1.0 - rho2/4.0))
	z = 1.0 - rho2/2.0
	z = max(z, 0.0)
	return np.array([s*X, s*Y, z], dtype=np.float64)


def chart(a, b):
	"""compose sq_to_disk and disk_to_hemi: [-1,1]^2 -> upper hemisphere S^2.

	at a==b==0, returns the pole [0,0,1] (explicit branch via _sq_to_disk).
	unit-norm guaranteed by construction (inverse lambert is isometric).
	the output (x, y, z) satisfies x^2 + y^2 + z^2 = 1 by construction
	(shirley-chiu maps to unit disk, inverse lambert projects disk to S^2).
	inputs: python floats a, b; output: np.ndarray (3,), dtype float64.
	"""
	u, v = _sq_to_disk(a, b)
	return _disk_to_hemi(u, v)


def suggest(trial):
	"""optuna trial interface: draw 2 unconstrained floats, map to unit vector.

	inputs: optuna.Trial; outputs: np.ndarray (3,).
	(trial.params contains the history for replay.)
	"""
	a = trial.suggest_float("a", -1.0, 1.0)
	b = trial.suggest_float("b", -1.0, 1.0)
	return chart(a, b)


def sobol_startup(n_startup, seed):
	"""scrambled sobol sequence in [-1,1]^2 for startup designs.

	uniformly samples the square without rejection, no power-of-2 requirement.
	inputs: n_startup (int), seed (int | None); output: np.ndarray (n_startup, 2).
	rows: caller enqueues each as (a, b) -> chart(...) via suggest mechanism.
	"""
	sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
	seq = sampler.random(n_startup)
	seq_scaled = 2 * seq - 1
	return seq_scaled
