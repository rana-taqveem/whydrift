"""whydrift: why do learned simulators drift?

Importing the package turns on float64 in JAX (see common.py). Physics,
data and the oracle are float64 by rule; training precision is [OPEN].
"""

from whydrift import common  # noqa: F401  (side effect: enables float64)
