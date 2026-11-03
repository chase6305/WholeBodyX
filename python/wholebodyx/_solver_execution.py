"""Coordinate solver libraries that install a process-wide SIGINT handler.

The Python and native OSQP versions have separate C globals. A common Python
lock covers both adapters; the C++ core also guards its own direct entry points.
Matrix construction and result validation remain outside this kernel boundary.
Instances still require a single owner: this lock does not protect their state.
"""

from threading import RLock

OSQP_EXECUTION_LOCK = RLock()
