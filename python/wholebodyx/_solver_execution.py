"""Coordinate solver libraries that install a process-wide SIGINT handler.

The Python and native OSQP versions have separate C globals. A common Python
lock covers the adapters' Python OSQP solve and complete native step/solve calls,
including their C++ formulation and result checks. Python-side preparation and
result conversion stay outside. Direct C++ callers also have a kernel guard.
Each instance still needs one owner across solve, reset and formulation.
"""

from threading import RLock

OSQP_EXECUTION_LOCK = RLock()
