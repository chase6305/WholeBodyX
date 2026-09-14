"""Detect a simulated joint-state offset and replan from the latest measurement.

Run: python examples/python/disturbance_recovery.py --viewer
This injects a kinematic position jump; it does not simulate external force.
"""

from _common import main

if __name__ == "__main__":
    main("disturbance_recovery")
