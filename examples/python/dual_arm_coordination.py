"""Center motion + relative-position task, with weak posture regularization.

Run: python examples/python/dual_arm_coordination.py --csv /tmp/coordination.csv
The relative position is a weighted task; no object dynamics or rigid grasp.
"""

from _common import main

if __name__ == "__main__":
    main("coordination")
