"""10 Hz joint MPC -> 50 Hz WBC, with target changes at 2 s and 4 s.

Run: python examples/python/mpc_retargeting.py --mpc-backend native --viewer
"""

from _common import main

if __name__ == "__main__":
    main("mpc_retargeting")
