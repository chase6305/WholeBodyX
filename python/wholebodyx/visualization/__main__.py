import argparse

from ..experiments import SCENARIOS
from .viewer import ControlViewer


def main():
    parser = argparse.ArgumentParser(description="WholeBodyX browser visualization")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--backend", choices=("python", "native"), default="python")
    parser.add_argument("--mode", choices=("WBC", "MPC + WBC"), default="MPC + WBC")
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), default="tracking")
    parser.add_argument("--mpc-backend", choices=("python", "native"), default=None)
    args = parser.parse_args()
    viewer = ControlViewer(args.host, args.port, args.backend, args.mode, args.scenario, args.mpc_backend)
    viewer.run()


if __name__ == "__main__":
    main()
