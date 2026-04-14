from __future__ import annotations

import argparse

from .config import EngineConfig
from .engine import FaceAuthEngine


def main() -> int:
    parser = argparse.ArgumentParser(prog="faceauth-engine")
    parser.add_argument("--user", required=False)
    parser.add_argument("--reason", required=False)
    parser.add_argument("--timeout-ms", type=int, required=False)
    parser.add_argument("--enroll", type=str, required=False)
    parser.add_argument("--db", type=str, required=False)
    args = parser.parse_args()

    config = EngineConfig(face_db_path=args.db) if args.db else EngineConfig()
    engine = FaceAuthEngine(config)

    if args.enroll:
        ok = engine.enroll(args.enroll)
        print("PASS" if ok else "ERROR")
        return 0

    print(engine.authenticate())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
