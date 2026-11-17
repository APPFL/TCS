"""Launch three local processes and assert every transmitted model hash."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="datasets")
    p.add_argument("--output", default="output/grpc-check")
    args = p.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out / "mpl"))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    address = f"127.0.0.1:{port}"
    processes, logs = [], []
    start = time.monotonic()
    try:
        for role in ["server", "client1", "client2"]:
            log = (out / f"{role}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "examples/run_grpc.py",
                        "server" if role == "server" else "client",
                        "--config",
                        f"configs/{role}.yaml",
                        "--address",
                        address,
                        "--data",
                        str(Path(args.data).resolve()),
                        "--output",
                        str(out),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
            if role == "server":
                for _ in range(90):
                    if processes[0].poll() is not None:
                        raise RuntimeError("server startup failed")
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=1):
                            break
                    except OSError:
                        time.sleep(1)
                else:
                    raise TimeoutError("server startup")
        deadline = time.monotonic() + 300
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("gRPC process failed; inspect logs")
            if time.monotonic() > deadline:
                raise TimeoutError("gRPC run")
            time.sleep(1)
        records = {
            name: [
                json.loads(s) for s in (out / f"{name}.jsonl").read_text().splitlines()
            ]
            for name in ["server", "Client1", "Client2"]
        }
        server = records["server"]
        assert len(server) == 11
        for name in ["Client1", "Client2"]:
            client = records[name]
            assert len(client) == 12 and client[0]["sha256"] == server[0]["sha256"]
            for i in range(1, 11):
                assert client[i]["round"] == server[i]["round"] == i
                assert client[i]["upload"] == server[i]["inputs"][name]
                assert client[i]["download"] == server[i]["sha256"]
                assert client[i]["status"] == ("DONE" if i == 10 else "RUNNING")
                if i == 1:
                    assert client[i]["training"]["num_steps"] == 10
        assert records["Client1"][-1] == records["Client2"][-1]
        summary = dict(
            address=address,
            pids=[p.pid for p in processes],
            exit_codes=[p.returncode for p in processes],
            verified_rounds=10,
            verified_uploads=20,
            verified_downloads=20,
            final=records["Client1"][-1],
            elapsed_seconds=time.monotonic() - start,
            steps=[r["steps"] for r in server[1:]],
        )
        (out / "result.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        for f in logs:
            f.close()


if __name__ == "__main__":
    main()
