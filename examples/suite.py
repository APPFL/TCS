"""Run the 63-run paired MNIST comparison and audit its logs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


KAPPAS = (0.1, 0.3, 0.5, 0.7, 0.9)
SETTINGS = [("tcs", k) for k in KAPPAS] + [("fixed10", None), ("fixed100", None)]


def run(condition, data, output):
    alpha, seed, method, kappa = condition
    dest = output / f"alpha{alpha}_seed{seed}_{method}"
    if kappa is not None:
        dest = dest.with_name(f"{dest.name}_k{kappa}")
    if not (dest / "final.json").exists():
        dest.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "examples/mnist.py"),
            "--method",
            method,
            "--alpha",
            str(alpha),
            "--seed",
            str(seed),
            "--data",
            data,
            "--output",
            str(dest),
        ]
        if kappa is not None:
            cmd.extend(["--kappa", str(kappa)])
        with (dest / "run.log").open("w") as log:
            subprocess.run(
                cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
            )
    config = json.loads((dest / "config.json").read_text())
    assert (
        config["work"] == 50000
        and config["clients"] == 500
        and config["participants"] == 5
    )
    assert (config["alpha"], config["seed"], config["method"]) == condition[:3]
    assert kappa is None or config["kappa"] == kappa
    assert (config["base"], config["cap"], config["batch_size"]) == (10, 100, 10)
    final = json.loads((dest / "final.json").read_text())
    expected = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in [
            "src/tcs/trainer.py",
            "src/tcs/aggregator.py",
            "src/tcs/mnist.py",
            "examples/mnist.py",
        ]
    }
    assert final.get("source_hashes") == expected, (
        "stale results: use a fresh --output directory"
    )
    rounds = [json.loads(s) for s in (dest / "rounds.jsonl").read_text().splitlines()]
    clients = [json.loads(s) for s in (dest / "clients.jsonl").read_text().splitlines()]
    assert final["status"] == "COMPLETED" and final["count"] == 10000
    assert len(clients) == 5 * len(rounds) and len(rounds) == final["round"]
    work = 0
    for i, r in enumerate(rounds):
        records = clients[5 * i : 5 * i + 5]
        lr = 0.1 * 0.5 ** (int(work >= 25000) + int(work >= 37500))
        assert len(set(r["participants"])) == 5
        assert [c["client"] for c in records] == r["participants"]
        for c in records:
            assert c["round"] == i and c["learning_rate"] == lr
            if method.startswith("fixed"):
                assert c["num_steps"] == int(method[5:])
            else:
                assert 10 <= c["num_steps"] <= min(100, (50000 - work) // 5)
                if i == 0:
                    assert c["num_steps"] == 10
        work += sum(c["num_steps"] for c in records)
        assert work == r["work"] <= 50000
    assert work == final["work"]
    base = int(method[5:]) if method.startswith("fixed") else 10
    assert 50000 - work < 5 * base
    stops = [c for c in clients if c["stop_reason"] == "trajectory"]
    final.update(
        alpha=alpha,
        seed=seed,
        method=method,
        kappa=kappa,
        mean_local_steps=work / len(clients),
        trajectory_stops=len(stops),
        direction_necessary=sum(
            c["norm_ratio"] is not None and c["norm_ratio"] > (1 + kappa) / 2
            for c in stops
        ),
        participation_hash=hashlib.sha256(
            json.dumps([r["participants"] for r in rounds]).encode()
        ).hexdigest(),
        participation_prefix_hash=hashlib.sha256(
            json.dumps([r["participants"] for r in rounds[:100]]).encode()
        ).hexdigest(),
    )
    print(
        f"DONE {dest.name}: {final['accuracy']:.2f}% / {final['round']} rounds",
        flush=True,
    )
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets")
    parser.add_argument("--output", default="output/paired")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(output / "mpl-cache"))
    conditions = [
        (a, s, m, k)
        for a in [0.01, 0.1, 1.0]
        for s in [1333319856, 867288040, 2238995067]
        for m, k in SETTINGS
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [
            pool.submit(run, c, str(Path(args.data).resolve()), output)
            for c in conditions
        ]
        results = [f.result() for f in as_completed(jobs)]
    for a in [0.01, 0.1, 1.0]:
        for s in [1333319856, 867288040, 2238995067]:
            paired = [r for r in results if r["alpha"] == a and r["seed"] == s]
            assert len({r["initial_model_hash"] for r in paired}) == 1
            assert len({r["partition_hash"] for r in paired}) == 1
            assert len({r["participation_prefix_hash"] for r in paired}) == 1
    means = []
    for a in [0.01, 0.1, 1.0]:
        for m, k in SETTINGS:
            rows = [
                r
                for r in results
                if r["alpha"] == a and r["method"] == m and r["kappa"] == k
            ]
            means.append(
                dict(
                    alpha=a,
                    method=m,
                    kappa=k,
                    accuracy_mean=statistics.mean(r["accuracy"] for r in rows),
                    accuracy_sd=statistics.stdev(r["accuracy"] for r in rows),
                    rounds_mean=statistics.mean(r["round"] for r in rows),
                    work_mean=statistics.mean(r["work"] for r in rows),
                )
            )
    artifact = dict(
        protocol="500 clients; 5 per round; 50k optimizer steps; batch 10; kappa grid 0.1/0.3/0.5/0.7/0.9; H=[10,100]",
        source_hashes={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(
                list((ROOT / "src").rglob("*.py"))
                + list((ROOT / "examples").glob("*.py"))
            )
        },
        runs=sorted(
            results, key=lambda r: (r["alpha"], r["seed"], r["method"], r["kappa"] or 0)
        ),
        summary=means,
        best_tcs=[
            max(
                (r for r in means if r["method"] == "tcs" and r["alpha"] == a),
                key=lambda r: r["accuracy_mean"],
            )
            for a in [0.01, 0.1, 1.0]
        ],
    )
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/mnist.json").write_text(json.dumps(artifact, indent=2))
    print(json.dumps(means, indent=2))


if __name__ == "__main__":
    main()
