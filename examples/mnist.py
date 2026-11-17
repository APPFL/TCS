"""Fixed-work MNIST experiment using APPFL's SyncScheduler and TCS components."""

import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Subset
from appfl.algorithm.scheduler import SyncScheduler
from tcs import TCSTrainer, TCSAggregator
from tcs.mnist import MNISTMLP, load_mnist, state_hash, evaluate


class NormOnlyTrainer(TCSTrainer):
    """Magnitude-only ablation, outside the upstream TCS implementation."""

    def _score(self, direction, reference):
        return (
            torch.linalg.vector_norm(direction)
            - (1 + self.kappa) * torch.linalg.vector_norm(reference) / 2
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=["tcs", "norm_only", "fixed10", "fixed100"], default="tcs"
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clients", type=int, default=500)
    parser.add_argument("--participants", type=int, default=5)
    parser.add_argument("--work", type=int, default=50000)
    parser.add_argument("--base", type=int, default=10)
    parser.add_argument("--cap", type=int, default=100)
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--data", default="datasets")
    parser.add_argument("--output", default="output/mnist")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source_hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in [
            "src/tcs/trainer.py",
            "src/tcs/aggregator.py",
            "src/tcs/mnist.py",
            "examples/mnist.py",
        ]
    }
    if not 1 <= args.participants <= args.clients or args.work < 1:
        parser.error("invalid participation or work budget")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "final.json").exists():
        raise FileExistsError("use a fresh output directory")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    train, test, groups = load_mnist(args.data, args.clients, args.alpha, args.seed)
    model = MNISTMLP()
    initial_hash = state_hash(model.state_dict())
    partition_hash = hashlib.sha256(json.dumps(groups).encode()).hexdigest()
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    aggregator = TCSAggregator(model)
    scheduler = SyncScheduler(
        OmegaConf.create({"num_clients": args.participants, "optimize_memory": False}),
        aggregator,
        None,
    )
    participation = np.random.default_rng(args.seed + 1)
    loss_fn = torch.nn.CrossEntropyLoss()
    work, round_id = 0, 0
    started = time.monotonic()
    next_eval = max(1, args.work // 10)
    interval = next_eval
    with (
        (out / "rounds.jsonl").open("w", buffering=1) as rounds,
        (out / "clients.jsonl").open("w", buffering=1) as clients,
        (out / "metrics.jsonl").open("w", buffering=1) as metrics,
    ):

        def evaluation():
            result = dict(round=round_id, work=work, **evaluate(model, test))
            metrics.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)
            return result

        evaluation()
        while True:
            residual_cap = (args.work - work) // args.participants
            fixed = (
                int(args.method.removeprefix("fixed"))
                if args.method.startswith("fixed")
                else None
            )
            base = fixed or args.base
            cap = fixed or min(args.cap, residual_cap)
            if residual_cap < base:
                break
            lr = 0.1 * (
                0.5 ** (int(work >= args.work * 0.5) + int(work >= args.work * 0.75))
            )
            selected = participation.choice(
                args.clients, args.participants, replace=False
            ).tolist()
            global_state, metadata = aggregator.get_parameters()
            futures, total_steps = [], 0
            for cid in selected:
                # Independent stream per (seed, global round, client), paired across methods.
                torch.manual_seed(
                    int(
                        np.random.SeedSequence(
                            [args.seed, round_id, cid]
                        ).generate_state(1)[0]
                    )
                )
                local_model = copy.deepcopy(model)
                cfg = OmegaConf.create(
                    dict(
                        mode="step",
                        num_base_steps=base,
                        num_local_steps=cap,
                        kappa=args.kappa,
                        optim="SGD",
                        optim_args={"lr": lr},
                        train_batch_size=args.batch_size,
                    )
                )
                cls = NormOnlyTrainer if args.method == "norm_only" else TCSTrainer
                trainer = cls(
                    model=local_model,
                    loss_fn=loss_fn,
                    train_dataset=Subset(train, groups[cid]),
                    train_configs=cfg,
                )
                trainer.load_parameters(global_state)
                trainer.train(round=0 if fixed else round_id, learning_rate=lr)
                local, info = trainer.get_parameters()
                info["round"] = round_id
                if fixed:
                    info["stop_reason"] = "fixed"
                total_steps += info["num_steps"]
                clients.write(json.dumps(dict(client=cid, **info)) + "\n")
                futures.append(scheduler.schedule(cid, local, **info))
            assert all(f.done() for f in futures)
            assert total_steps <= args.work - work
            work += total_steps
            round_id += 1
            rounds.write(
                json.dumps(
                    dict(
                        round=round_id,
                        work=work,
                        local_steps=total_steps,
                        participants=selected,
                        learning_rate=lr,
                        state_hash=state_hash(model.state_dict()),
                    )
                )
                + "\n"
            )
            if work >= next_eval:
                evaluation()
                next_eval = (work // interval + 1) * interval
        result = evaluation()
    result.update(
        status="COMPLETED",
        elapsed_seconds=time.monotonic() - started,
        initial_model_hash=initial_hash,
        partition_hash=partition_hash,
        python=platform.python_version(),
        torch=torch.__version__,
        source_hashes=source_hashes,
    )
    (out / "final.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
