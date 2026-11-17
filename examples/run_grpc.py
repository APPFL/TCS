"""APPFL gRPC server/client entry points, with model hashes for verification."""

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from appfl.agent import ClientAgent, ServerAgent
from appfl.comm.grpc import GRPCClientCommunicator, GRPCServerCommunicator, serve
from tcs import register
from tcs.mnist import state_hash, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["server", "client"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--data")
    parser.add_argument("--address")
    parser.add_argument("--output", default="output/grpc")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    register()
    config = OmegaConf.load(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.role == "server":
        grpc_cfg = config.server_configs.comm_configs.grpc_configs
        if args.address:
            grpc_cfg.server_uri = args.address
        agent = ServerAgent(config)
        with (out / "server.jsonl").open("w", buffering=1) as log:
            log.write(
                json.dumps(
                    dict(event="initial", sha256=state_hash(agent.model.state_dict()))
                )
                + "\n"
            )
            aggregate = agent.aggregator.aggregate

            def traced(models, **kwargs):
                inputs = {cid: state_hash(m) for cid, m in models.items()}
                result = aggregate(models, **kwargs)
                log.write(
                    json.dumps(
                        dict(
                            event="aggregate",
                            round=agent.aggregator.round,
                            inputs=inputs,
                            sha256=state_hash(result[0]),
                            steps=kwargs["num_steps"],
                        )
                    )
                    + "\n"
                )
                return result

            agent.aggregator.aggregate = traced
            serve(
                GRPCServerCommunicator(agent, logger=agent.logger, **grpc_cfg),
                **grpc_cfg,
            )
    else:
        if args.data:
            config.data_configs.dataset_kwargs.root = args.data
        if args.address:
            config.comm_configs.grpc_configs.server_uri = args.address
        agent = ClientAgent(config)
        comm = GRPCClientCommunicator(
            client_id=agent.get_id(),
            logger=agent.logger,
            **config.comm_configs.grpc_configs,
        )
        agent.load_config(comm.get_configuration())
        model, metadata = comm.get_global_model(init_model=True)
        agent.load_parameters(model)
        comm.invoke_custom_action(
            "set_sample_size", sample_size=agent.get_sample_size()
        )
        with (out / f"{agent.get_id()}.jsonl").open("w", buffering=1) as log:
            log.write(
                json.dumps(dict(event="initial", sha256=state_hash(model))) + "\n"
            )
            while True:
                agent.train(**metadata)
                local, info = agent.get_parameters()
                model, metadata = comm.update_global_model(local, **info)
                agent.load_parameters(model)
                assert state_hash(agent.model.state_dict()) == state_hash(model)
                log.write(
                    json.dumps(
                        dict(
                            event="update",
                            round=metadata["round"],
                            upload=state_hash(local),
                            download=state_hash(model),
                            status=metadata["status"],
                            **{"training": info},
                        )
                    )
                    + "\n"
                )
                if metadata["status"] == "DONE":
                    break
            result = evaluate(agent.model, agent.val_dataset)
            log.write(json.dumps(dict(event="final", **result)) + "\n")
        comm.invoke_custom_action("close_connection")


if __name__ == "__main__":
    main()
