# TCS: Trajectory Consistency-based Stopping

**Halt before Deviate! Adaptive Local Effort in Federated Learning**

Seok-Ju Hahn, Kibaek Kim

*Let the local trajectory decide when to synchronize.*

Official APPFL-native implementation. The paper link will be added when available.

## Abstract

The number of local steps is an important control knob in federated learning, shaping both communication cost and the optimization trajectory. Choosing this count in advance is challenging because its effect on communication and global progress depends on data heterogeneity. This motivates adapting local step counts during training to reduce communication while preserving accuracy. We propose an adaptive stopping rule that selects each client's local step count by comparing both the direction and magnitude of its average update with a reference formed early in the round. The rule uses signals available from local SGD, without additional model evaluations or communication during local training. Our analysis explains how the rule responds to differences in local curvature and relates local alignment to global descent. Under a fixed budget of total local updates, we empirically demonstrate that our rule can reduce communication rounds with accuracy close to the best fixed count. These findings show how local trajectory consistency can guide the allocation of computation between synchronizations.

## Method

Each client starts at the global model and runs plain SGD with a learning rate fixed within the round. After the base steps, it fixes a reference:

$$d_h=(\theta_t-\theta_h)/(\eta_t h),\qquad r=d_{H_{\min}}.$$

Another local step is allowed only while

$$\psi(d_h)=\lVert d_h\rVert-\lVert d_h-r\rVert-\kappa\lVert r\rVert>0.$$

The first failed check stops training **without undoing that step**. The global first round uses only base steps. A client's first participation in a later round does not restart this warm-up.

The server uses the realized counts, not the configured cap:

$$p_i=\frac{n_i}{\sum_{j\in S_t}n_j},\quad \bar H_t=\sum_{i\in S_t}p_iH_i,\quad
\theta_{t+1}=\theta_t+\bar H_t\sum_{i\in S_t}p_i\frac{\theta_i-\theta_t}{H_i}.$$

Equal counts recover sample-weighted FedAvg. Different counts require this work normalization.

## Install

Python 3.11 is the tested environment. From this repository:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-reproduce.txt
pip install -e '.[examples,test]'
pytest -q
```

`requirements-reproduce.txt` pins APPFL to the tested upstream commit. No APPFL source patch is needed.

## APPFL integration

```python
from tcs import register
register()  # once per process, before constructing APPFL agents

from appfl.agent import ClientAgent, ServerAgent
```

Use the normal APPFL YAML syntax:

```yaml
client_configs:
  train_configs:
    trainer: TCSTrainer
    mode: step
    num_base_steps: 10
    num_local_steps: 100
    kappa: 0.5
    optim: SGD
    optim_args: {lr: 0.1}
server_configs:
  scheduler: SyncScheduler
  aggregator: TCSAggregator
```

The complete model, data, loss and communication configurations are in [`configs/`](configs/).

`TCSTrainer` returns a model state and metadata (`round`, `num_steps`, `num_samples`, `learning_rate`, stopping diagnostics). `SyncScheduler` collects these per client; `TCSAggregator` validates and combines them. Responses include the next global `round`; pass response metadata to `client_agent.train(**metadata)`, including on initial model download.

**Step mode is recommended.** It checks every minibatch after `num_base_steps`; `num_local_steps` is an upper bound. Data is reshuffled on exhaustion. The optional runtime `local_steps` argument can tighten the cap for a residual work budget, but cannot shorten the reference phase.

**Epoch mode** uses one complete local dataloader pass as its base:

```yaml
mode: epoch
num_local_epochs: 10  # maximum, not a mandatory duration
```

The reference is fixed after the first epoch. Every subsequent minibatch is checked: 1.2 or 1.8 epochs are possible, as are 2, 5 or 10 epochs if the rule continues. The global first round uses one epoch. Epoch lengths may differ across clients; aggregation always uses actual optimizer-step counts. `num_base_steps` does not override the one-epoch base in this mode.

The paper path uses zero-momentum SGD, no weight decay, clipping or DP. The stopping signal covers trainable parameters. Floating buffers are sample-weighted separately; nonfloating buffers must agree across clients. Stateful-buffer models are not used in the paper's MNIST experiment.

## MNIST

The authors specified two hidden layers: `Flatten → Linear(784,200) → ReLU → Linear(200,200) → ReLU → Linear(200,10)` (logits). Inputs are float pixels in `[0,1]`, without additional standardization.

```bash
# Quick execution check
python examples/mnist.py --clients 50 --work 1000 --output output/smoke

# Paper-sized TCS run
python examples/mnist.py --alpha 0.1 --kappa 0.7 --seed 1333319856 --output output/tcs

# Paired comparison: 3 partitions × 3 seeds × (5 TCS thresholds + 2 fixed baselines)
python examples/suite.py --workers 3
```

The comparison uses 500 clients, 5 sampled uniformly without replacement per round, minibatches of 10, and 50,000 total client optimizer updates. SGD starts at 0.1 and halves after 25,000 and 37,500 completed updates, at the next round boundary. TCS uses `[10,100]` and sweeps `κ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}`; Fixed-10/100 use their fixed counts from round zero. A common residual cap prevents overshooting the budget; participation is never reduced to spend the remainder.

Partition, initialization and round participation seeds are paired across methods. Client minibatch seeds depend on `(seed, round, client)`. Class-wise Dirichlet allocations use `α∈{0.01,0.1,1}`. Empty allocations receive one sample from the largest client; no images are duplicated or discarded. Training includes a short last minibatch. The original experiment's partition manifest is not available here, so this implementation records its own partition and initialization hashes and does not claim bitwise reproduction of manuscript tables.

Logs include actual work, rounds, learning rates, local counts, stopping reasons and official test-set metrics. Test evaluation is separate from the stopping rule. [`results/mnist.json`](results/mnist.json) records the completed comparison; results are summarized below.

## Recorded results

We report official MNIST test accuracy at the selected TCS settings: κ=0.9 for α=0.01, κ=0.7 for α=0.1, and κ=0.1 for α=1.0. Accuracy is mean ± sample SD (%), and rounds are averaged over the same seeds. All runs used a 50,000-update ceiling.

| Dirichlet α | Method | Selected κ | Test accuracy (%) | Mean rounds |
|---:|---|---:|---:|---:|
| 0.01 | TCS | 0.9 | 93.41 ± 0.21 | 688.0 |
| 0.01 | Fixed-10 | — | 94.89 ± 1.18 | 1000.0 |
| 0.01 | Fixed-100 | — | 65.09 ± 10.09 | 100.0 |
| 0.1 | TCS | 0.7 | 96.83 ± 0.12 | 777.0 |
| 0.1 | Fixed-10 | — | 97.05 ± 0.31 | 1000.0 |
| 0.1 | Fixed-100 | — | 92.92 ± 1.57 | 100.0 |
| 1.0 | TCS | 0.1 | 97.58 ± 0.09 | 514.7 |
| 1.0 | Fixed-10 | — | 97.79 ± 0.02 | 1000.0 |
| 1.0 | Fixed-100 | — | 96.01 ± 0.17 | 100.0 |

The [full sweep](results/mnist.json) includes every tested κ, seed-level results and both fixed baselines.

Validation: 52 standalone tests; 78 focused/upstream regression tests; unchanged APPFL serial script; and a [three-process gRPC check](results/grpc.json) with 20 matched uploads and 20 matched downloads. The gRPC check is a short transport test, not the fixed-work comparison.

## Real gRPC execution

From the repository root, run these in three terminals:

```bash
python examples/run_grpc.py server --config configs/server.yaml
python examples/run_grpc.py client --config configs/client1.yaml
python examples/run_grpc.py client --config configs/client2.yaml
```

Or launch and verify all three processes:

```bash
python examples/check_grpc.py
```

This is a 2-client, 10-round MNIST transport check, distinct from the fixed-work paper comparison. It checks upload/download hashes, round metadata, initial synchronization and normal termination. Clients load the final `DONE` response before evaluating it. The supplied configurations bind localhost with TLS disabled; APPFL's existing TLS/authentication options remain available.

## Upstream integration

This repository contains the paper resources, baselines and experiments. [APPFL draft PR #482](https://github.com/APPFL/APPFL/pull/482) contributes the two native components, focused mathematical/integration tests and configuration documentation. It does not copy the experiment framework into APPFL. After upstream integration, `register()` is unnecessary.

## Citation

Paper link and publication metadata forthcoming.

```bibtex
@misc{hahn_tcs,
  title  = {Halt before Deviate! Adaptive Local Effort in Federated Learning},
  author = {Hahn, Seok-Ju and Kim, Kibaek},
  note   = {Manuscript; paper link forthcoming}
}
```
