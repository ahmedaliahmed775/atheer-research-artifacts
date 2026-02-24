# Parameter Mapping — Paper Tables ↔ Code

This document maps every scenario parameter in `configs/paper.yml` and
`atheer_sim.py` to the corresponding entry in the paper's Table III.

Update this file whenever scenario parameters are changed in the paper.

---

## Table III mapping

| Paper Table III Column     | Code Field (`ScenarioConfig`)  | Unit    | S1 (Public) | S2 (Private APN) |
|----------------------------|-------------------------------|---------|-------------|-----------------|
| Mean one-way latency        | `latency_mean_s`              | seconds | 0.400       | 0.060           |
| Latency std-dev (jitter)   | `latency_std_s`               | seconds | 0.200       | 0.015           |
| Packet loss probability    | `packet_loss`                 | [0, 1]  | 0.10        | 0.001           |
| Max retransmission attempts | `retries`                     | count   | 2           | 1               |
| Queue timeout              | `queue_timeout_s`             | seconds | 12.0        | 4.0             |
| End-to-end timeout         | `e2e_timeout_s`               | seconds | 15.0        | 5.0             |
| Bank server concurrency    | `bank_capacity`               | slots   | 12          | 12 (same)       |
| Bank service time          | `service_time_s`              | seconds | 0.02        | 0.02 (same)     |
| Local processing time      | `local_time_s`                | seconds | 0.05        | 0.05 (same)     |
| Congestion degradation     | `enable_degradation`          | boolean | True        | False           |

> **Network-only comparison**: `bank_capacity`, `service_time_s`, and
> `local_time_s` are held **identical** across S1 and S2 so that all
> observed performance differences are attributable solely to the network
> path (Section VI, experimental design).

---

## Experiment constants

| Parameter         | Code Symbol       | Value | Notes                                      |
|-------------------|-------------------|-------|--------------------------------------------|
| Load points (TPS) | `LOAD_POINTS_TPS` | 1 … 50| x-axis for Fig. 6 and Fig. 7               |
| Warm-up period    | `WARMUP_S`        | 5 s   | Discarded; eliminates transient-state bias |
| Measurement window| `MEASURE_S`       | 60 s  | Statistics collected in this window        |
| Replications      | `NUM_RUNS`        | 5†    | Paper used 30; see `configs/paper.yml`     |
| Random seed       | `BASE_SEED`       | 20260211 | Fixed for full reproducibility          |

† `NUM_RUNS=5` in the code enables fast CI reproduction.  To replicate the
  paper's 30-replication results, set `NUM_RUNS = 30` in `atheer_sim.py`.
  Note: `configs/paper.yml` documents the paper value for reference only —
  it is **not read at runtime** and changing it has no effect on execution.
