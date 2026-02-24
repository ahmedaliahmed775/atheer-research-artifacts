# Model Assumptions — Section VI

This document records the modelling choices underlying the discrete-event
simulation in `atheer_sim.py`.  Update this file if the model is changed.

---

## Arrival process

Transactions arrive according to a **Poisson process** (exponentially
distributed inter-arrival times) at the configured offered load (TPS).
This is the standard assumption for independent, memoryless client traffic
in queueing-theoretic analyses.

## Network latency distribution

One-way network latency is drawn from a **LogNormal distribution**, which is
strictly positive and right-skewed — consistent with empirical round-trip-time
measurements on both cellular and fixed Internet links.  The distribution is
parameterised by a desired mean and standard deviation (see Table III);
`lognormal_mu_sigma()` converts these to NumPy's (mu, sigma) parameterisation.

## Packet loss model

Packet loss is modelled as an independent **Bernoulli trial** on each
transmission attempt with the configured per-attempt probability.  Failed
packets trigger a retransmission timeout equal to twice the sampled latency
before the next attempt.  A maximum number of retries is enforced
(`retries` field); exhausting all retries yields `FAILED_{DIR}_NETWORK`.

## Bank / back-end server

The bank server is modelled as a **finite-concurrency FIFO queue** (SimPy
`Resource`) with `bank_capacity` concurrent service slots and deterministic
service time `service_time_s`.  Transactions that wait longer than
`queue_timeout_s` are dropped with reason `FAILED_QUEUE_TIMEOUT`.

## End-to-end timeout

A hard wall-clock timeout (`e2e_timeout_s`) is enforced after each pipeline
stage.  Any transaction whose cumulative elapsed time exceeds this threshold
is failed with reason `FAILED_E2E_TIMEOUT`, regardless of which stage it is in.

## Congestion degradation (S1 only)

For the public-Internet scenario (S1), both latency and packet-loss increase
linearly with the normalised overload factor once the offered load exceeds
`degrade_threshold_tps`.  This models the well-documented behaviour of
congested shared Internet paths.  S2 (Private APN) is modelled without
degradation, reflecting the dedicated, QoS-controlled nature of the Atheer
network path.

## Network-only comparison

`bank_capacity`, `service_time_s`, and `local_time_s` are **identical** across
S1 and S2 so that any measured performance difference isolates the network
effect.  This is the experimental design described in Section VI of the paper.

## Warm-up / cool-down

Statistics are collected only within the `[WARMUP_S, WARMUP_S + MEASURE_S]`
window to eliminate transient-state bias (initial empty-queue transient) and
boundary effects (partially completed transactions at the measurement cutoff).
