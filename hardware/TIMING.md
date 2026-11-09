# Timing — Table 11 vs hardware records

## The two numbers in the same file

Every job record in `hardware/ibm/*.json` and `hardware/ionq/*.json`
contains BOTH numbers, with their scopes defined in the file
itself (`job_record.timing_scope`):

| Field | What it is | ICU example |
|---|---|---|
| `execution_time_s` | Total job execution phase for the whole evaluation run for that case study — all circuits / COBYLA iterations / kernel batches, transpilation, classical pre/post-processing, readout-error mitigation. | 253 s |
| `qpu_time_ms` | Mean per-circuit QPU execution time, Table 11. Excludes queueing, network, encoding, mitigation and classical post-processing. | 95 ± 8 ms |
| `queue_time_s` | Queue time before execution. | 512 s |
| `wall_clock_s` | `completed - submitted`. | 765 s |
| `other_overhead_s` | Wall-clock residual: `wall - queue - execution`. | 0 s |

For 4 of 5 jobs: `wall = queue + execution + other` with `other = 0`.
For the drug-target job: wall 823 s, queue 445 s, execution 278 s,
`other_overhead_s = 100 s`, recorded here as provider-side result
retrieval / post-processing time. It is recorded explicitly in the
job file rather than absorbed by changing a timestamp.

## Why 253 s and 95 ms are not a contradiction

`execution_time_s` is a *run* total; `qpu_time_ms` is a *per-circuit*
mean. A single ICU evaluation run executes the 6-qubit VQC many
times (COBYLA iterations, each evaluating the objective on the
training set, plus the final hardware evaluation at 1,000 shots).
253 s of total execution phase containing many ~95 ms QPU
executions is the expected order of magnitude. The two numbers
measure different quantities and are not directly comparable;
both are stored in the same file so that is visible.

## Scope

- The `qpu_time_ms` values are the manuscript's Table 11 reported
  values (mean ± SD). They are not re-derived from the provider
  JSONs in this package, because the provider responses supplied
  here do not contain a per-circuit QPU-time field. If the
  original provider usage records are available, add them as
  `provider_qpu_time_ms` and cite them in the manifest.
- Table 11's simulator fidelities (0.985–0.992) are paper-reported
  values. The package's ideal-Aer fidelities (0.98–1.00,
  `hardware/simulation/`) are computed by `run_simulation.py` and
  are labelled as ideal-Aer values in that file.
