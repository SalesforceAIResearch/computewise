# Registry record format, version 0.1

Written 2026-10-19 by session `gpu-tasks`. This is the contract between the GPU task registry and gpu-mock, which replays records on its fake GPUs. It is pinned to schema `p006-gpu-record/0.1` and `p006-gpu-task/0.1`. gpu-mock reviewed it on 2026-10-19 and accepted it. cpu-tasks asked for the host fields in §12 the same day. Fields added later are listed where they are defined. `build/validate.py` enforces what is checkable here. Where this file and that script disagree, the script is what ran; report the difference.

The agreements behind it are gpu-mock's notes in `experiments/gpu-mock/`: `REPLAY-INTERFACE-NOTE-2026-10-19.md` §6 and §8, `GPU-COUNT-CONVERSION-NOTE-2026-10-19.md` §7 and §8, and `docs/REVIEW-RESPONSE-2026-10-19.md`. The study's rulings on reference schedules are in `design/IDEA.md` under "Reference schedules".

## 1. Terms

- A **job** is one separately launched executable: a training run, an evaluation, one sweep member.
- A **record** is one recorded real run of a job: what it printed, how long it took on what hardware, how it ended. A record is the hidden side of a job; the agent never sees it.
- A **task** is what one agent run is handed: a brief, files, and the jobs the brief implies. A task's job points at the records that can serve it.
- A **replicate group** is the set of records of one configuration that differ only by seed.

## 2. Layout

```text
experiments/gpu-tasks/
├── RECORD-FORMAT.md   this contract
├── AGENTS.md          how to work here, and how to rebuild the registry
├── build/             generators and the validator (stdlib Python, run with python3 -I)
└── registry/
    ├── records/<record_id>.json
    ├── tasks/<task_id>.json
    ├── artifacts/<source>/scripts/<sha256[:12]>/<file>   the job scripts records point at
    └── CATALOG.md     generated index: counts, task table, blockers
```

Records and tasks are generated, never hand-edited. A record whose source changes is regenerated.

## 3. Provenance

Every value the registry did not copy verbatim says how it was obtained. A value with provenance is an object `{"value": ..., "provenance": ..., "how": "..."}`; `how` is free text naming the source field or the arithmetic.

| Provenance | Meaning |
|---|---|
| `recorded` | Read from the source as it stands: a log line, a W&B field, a leaderboard cell. |
| `derived` | Computed from recorded values by stated arithmetic, with no assumption beyond the arithmetic. |
| `inferred` | Rests on an assumption the `how` names, such as hardware stated in a README rather than the log. |
| `interpolated` | Read between recorded points. No current record uses it. |
| `modelled` | Produced by a model, such as an evaluation time from step time and token counts. |
| `absent` | Not available. `value` is null, and `how` says why when that is known. |

A null `value` with any provenance but `absent` is a contract violation unless the object also carries `null_means`, which says what the null stands for (for example `save_total_limit` null meaning every checkpoint is kept). A non-null value marked `absent` is always a violation.

## 4. A record

Top-level fields. Fields the daemon read before this contract existed (`identity`, `mode`, `reference`, `steps`, `per_step_gpu_s`, `memory`, `series`, `failures`) keep their meaning, so `experiments/gpu-mock/tests/sample_record.json` stays a valid subset.

| Field | What it holds |
|---|---|
| `schema` | `p006-gpu-record/0.1`. |
| `identity` | How a launched process is matched to this record (§5). |
| `mode` | `replay`, `model` or `refuse`, as in REPLAY-INTERFACE §2. Every current record is `replay`; readiness (§11) says whether replay can work yet. |
| `source` | Where the record came from: repository and commit, W&B project and run, path, licence, date read. |
| `reference` | The recorded hardware: `gpu`, `gpu_count`, each with provenance, plus `interconnect` and `precision`. Times in the record are on this hardware, unconverted. |
| `launch` | Launch shape for conversion (§7). |
| `steps`, `planned_steps` | Optimizer steps the run executed, and the steps it was configured for. |
| `per_step_gpu_s` | Steady time per optimizer step on the reference hardware, seconds; null when unknown. |
| `timing` | Duration by component (§6). |
| `memory` | `peak_allocated_bytes` (a plain number, for the daemon) and `peak` (the same value with provenance). |
| `series` | Values the job materialises, by role (§8). |
| `observed` | Recorded measurements kept for calibration only, never served: for example NVML utilisation from W&B. |
| `materialisation` | What the job reads from the GPU (`reads`), the series replay needs (`needs_series`), how gpu-mock serves it (`serving`, §11), and anything that blocks replay (`blockers`). |
| `result` | The job's final results, each with provenance. For TRL records `train_loss` is the last logged value, which the Trainer logs as the mean over its logging interval, and `trainer_train_loss` is the Trainer's end-of-run train_loss, the mean over every optimizer step. They differ widely for DPO, where the loss falls fast; each `how` says which it is. |
| `outcome` | `status`: `completed`, `crashed`, `failed`, `killed`, `truncated` or `timeout`; `provenance`, `how`; optional `suspect` when a recorded value looks wrong, such as a duration under a tenth of the same job's median across sources. A suspect record is kept as recorded and flagged, never corrected. |
| `failures` | Events replay must reproduce: `{mode, step, time_s, at, provenance, how}`, with `mode` one of `oom`, `xid`, `nan`, `hang`, `crash`, `killed` and `at` one of `during_setup`, `during_training`, `after_training`, `unknown`. A non-completed outcome has at least one entry. An entry also carries `cause`, a value with provenance: the last exception line of the run's console log where the source kept one. `concurrent_same_output_dir` and `concurrent_same_hub_repo` list other runs that shared the path or repository at the time; they show overlap only. `collision` applies the study's rule (`design/IDEA.md`): `class` is `local_output_dir_race`, `hub_repo_push_race` or `unknown`, and `matched` is true only when the logged cause names the shared path. `replay` is a value with provenance, `{value, how}`, whose `value` is `fire` or `suppress`. Read `replay.value`: comparing the whole field with a string fired every suppressed failure in gpu-mock's daemon until 2026-10-20. A matched collision is suppressed, because a collision in the original project is not the job's failure (study's ruling, `design/IDEA.md`, shared-state collisions), and so is a failure in a Hub push after training, because tasks never push. A suppressed failure also carries `replay.continuation`, saying how replay continues: `complete` when training had finished and the series cover the whole job, or `sibling_required` when the run died early and the record holds no values past that step. A `sibling_required` record carries a blocker, is not replay-ready on its own, and is never drawn by a task; a sibling of its replicate group serves instead. |
| `artefacts` | Files the job writes, such as checkpoints, with sizes where known. |
| `host` | CPU, RAM and disk profile for cpu-tasks' shared layer (§12). |
| `agent_view` | What the agent sees: `script` (a registry path, or null), `launch_command`, `inputs`, `recognisability`. |
| `disclosure` | The true duration and what an agent could be told (§10). |
| `readiness` | Whether gpu-mock can replay the record today (§11). |

## 5. Identity and matching

`identity` holds:

- `record_id`: unique, stable, and the file name.
- `kind`: `train` or `eval`.
- `entry_hash`: sha256 of the exact entry-script bytes, or null when the registry has not materialised them, with `entry_hash_how`.
- `argv`: the launch arguments, with the script's basename first. gpu-mock compares `argv[0]` by basename (GPU-COUNT-CONVERSION-NOTE §8). `argv_exempt` and `env_exempt` list the arguments and variables a launch may change and still match: logging, output paths, caches and the Hub push.
- `env`: launch environment variables that change behaviour, such as a seed variable.
- `seed`: a value with provenance.
- `replicate_group`, and `trajectory.config_hash` that keys it.
- `match_policy`, `argv_exempt`, `trajectory`, `fingerprint`, `fingerprint_inputs`.
- `consumes`: artefacts from other jobs this job reads, such as a checkpoint.
- `source_key`: source-specific keys, such as MLS-Bench's task, baseline and label.
- `derived_from`, on a derived record only: `kind` (`prefix`), the parent `record`, `steps` [1, k], `stop_after_step`, the parent's `entry_hash` and `replicate_group`, and the script `change` in words.

**Derived prefix records** (`build/gen_prefix.py`, since 2026-10-20; PACKAGES-2026-10-20.md §1.2). A prefix record is a job that runs the first k optimizer steps of a recorded modded-nanogpt run. Its script is the parent's with one change: a constant `stop_after_step = k` on the line after the loop bound, and the loop's end check also breaks at it. The schedule, data order, seed and every hyperparameter are the parent's, so its series are the parent's, cut at k. k is one of the parent's validation steps. Following gpu-mock's three conditions (2026-10-20), `timing.eval_s` is the parent's scaled to the validations kept, failures past k are dropped, and the record is `exact_only` in its own replicate group, keyed by the prefix script's bytes. `first_step_s`, `startup_in_train_s`, `steady_step_s` and `setup_s` are the parent's. Its `record_id` is the parent's plus `-stop<k>`. A parent is used only when it replays as briefed: ready, recorded on the image's torch, a script that reads no setting from the environment, and a seed from `--seed` or none. The validator re-derives each prefix script from its parent and refuses any other difference.

**Derived segment records** (`build/gen_segment.py`, PACKAGES-2026-10-20.md §8). A segment record is steps start+1..stop of a recorded run, by the parent's script with a save-and-resume change: `--stop-at N` saves `checkpoints/seed<SEED>-step<N>.pt` and exits, and `--resume PATH` loads it, restores each optimizer's `step_count`, and advances the data loader by the steps already trained. `derived_from.kind` is `segment` and `steps` is [start+1, stop]. The series are re-based to the segment's own steps 1..stop−start, because gpu-mock counts a process's optimizer steps from 0. A resumed segment's first validation is at its own step 0, with the parent's value at `start`. `argv` carries the segment's own arguments, so each segment is its own exact key and its own replicate group, and `identity.consumes` names the checkpoint it loads. `host.train_data.start_batch` tells `common.shard_load_steps` where its loader starts. `startup_in_train_s` and `first_step_s` are paid again. gpu-mock's save-then-resume round trip passed on 2026-10-20, and segment records are now under `registry/records/` (`build/gen_segment.py`'s `CHAIN_RECORDS`). `registry/candidates/segment-contra-soft-muon` stays, because it is gpu-mock's test input.

**Derived prepared records** (`build/gen_prep.py`, the preparation arm). In a prep package the agent writes each job's script itself, so the record is served by path, not by bytes. A prepared record is a prefix record's run with `match_policy` `path_bound` and a `bound_path`. gpu-mock serves it to any script whose resolved path ends with `/`+`bound_path`, whatever its bytes; an exact match on the reference's bytes and argv still comes first. `derived_from.kind` is `prepared`, and `via` names the prefix record. There are four paths, and the record-id suffix tells them apart:

| path | suffix | used for |
|---|---|---|
| `runs/<setup>/train_gpt.py` | `-prep` | one job per setup |
| `runs/<setup>/seed<N>/train_gpt.py` | `-prepseed` | a setup run with several seeds |
| `quick/<setup>/train_gpt.py` | `-prepq` | wave Q, one job per setup |
| `quick/<setup>/seed<N>/train_gpt.py` | `-prepqseed` | wave Q, several seeds |

A per-seed job lists its per-setup twin as well. The twin is the same recorded run with the same exact key, so a scripted preparer's exact match may draw either. Each per-setup record and each per-seed record is its own replicate group.

**Derived data-checked records** (`build/gen_dataprep.py`, the data-preparation arm). A data-checked record is a wave-Q prefix record's run, served to that prefix script with one more change inserted after its `code = f.read()` line: at start-up it checks that every data file the run reads (the validation shard and train shards 1..`shards_read`) exists with a complete header and body, and stops with the file's name if not. `derived_from.kind` is `data_checked`, `via` names the prefix record, and `files` lists what the check reads. `record` and `steps` stay the prefix's, so `common.served_unit` traces it to the recorded run. The record id is the prefix's plus `-dcheck`. It is `exact_only`, in a replicate group of its own per script. The validator re-derives the checked script from its prefix's and refuses any other difference.

**`rerun_siblings` on a task's jobs** (gpu-mock's rerun design (b), agreed 2026-10-21). An exact-seed modded-nanogpt job that is not a segment lists, in serving order, other recorded runs of the same script at its stop step and of its own record kind (`-prepq`, `-dcheck`, a plain prefix or full run). gpu-mock serves a job's primary record once per run; a later lookup of it under another job key, which is a rerun, gets the first unused sibling, served by id whatever its `bound_path`, and the primary again once the list is spent. Siblings come from runs no record of the task lists, and jobs that draw on one script's runs share them round-robin (`build/siblings.py`). The validator refuses a sibling of another kind or step count, one that overlaps a listed record, or one that overlaps another sibling. Segments have none: a resumed segment follows the run of the checkpoint it loads, which is gpu-mock's lineage rule. When siblings are later added for chains, sibling k of every part must be a segment of the same base run.

**The integrity rule.** Replay serves the run the agent actually selected. A launch whose configuration differs from every record in a way that changes the trajectory (learning rate, data, precision, global batch, the script's logic) is not served the nearest curve. It goes to gpu-mock's modelled path or is refused, and the daemon logs it. A zero-learning-rate control must never show a decreasing recorded loss.

**`trajectory`** is what determines the run apart from noise:

- `config_hash`: what the replicate group is keyed by, with `config_hash_how` stating its inputs. Two learning-rate values are two groups, wherever the value lives: in script bytes (track 3), in argv (TRL), or in a job script's environment defaults (MLS-Bench).
- `environment`: library versions and commits that matter.
- `lr_schedule`: `identity` (a string naming the schedule and its constants), `source`, and `trace` when the source logged the learning rate (below).
- `data`: dataset identity, the files, and the order they are read in.
- `precision`: a value with provenance.
- `global_batch_tokens` or `global_batch_sequences`.

**`match_policy`** says when a launch may be served by this record:

- `exact_only`: only when the script bytes, argv and env match exactly, with seed handled as below. Any other launch goes to the modelled path. This is the default for any record without a learning-rate trace.
- `structural_verified`: a launch that matches structurally may replay, but only while values gpu-mock observes at run time agree with the record:
  - the learning rate of every param group at each `optimizer.step()`, against `trajectory.lr_schedule.trace`;
  - arguments to `torch.manual_seed`;
  - data files opened, by path and size;
  - the autocast dtype;
  - the global batch: backward calls per step × micro-batch × world size.

  On the first mismatch the daemon switches the job to the modelled path and logs it. Argv may differ from the record only in the arguments listed in `argv_exempt`, which change logging and output paths, not the trajectory.

**Learning-rate trace** (`trajectory.lr_schedule.trace`):

- `steps` and `values` are equal-length lists; `axis` names the step counter, which is optimizer steps (`train/global_step` for the HF Trainer), never micro-steps.
- `alignment` is `after_step` or `at_step`, with its own provenance. `after_step` means the value logged at step k is the rate optimizer step k+1 uses; the HF Trainer logs this way, since it logs after `lr_scheduler.step()`. gpu-mock then compares the rate observed at the (k+1)-th `optimizer.step()` with `trace[k]`. `at_step` compares at step k. Where a run's first logged rate and its warmup arithmetic settle the alignment, provenance is `derived`; otherwise `inferred` from the Trainer's behaviour.
- A step absent from the trace places no constraint. A different logging cadence is therefore never a mismatch.
- One value applies to every param group unless `per_group` holds one list per param-group index.
- `tolerance` is relative 1e-6 or absolute 1e-12.

**Seeds and replicate groups.** The daemon draws from a replicate group without replacement, so launching the same configuration twice serves two different recorded runs; when a group is exhausted it recycles and logs it. It draws only from records whose `readiness.replay` is true. A group with no drawable record sends the launch to the modelled path, logged as undrawable.

**Lineage.** Some jobs are told apart only by a file they read: the lm-eval evaluation of each pretraining variant runs identical bytes with identical argv and non-exempt env, and differs only in which checkpoint `--checkpoint` names, a path the agent chose. Their records list the path flags in `argv_exempt` and carry a blocker until gpu-mock can resolve a loaded file to the job that wrote it; the natural place is its patched torch, which sees both the `torch.save` that wrote the checkpoint and the `torch.load` that reads it. Until then those records are not drawable, and the task's evaluation jobs run on the modelled path. A launch naming a seed with a record gets that record. A launch naming a seed with no record gets another seed's record from the same group, logged as `seed_substituted`. This is gpu-mock's decision (REVIEW-RESPONSE, "Replay-integrity contract"): a different seed changes only noise. A script with no seed control is nondeterministic, so a relaunch is a fresh draw. A script whose seed is a constant in its bytes is a different file per seed and is matched exactly.

**`fingerprint`** is gpu-mock's structural hash, defined in `experiments/gpu-mock/docs/FINGERPRINT-CANONICALISATION.md`. Its fields and their order are gpu-mock's, and this file does not restate them. The generators load gpu-mock's implementation, `experiments/gpu-mock/src/torch_patch/fingerprint.py`, rather than re-implementing it, and the validator recomputes every stored hash. Each record stores the version it was hashed under in `identity.fingerprint_version`. Since fp256/0.3 field 11 is the root module's parameter count, `fingerprint_inputs.num_parameters`: TRL records take it from `artefacts.model_parameters`; track-3 and MLS-Bench records carry null with a `num_parameters_how`, because their scripts define their own model classes and those records are exact_only, which the structural index never serves. When gpu-mock changes the version, stored hashes stay stale until the registry is rebuilt, and the validator says so. `fingerprint_inputs` holds the fields it reads, and `fingerprint_how` says where they came from. Two rules fix what the running job must report for the hashes to agree:

- `model` is the HF config's `model_type` when the root module has one (`gpt2`, `qwen2`, `opt`), and otherwise the root module's class name, lowercased (`gpt` for track 3 and MLS-Bench).
- `precision` is the mixed-precision mode: `bf16`, `fp16`, or `fp32` when neither is set.

A record whose fields cannot be read statically, such as a script that never assigns its optimizer to a named variable, carries no fingerprint, and `fingerprint_how` says why. Such a record is `exact_only`, so the entry hash matches it.

## 6. Timing

`timing.pacing` says what the duration fields can support:

- `per_step`: a steady step time from per-step or per-validation lines (track 3).
- `per_logged_interval`: wall time at each logged row (`wall_series`, W&B `_runtime` against the step).
- `total`: one whole-job wall time only (MLS-Bench).

Components, each a value with provenance: `train_time_s`, `steady_step_s`, `startup_in_train_s`, `first_step_s`, `setup_s`, `eval_s`, `checkpoint_s`, `wall_total_s`, and `wall_series` where the source has it. Only `pacing` and `wall_total_s` are required; a source without a component omits it or marks it `absent`. They are on the reference hardware and unconverted. The speed knob never touches them: gpu-mock dilates the job's clock instead (REPLAY-INTERFACE §8).

Source-specific facts that shape these fields:

- modded-nanogpt's `train_time` stops during validation and includes `torch.compile` warm-up at step 1, so compile is a derived component and validation is modelled.
- W&B `_runtime` is wall time from `wandb.init` to exit. When it is zero for a crashed run, the record uses `heartbeatAt` minus `createdAt`, marked `derived`.
- MLS-Bench's `elapsed_<label>` is the harness's wall clock around the whole command.

## 7. Conversion inputs

gpu-mock's `convert(serving, gpu_count, launch)` reads those three blocks. `launch.batch_policy` decides how a different GPU count changes the job (GPU-COUNT-CONVERSION-NOTE §7):

| `batch_policy` | When the GPU count changes | Conversion |
|---|---|---|
| `fixed_global_grow_microbatch` | The global batch is kept and each GPU's micro-batch grows | ElasticFlow's per-GPU-batch curve, 1.07–1.37× for 8 to 4 GPUs |
| `fixed_global_grow_accumulation` | The global batch is kept and each GPU runs more accumulation micro-steps | About 2× for 8 to 4 on NVLink; batching gains nothing |
| `fixed_per_device` | The per-device batch is kept, so the global batch and the step count change | The recorded curve does not apply; another record or the modelled path |

`launch` also carries `world_size`, `per_device_microbatch`, `accumulation_steps` and the global batch, each with provenance, plus `gpu_counts_supported` and source-declared `alternatives` (such as MLS-Bench's H200 profile).

## 8. Series

`series` maps a role (`val_loss`, `train_loss`, `grad_norm`, `learning_rate`, `eval_accuracy`, and source keys with `/` replaced by `.`) to `{"steps": [...], "values": [...], "provenance": ..., "how": ...}`. Steps are strictly increasing, values are finite numbers, and the step axis is named in `how`. The registry always writes this object form; the daemon also accepts a plain list as values at steps 0 to n−1.

Serving order for a value the job materialises is gpu-mock's (REPLAY-INTERFACE §8): a series by role, then a value derived from shapes and the memory model, then derivation from the loss as a last resort, each logged with the job's identity. Every rank of a job gets the same served value. A recorded `nan` failure fires on all ranks together.

## 9. A task

| Field | What it holds |
|---|---|
| `schema` | `p006-gpu-task/0.1`. |
| `task_id`, `rung`, `title` | Rung 1 is independent jobs; rung 2 has jobs that depend on results; rung 3 is open-ended and not yet in the registry. |
| `task_class`, `task_class_how` | The study's class, ruled in `design/IDEA.md` under "Reference schedules" on 2026-10-19: gaps are reported per class and never pooled across classes. `fan_out_required`: no job can use the whole box. `width_flexible`: some job can take the whole box. `control`: every job needs the whole box. The generator reads the class off the jobs' `gpu_options` (`build/common.py:task_class`) and the validator recomputes it. |
| `box` | The machine the task assumes: 4× H100 80GB with NVLink, gpu-mock's default. |
| `brief` | `text`, what the agent is told, and `provenance`. A brief names no benchmark and leaks no test framing. |
| `files` | What the agent's workspace holds: registry artefacts by `from`, or literal `content`. |
| `jobs` | Each with `job_id`, `label`, `records` (or null when a decision binds it), `replicate_group`, `draw` (`exact_seed`, `seed_substituted`, `without_replacement`), `seed`, `gpu_options` (GPU counts the job may use on the box), `recorded_gpus`, `batch_policy`, `after` (jobs that must finish first), `consumes` and `produces` (artefact paths), `launch`, `disclosure` (§10). A job a decision binds has `records: null`, `bind` and `candidate_pool`, the records of every candidate it could become; its disclosure estimate is taken over that pool. `bind` names the `decision` and `slot`, and then either a `seed` to draw from the winning candidate's replicate group, or `select_by`, field-value pairs on `identity` (such as `source_key.label`) that pick exactly one of the winning candidate's records. |
| `decisions` | Rung-2 branching. Each has `decision_id`, `after`, `rule` in words and `rule_spec` in fields (`metric`, `aggregate`, `direction`, `top`), `observable`, `launches` (the job IDs it binds), `slots`, `candidates` (each with its `records` and screen jobs), and `shape_independent_of_outcome` with `shape_how`. Per the study's ruling, a follow-up's resource shape does not depend on which outcome wins; the record set says what the online reference must assume. `recorded_outcome` holds the winners the rule picks from the recorded screen values when every screen job has exactly one record, and otherwise says why the outcome is open; a bound job of a resolved decision carries `recorded_winner`, which the task's `stats` use and its disclosure does not. |
| `shared_state` | What the study's shared-state ruling needs from a task: `default_shared`, the groups of jobs whose launch lines write one path by default; `by_design`, jobs that read a producer's output and so share its path legitimately; and `isolation`, the ordinary flag that gives each job its own path (TRL's `--output_dir`, MLS-Bench's `OUTPUT_DIR`), which the records exempt from matching (`argv_exempt`, `env_exempt`). Both references isolate shared paths; an agent that leaves them shared pays through the collision class, never through a recorded failure. |
| `stats` | GPU-seconds (the field is still called `recorded_gpu_seconds`), a lower-bound makespan and a serial makespan, each with its `how`. Since 2026-10-20 every duration is what gpu-mock's replay paces (`build/common.py:paced_duration`: one unit per optimizer step, converted by gpu-mock's `convert()`), not the recorded wall time, which includes host time replay does not pace; so no replayed run can beat the bound. The bound takes each job's shortest drawable record, the serial figure its median. These are planning figures only, and a decision-bound job counts at its recorded winner's duration where the decision is resolved and at the candidates' median otherwise. The reference schedules come from `convert()` and the PyJobShop solver. |
| `horizon_s`, `horizon_how`, `horizon_terms` | The run's hard stop and the end of the scoring window when work remains, in recorded seconds, **never shown to the agent**: colleague_round(1.5 × (`stats.serial_makespan_s` + every job's start-up)), rounded up to what a colleague would say (5 minutes under an hour, 15 minutes under six hours, whole hours above), so that a serial agent with no overlap finishes well inside it. `horizon_terms` holds the two terms: the start-up total is the serial scripted policy's start-ups with measured host terms, per machine type (`startup_by_machine_type`, from `survey/host-terms/<task_id>.<machine type>.json`), and the one used is the largest among the types measured, so a single horizon is generous on every type. Until the fleet's type is measured it is marked `provisional` and expected to move; with nothing measured it is 0. **No brief states a deadline or a time limit**; the validator refuses one. This replaced `deadline_s`, which the brief stated, on 2026-10-20; runs under the old briefs are not pooled with runs under the new ones. |
| `model_runs` | Whether a model run may get this task: `allowed`, the `problems` that forbid it, and `how`. It is forbidden when any two jobs can be served one recorded run, or overlapping steps of one (`common.repeated_records`). That covers a seed-substituted job, two prefix jobs cut from one parent, a prefix and its parent's full run, and a resumed segment and a prefix that share the boundary validation. Consecutive segments of one run are allowed, because a real resume shows that boundary again. Same-group jobs drawn without replacement are allowed when their pool is at least as large as their number. The reason is the study's ruling of 2026-10-20 (design/IDEA.md, Detectability): an agent reading its logs sees repeated losses. Scripted runs read no values, so they may run any task. The generator refuses a `pkg-*` task that is not allowed. |
| `measuring_run` | The scripted run that measures the task's host terms on the fleet's machine type: `harness`, `policy`, `speed` 1, `machine_type`, and the host-terms file it feeds (`survey/host-terms/<task_id>.<machine type>.json`, by `build/derive_host_terms.py`). Package tasks carry it. Once that file exists the horizon is no longer provisional, and the runner may declare `task-ready:<task_id>`. |
| `environment_floor` | F, the environment floor, **per machine type** (`by_machine_type`): the overlapping packed policy's (`packed_overlap`, the study's rule: each successor's start-up runs during its predecessor's GPU work) makespan with that type's host terms at its box's caps, minus the lower bound. The machine is the isolation boundary and the box takes the whole machine, so each entry's `measured` records the machine type, run, caps and image digest it came from, and a run's scores are read against its own type's entry; a type with no entry has no floor until its measuring run lands. `fleet_machine_type` and `fleet_measured` say where the fleet stands. F is reported beside the scores and sets nothing. |
| `work_remaining` | How "work remained at the horizon" is decided. For rungs 1 and 2 `rule` is `jobs_not_completed`: a job not launched, still running, or failed and not rerun to completion. |
| `environment` | What the box must hold for the task's scripts to start offline: `python`; `install` (top-level packages, a constraints file under `registry/artifacts/`, the pip line, and `cpu_lines` for a host without CUDA: torch's CPU wheel first, then the rest); `source` (a library staged into the task tree); `hub` (kind, repo, `revision` with `revision_how`, files with byte sizes); `env` (variables the box sets); `verified` (what was run, when, and what was not). TRL tasks only, so far. `build/stage_task.py` stages the tree and writes a manifest from these fields; `build/stage_hub.py` fills a Hugging Face cache from the manifest. |
| `registry/expected/<task_id>.json` | Not a task field but a sibling output: the expected scores of the runner's scripted agents on the task, per policy, from `build/expected_scores.py`. It takes the draw and lookup from gpu-mock's own Registry code and the pacing rule from `build/common.py:paced_duration`, so it is the registry's half of the scorer check (C1 in `STATE.md`); the runner computes the other half independently. Host terms are zero until measured on the box. Whole-box idle inside a job's start-up is split between `job_host_phase` (the environment's) and `startup_not_overlapped` (the agent's) by `startup_split` in `build/expected_scores.py`, whose docstring holds the definition agreed with the study (design/IDEA.md, "job host phase"); `startup_split_by_job_recorded_s` gives each job's start-up, its whole-box idle, its own whole-box agent share (wall seconds) and its partial agent share (`partial_startup_not_overlapped_gpu_s`, GPU-seconds). Partial idle carries causes per GPU lane (design/IDEA.md, "Partial idle also carries causes"): `partial_idle_by_cause_recorded_gpu_s`, from `partial_attribution`, sums to `partial_idle_recorded_gpu_s`, and adds `no_work_left` for a lane idle while no task job is both unlaunched and launchable. `build/test_startup_split.py` holds the known-answer cases for both. |
| `files[].git` | On a `files[]` entry that is a source checkout: `{repo, commit, keep}`. Only directory `keep` is staged, so no `.git`, tests or other examples reach the agent. |
| `readiness` | Replay-ready jobs, and blockers counted by record. |
| `notes`, `sources` | What a reader needs to know about the task, and where it came from. |

## 10. Disclosure

What the agent knows about durations is an experiment dimension. Each record carries `disclosure.duration`: `true_s` (the job's whole recorded duration, with `true_how`) and the recorded runs of the same configuration (`group_median_s`, `group_range_s`, `group_n`). MLS-Bench records add `limit_s`, the job's declared wall-clock limit. Each task job carries four levels the runner may choose between:

- `none`: nothing.
- `limit_s`: the source's declared limit, where one exists.
- `estimate`: the median and range of the job's recorded runs, and the hardware they ran on.
- `exact`: the duration gpu-mock's `convert()` computes at launch for the GPU count chosen.

Nothing in the registry decides which level an experiment uses.

## 11. Readiness

`readiness` is computed by `build/common.py:readiness` and checked by the validator, never written by hand. A record is replay-ready when it has an entry hash and a script on disk, known hardware and GPU count, a duration its pacing can use, every series the job reads, and no blocker. Blockers are named in `materialisation.blockers`. Readiness is the registry's side only: whether the record carries what replay needs. Which serving modes gpu-mock's daemon supports is in gpu-mock's `docs/BUILD-STATUS.md`.

`materialisation.serving.mode` says how a job's values are served:

- **`per_op`**: every value the job reads is answered from a recorded series or a derivation from one.
- **`results_at_end`**: jobs whose recorded result comes from text they decode and score, such as PPO and online DPO, and evaluations that score on the CPU. Their per-token values were not recorded. gpu-mock paces the recorded duration, serves any logged series, and serves `result.*` at the job's final print.

  What that mode serves today is a GPU tensor read at the final print. Where the final metric is a CPU-side Python number computed from strings or log-likelihoods, as in the MLS-Bench KV evaluations and the lm-eval jobs, no tensor hook sees it, and a decoding loop expects `[B, T, V]` logits at every step. gpu-mock reported on 2026-10-19 that serving those needs a `generate()`-level short-circuit and a serving point at the job's own results write, neither built yet; until then such records carry a blocker saying so and are not replay-ready whatever the scaffold does. To let gpu-mock build the serving point against the registry, those records' `serving` also names `final_print` (the stdout line the benchmark's parser reads), `results_file` (path pattern under `OUTPUT_DIR`, format, the recorded keys) and, for decoding jobs, `decode` (workload, `max_new_tokens`; per-example generated lengths were not recorded).

Counts are in `registry/CATALOG.md`, which is regenerated with the registry.

## 12. Host profile for cpu-tasks

`host` carries what cpu-tasks' shared layer needs to make a job consume real CPU, RAM and disk. Every field is a value with provenance, in bytes, seconds, counts or percent. A field the source did not log is `absent`.

Fields every source can fill: the process count and threads per process, the data files read with their sizes and order, pinned host memory per rank, and data-loader settings.

Fields from W&B system metrics (TRL records), added on 2026-10-19 at cpu-tasks' request:

| Field | What it is |
|---|---|
| `rss_bytes_at_fraction` | Eleven readings of the main process's RSS, at 0, 10, ..., 100% of the run's wall time: `values`, and `t_s` saying when each reading was taken. The first reading arrives about 10 to 60 s after start, so entry 0 is that reading. |
| `process_rss_peak_bytes` | The maximum of the same RSS series. |
| `system_mem_used_peak_bytes` | Peak memory in use on the whole host. It counts every process: all ranks, loader workers, and any other job on a shared host. Use it as an upper bound. |
| `host_mem_total_bytes` | The host's memory, as the same fields imply. |
| `host_cpu_util_pct_median`, `host_cpu_util_pct_p90` | The host's CPU use: the mean over logical CPUs of each CPU's busy share, taken per sample. |
| `main_proc_cpu_pct_of_all_cores_median`, `_p90`, `_lifetime_mean` | The main process's CPU use as a percent of all logical CPUs, where one busy core is 100 divided by the CPU count. |
| `wandb_sdk`, `sample_window`, `cpus`, `dataloader_num_workers`, `dataset_num_proc` | What the numbers above rest on. |

What these rest on:

- **Rank 0 only.** W&B samples the process that called `wandb.init`. HF's Trainer and accelerate call it only on the main process, so per-process figures of a multi-GPU job cover rank 0.
- **The SDK changed in 0.18.** Before it, each sample is the mean of 15 readings taken 2 s apart, published every 30 s, so a spike shorter than 30 s is averaged down. From 0.18 a sample is one reading every 10 s.
- **Process CPU from 0.18 is a lifetime average.** wandb-core reports the process's CPU time since start divided by its age. For those runs only `_lifetime_mean` is filled, and the median and p90 are `absent` with the reason.
- **Memory precision.** Before 0.18, host memory is recovered from available memory and a percent rounded to 0.1%, which leaves an error of about 1 GiB on these 2 TB hosts. The `how` gives it per record.

`artefacts` carries disk demand:

- `checkpoint_bytes`: bytes per save.
- `save_every_steps`: the save interval.
- `save_total_limit`: how many checkpoints are kept; a null with `null_means` says every checkpoint is kept.
- `checkpoint_saves`: how many checkpoints the run wrote.
- `final_model_save`: whether the script also saves the model when training ends.
- `log_bytes`: the run's log on disk.

No source logged a checkpoint size, so TRL's are `derived` (requested by cpu-tasks on 2026-10-19), from fields the records already hold:

- `model_parameters`: the standard transformer parameter formula for the architecture, from the HF config W&B logged, adjusted for the head the script trains (a sequence-classification score layer replaces the LM head). `build/trl_models.py` holds the formulas, and each `how` names the Hub's safetensors count it was checked against.
- `trainable_parameters`: every parameter, or the LoRA adapter's `r x (in + out)` per targeted layer for a PEFT run.
- `checkpoint_bytes`: saved weights in their dtype (fp32 unless argv sets `--torch_dtype`; a LoRA adapter is saved alone, in fp32) plus the optimizer state in the same dtype, two moments per trainable parameter for AdamW and one for RMSprop. Tokenizer, scheduler, RNG and trainer-state files, a few megabytes, are not counted.
- `final_model_bytes`: the weights alone, which `trainer.save_model` writes after training.
- `disk_at_end_bytes`: kept checkpoints times `checkpoint_bytes`, plus the final model; the peak, since nothing is deleted except what `save_total_limit` removes.

A record whose config names no model keeps these `absent`. MLS-Bench's fixed-architecture training jobs carry a checkpoint size inferred from the parameter count. A track-3 run that wrote no checkpoint has `null_means` on all three save fields.

## 13. Open items, as of 2026-10-19

These are expected to move:

- MLS-Bench records now carry the bytes their python process runs: the Harbor scaffold's entry file with the baseline's edit ops applied by `build/gen_mlsbench.py` (the same line-based replace/insert/delete as MLS-Bench's harbor_adapter), under `registry/artifacts/mls/scripts/`. The per-method copies the tasks describe are those files; assembling them with the task scripts into a working directory is still job-program work.
- The lm-eval jobs of the two pretraining tasks share script bytes, argv and non-exempt env across variants and differ only by the checkpoint they load. Matching them needs lineage, resolving the loaded file to the job that wrote it; not yet raised with gpu-mock, because the MLS-Bench tasks are deferred, and carried as a blocker until it exists (§5).
- The MLS-Bench KV evaluations and the lm-eval jobs also wait on gpu-mock's serving of a CPU-side final metric (§11). Their records name the results file and metric keys for it.
- gpu-mock's `convert()` is live (2026-10-19), so no task carries a conversion blocker any more; the tasks' `stats` call it.
- Track-3 records infer their peak memory from track 1's logs of the same model and micro-batch, because track 3 logs none.
