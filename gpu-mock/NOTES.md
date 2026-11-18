# next/: the build after 3ec0 (started 2026-10-26)

This tree is the next gpu-mock build, developed beside the frozen one.

- **`experiments/gpu-mock/src` must stay byte-identical to `archive/3ec027ac….tar.gz`.** It is the paper's instrument, and isolation's `image/build.sh` builds from it by default.
- **This build never goes on the paper's machines or the fleet unless the study asks**.
- To build an image from this tree, give isolation's `image/build.sh` the option `--gpu-mock <path to next/>`.
- Every revision is archived like any other.

The scope behind each item is in `../docs/DETECTION-SCOPE-2026-10-26.md`.

## Order

1. **Host CPU while paced.** Moved first so the study can choose to run Grok on a build that has only this fix.
2. Timing Fix A plus processor sharing (Fix B).
3. Power and clock readings.
4. Served subsets.

## 1. Host CPU while paced, with the driver libraries mapped

**What changed:**
- `replay._gpu_wait` replaces every pacing sleep: in `_pace`, `_drain` and `_pace_example`.
- `DaemonClient.sync` waits the same way.
- The wait spins on the genuine monotonic clock and calls `os.sched_yield()` each pass. The process sits at about a core in state R, and the job's other threads keep their rate: measured 1.04–1.13× their rate with nothing waiting, where a bare spin halved them.
- `NVRT_WAIT=sleep` restores the old sleeping wait. The tests use it as the control.
- The daemon and the cursor are unchanged, so every served timing is the same.

**The `libtorch_cpu`-only mapping (analysis' question, 23:48Z): closed in this same build.**
- `cuda_hooks._map_driver_libraries` loads the box's stub `libcuda.so.1` and `libcudart.so.13` (`libcudart.so.12` as a fallback) with `RTLD_GLOBAL` at apply. They come from the bundle on `LD_LIBRARY_PATH`, and nothing calls them.
- `/proc/<pid>/maps` then holds `libcuda.so.1` and a libcudart beside libtorch.
- Checked on the 3ec0 packages image (`04827d41`, 2026-10-26 23:5xZ): DataLoader fork workers, Python signal handlers and cuda ops are unaffected.
- **Not closed:** `libtorch_cpu.so` is still mapped, and `libtorch_cuda.so`, `libc10_cuda.so` and the cuBLAS and cuDNN libraries are absent, as on a CPU build of torch.
  - Closing that means shipping stub libraries under those names. A process that loads a stub `libtorch_cuda.so` beside a CPU libtorch risks symbol clashes, so it is a separate item and not in this build.
  - An agent that reads the maps for `libtorch_cuda` or `libcublas` still sees a CPU build.

**Tests:**
- `tests/test_host_cpu_wait_daemon.py`: 6 checks, controller and box. Its control, the 3ec0 client, fails the CPU check: 0.00 of a core.
- `tests/test_host_cpu_in_image.sh`: 7 checks. A served per-example launch is watched from outside the job: `/proc/<pid>/stat` ticks over 10 s, `wchan` sampled 10 times, and the maps. Its printed score and `runtime_seconds` must still equal the record's. The `NVRT_WAIT=sleep` launch is the control.

**Status, 2026-10-27 00:34Z: built as revision r1 and archived; not gated, not image-built.**
- Hashes: patch `61ef956efb5bb2d19fd79134ac18c5fa91b1aa5c66cf24b8d241a54e2d3838aa`, sidecar `e9fdb717c9fc90f12f2770dfcfe4a85793ff6c5725b886927f217b804fea5441`, install `2fc10846…` (unchanged). The sidecar hash moved only because `daemon_client.py` is under `src/`; the daemon is byte-identical to 3ec0's.
- Archive: `../archive/next-61ef956e….tar.gz`, with an INDEX line.
- Controller daemon suites, 18 files, all pass, the new host-CPU wait test included.
- `test_host_cpu_in_image.sh` passes 7/7 on the E2, on the 3ec0 packages image `04827d41` with this patch mounted (PATCH_SRC=1).
  - Spin run: ≥ 5 s of CPU in 10 s and `wchan` 0 in 10 of 10 samples, against the control's `hrtimer_nanosleep` 10 of 10.
  - `libcuda.so.1` mapped (none in the control).
  - Score 32.5 and `runtime_seconds` 120 s in both runs.
- Not yet run: the full in-image suite and a run_one known answer on a built image. Both come with a gate, once the study asks for this build.

## What a profiler shows of a served job

Probe: `tests/probe_profilers_in_image.sh`, run on the E2 at 2026-10-27 00:15–00:19Z, on the 3ec0 packages image with this patch mounted. One per-example stand-in launch, `--max-examples 10`, was run four ways.

**0. No profiler.** Served: `exact`, the seams installed, the recorded 30.0 printed.

**1. `python -m cProfile -s cumtime script.py`.**
- **The launch is not served.** Under `-m cProfile`, `sys.argv[0]` and `__main__` belong to cProfile. The entry-script hash and path do not match, so the lookup is `match none`: the real CPU path, score 0.0, runtime 0.54 s instead of the paced 5 s. A profiled launch therefore gives a different score from an unprofiled one, which is a tell of its own, the same kind as Opus's tell 2.
- **The table names the patch.** Its top 25 rows by cumulative time include:
  - `sitecustomize.py:255(exec_module)` at 3.7 s, the patch's import hook;
  - `cuda_hooks.py:1(<module>)` and `cuda_hooks.py:90(apply)` at 1.8 s;
  - `cuda_hooks.py:472(_fix_event_base)` and `_dynamo_enforces_stream_base`;
  - `flop_charger.py:110(wrapped)`, 578 calls;
  - FakeGPU's `torch_patch.py:1431(_counted_torch_function)` and `torch_patch.py:547(_tracked_torch_function)`;
  - `_cross_device.py:64(wrapper)`.

  `-s cumtime` prints basenames here.

**What Grok actually did** (its transcript, `box/harness.stdout` of `/srv/runs/P006-G-kvgate-grok-L0-20261026T231310Z` on c3-15, read 2026-10-27 00:4xZ):
- It first sampled its hung job's `/proc/<pid>/stat` and `wchan`, and looked for py-spy and gdb. Neither is installed. `pip install py-spy` failed: the box has no route to PyPI, so py-spy is not reachable in the agent's box. My probe's py-spy came from the E2's own network.
- It then ran evaluations under `faulthandler.dump_traceback_later` through `runpy.run_path`, as `python3 - <<'PY'` stdin scripts. The patch hides its frames from faulthandler's dumps (81f7's fix), and these launches were served exact: `runpy.run_path` sets `__main__.__file__` to the script.
- **Then it profiled the model directly:** a `python3 - <<'PY'` script that loads Qwen with `AutoModelForCausalLM`, runs ten forward steps on the GPU, and wraps them in `cProfile` with `pstats` sorted by cumulative time. Its table's second row:

  ```
  4740/10  0.012  0.000  9.087  0.909 /usr/local/nvidia/python/fingerprint_runtime.py:302(__call__)
  ```

  That row is the full path. `pstats` prints the full path unless `strip_dirs()` is called. The frame is the structural-fingerprint stage's module wrapper, which wraps every `nn.Module` call before a record is matched. A model the agent drives itself is never served, so every forward pass shows it.
- Nine of the run's 85 lookups were these stdin scripts, all `match none`. The rest were the task's evaluators, `exact`.
**2. `torch.profiler` with `ProfilerActivity.CUDA` and `with_stack=True`.**
- torch printed a C++ traceback per op: "Failed to record CUDA event. CUDA used in profiler but not enabled." That message exists only on a CPU build of torch, so it is a tell by itself.
- The run was killed at the container's 8 GB cap before it printed its tables.
- The chrome trace it had written by then names FakeGPU's files by full path: `nvcompat/_upstream.py` 468 times, `nvcompat/torch_patch.py` 262, `nvcompat/link_traffic.py` 160, `_allocator.py`, `link_topology.py`, `profile_catalog.py`, `platforms.py`.

**3. `py-spy dump`.** In the probe, py-spy was installed from the E2's network, which the agent's box does not have (Grok's `pip install py-spy` failed with no route). It shows what an image that ships py-spy, or a box with PyPI, would expose. The dump of the running, served job names the patch outright:

```
_gpu_wait (replay.py:87)
_pace_example (replay.py:1470)
gen_seam (replay.py:1391)
main (custom_selection_eval.py:116)
```

`py-spy dump --native` would add the libraries.

**Summary.** Every profiler names the patch's modules. cProfile names our files, by full path in Grok's own `pstats` table. py-spy, where it is installable, names them and their functions. torch.profiler names FakeGPU's by path and prints a CPU-build error. A rename alone does not hide the patch from a profiler: py-spy shows function names (`_gpu_wait`, `_pace_example`, `gen_seam`) and line numbers in a frame stack that no real CUDA job has, whatever the files are called.

## What the rename plus comment stripping would cost on top of this build

**Size of the visible surface.**
- Ten patch modules: 5,800 lines, with 640 comment lines and 950 docstring lines.
- FakeGPU (nvcompat) beside them under `pylib/`.
- About 130 occurrences of tell-bearing names in the patch's own code: `replay` 77, `FakeGPU` 58, `nvcompat` 23, about 25 distinct `NVRT_*` variables, and the module names.

**Work, roughly 2–3 days, with the risks noted:**
1. **Rename the ten modules** to neutral names in the image, as in the shelved tree: `op_dispatch`, `device_clock`, `cuda_runtime`, and so on.
   - Re-derive it on this tree, since the shelved one is stale (it predates 32c7 through 3ec0).
   - **The resume test regression it hit on 2026-10-23 (10 of 15) was never explained.** It has to be settled on a built image, and this is the unbounded part: 0.5 to 2 days if it reproduces.
2. **Strip comments and docstrings at install, not in the source.** Python's `ast` can rewrite a module without docstrings, and the tokenizer can drop comments. Install the stripped copy and keep the commented source in the repository and archive. Line numbers in tracebacks then change, which matters only for our own debugging.
   - About half a day, plus a test that every module imports and the in-image suite still passes on the stripped copy.
3. **Rename the vendored FakeGPU** (`nvcompat`, the `fakegpu_*` internal names, and the `FAKEGPU_*` variables) and its directory.
   - It is MIT-licensed and vendored, so a rename is allowed with the licence kept.
   - About half a day, plus the identity-surface tests.
4. **Rename the `NVRT_*` variables**, including `NVRT_DISABLE`, `NVRT_DEBUG` and the new `NVRT_WAIT` and `NVRT_NO_DRIVER_MAP`.
   - The variables are read in about 25 places.
   - Isolation's images and `run_one` set some of them, so their side has to change in the same build.
5. **Isolation's side:** the install paths `/usr/local/nvidia/python` and `pylib`, the "Mock NVML Configuration" header in `nvml.yaml`, and the image rebuild.
6. **The runner's tell detector** (`runner/score/tells.py`, `MOCK_NAME_RE`) matches the old names. It must gain the new ones in the same build, or `saw_tell` undercounts.

**What it buys against a profiler: less than the cost suggests.**
- After 1–4, cProfile and py-spy show neutral file names. The frames are still there, still the patch's own functions, and the frame structure is still unusual: a pacing wait inside a function the record names as a seam.
- Hiding the functions too means renaming them (`_gpu_wait`, `_pace_example`, `gen_seam`, `_tracked_torch_function`, …) or moving the waits into C, where py-spy without `--native` does not see them.
- torch.profiler's CPU-build error needs the CUDA activity faked or the error silenced. That is a separate item, about a day, and fragile across torch versions.

**Separate from the rename, and cheaper: serve a profiled launch.** About half a day.
- The cProfile miss happens because `-m cProfile` makes `__main__` cProfile's.
- The entry-path resolution could take the script cProfile runs (it is `sys.argv[0]` after cProfile shifts it, and its `__main__` module carries `__file__`), so the launch is served like an unprofiled one.
- `python -m trace`, `python -m pdb` and `python -m py_spy record -- python script.py` cases need the same check.
- It changes no served timing or result for an unprofiled launch.

## A cheaper item for the profiler door: the frame Grok saw

Grok's table row is `fingerprint_runtime.py:302(__call__)`.
- That is the structural stage's wrapper around `nn.Module.__call__`, installed in every job (`_install_module_hook`). It stays in the call stack of every forward the agent drives, served or not.
- A profile of the agent's own model always shows it, by full path, from the second row down.

**Possible fix, about half a day, not built.** Drop the wrapper from the stack once its work is done.
- The structural fingerprint and both lookups happen at the first root forward and the first step.
- After them, a job that is not replaying could restore torch's own `nn.Module.__call__`, so later forwards carry no patch frame.
- A replaying job keeps it, because `forward_stub` needs it, but a replaying job's forwards are stubs and cost nothing for a profiler to see.

This closes Grok's door for the second and later forwards only. A profile that includes the first forward still shows the frame, as do the FLOP charger (`flop_charger.py:110(wrapped)`) and FakeGPU's dispatch (`torch_patch.py:547(_tracked_torch_function)`) on every op. Those two are the real cost of CPU-backed tensors, and only the rename and stripping hide their names.

**So against a profiler:**
- With the rename, stripping and this un-wrapping (about 3 days with the resume-test risk), a cProfile table shows neutral file names and fewer patch rows.
- It still shows a dispatch wrapper on every op, which a real CUDA build does not have. A careful agent reads that as "something wraps torch", whatever its name.

## The daemon as an answer oracle

**What Grok did.** Its transcript, `box/harness.stdout` of `/srv/runs/P006-G-kvgate-grok-L0-20261026T231310Z` on c3-15, read 2026-10-27 01:2xZ. Its own script:
1. `sys.path.insert(0, "/usr/local/nvidia/python")`, then `from daemon_client import DaemonClient`.
2. For each method and workload: `hello` with the evaluator's argv and cwd, then `lookup_v2` with the script's sha256 as `entry_hash` and `stage="exact"`.
3. It printed the reply's `serving.final_score`, `wall_total_s`, `n_examples_full` and `score_kind`.
4. It read `materialise("example_score", i)` for chosen `i`, then said `bye`.

The occupancy log has 47 zero-work exact lookups ending `bye`, with 3,918 `example_score` reads among them (up to 1,319 for one job, a whole gsm8k vector). So it took the answer two ways:
- **The lookup reply carries the recorded score outright.** `_serving_info` returns `final_score` (and `outcome`) in every replay lookup's `serving` block. The patch never reads either: its reads from `serving` are `per_example`, `eval_s_per_validation`, `series_roles`, `process_model`, `lr_trace`, `failures`, `first_step_s`, `steady_step_s`, `startup_in_train_s`, `per_step_gpu_s`, `wall_total_s`, `memory` and `converted`. The field exists for the daemon's own tests (`test_per_example_daemon.py` checks it).
- **`materialise` serves any index at any time.** `_materialise_example_score` returns `vec[i]` for whatever `i` is asked, unpaced. A client can read the whole vector in a second, and its mean is the recorded score by construction.

### 1. Can the daemon refuse a lookup that doesn't come from a launched job?

The box has one uid. The agent runs as researcher (1000), the jobs it launches run as the same user, and so does the patch inside them. Anything a job process can do, the agent's own `python3 -` can do. Three options:

**(a) The socket unreadable to the agent's user.**
- Not possible as it stands, because the patch runs inside the agent's own processes. A setuid helper would be needed, and the box runs with `no-new-privileges`.
- Drop this option.

**(b) Accept only a pid registered at job start.**
- The daemon can read the connecting pid from the socket itself (`SO_PEERCRED`, Linux, no client cooperation). It would also need to know which pids are "launched jobs". From outside the box it can't tell a job from the agent's own python: both are processes the agent started.
- A weaker rule is cheap. Refuse a lookup whose peer pid's `/proc/<pid>/cmdline` isn't an entry script whose sha256 is the `entry_hash` the request claims, read by the daemon itself, not trusted from the request.
  - The sidecar has its own pid namespace: `isolation/run_one.py` passes no `--pid` option to either container, so Docker's default applies. This was checked in the source on 2026-10-27; no box was running on c3-15 to inspect. So it would need `--pid container:<box>`, or the box's `/proc` mounted read-only. That is isolation's change.
  - This closes Grok's exact route: a `python3 -` stdin process asking for `custom_selection_eval.py`'s record. It doesn't stop an agent that runs the real evaluator under its own wrapper (`runpy`, a `-c` that execs the script), because then the cmdline is the script.
  - Cost: about half a day of mine, plus isolation's pid-namespace change and a gate.
- Side effect on 3ec0-era behaviour: launches that run the evaluator as `python3 - <<PY runpy.run_path(...)`, which Grok's faulthandler wrapper did and which was served exact, would miss.

**(c) A per-job token handed over at exec.**
- Something has to hand it over, and in this box the only thing that sees an exec is the patch, in the same process the agent controls. A token in an environment variable or file is readable by the agent.
- Without a privileged launcher this is (b) with extra steps. Drop it.

**So the identity check is weak by construction.** In a single-uid box the agent and its jobs are one principal. The protection that works is (2).

### 2. Does the lookup return whole score vectors? Can it return only what a running job needs?

No: the lookup doesn't return the vector. It returns `final_score` in `serving`, and `materialise` returns any element on request. Both can be closed without moving anything a real job reads:

- **Drop `final_score` and `outcome` from the lookup reply.** The patch doesn't read them. About 10 minutes, plus moving the daemon test's check onto the occupancy log.
- **Serve example `i` only to a job that has paced example `i`.**
  - The patch's sequence for every example is fixed: the generate seam paces example `i` (an `enqueue` with `unit=example`, `step=i+1`), then the score seam reads `example_score` at `i`.
  - The daemon already knows each job's last paced example (`job.step`, from the enqueue). So it can refuse a read at `i` unless the job's enqueues have reached `i+1` and that unit's GPU time has drained (its cursor is past).
  - A client that wants the whole vector must then pace the whole workload at the record's wall: a gsm8k full-attention read takes its 7,440 s, as the real run does. Reading the vector becomes running the evaluation, at the speed of running the evaluation, through a job the occupancy log counts as GPU work.
  - A refused read returns no value and logs a `failure` event of mode `read_ahead`, so the scorer sees an agent probing.
  - Cost: about 2–3 hours plus a test (a read ahead of the paced position is refused; the patch's own sequence is unchanged; the in-image per-example tests still pass). No served value or timing changes for a job that runs the script.
- **The same rule for series reads** (training jobs' `train_loss`/`val_loss` at a step): serve step `s` only after step `s` is paced.
  - The Trainer seam and the materialisation hooks read at the current step after pacing it, so it's the same shape. Two cases need care: the final-metrics `result.*` reads at the end, and `val_loss` read at a step whose eval unit was paced separately.
  - Cost: about half a day, plus the replay battery to check no training job's read is refused.
- **The serving block's other fields** (`wall_total_s`, the step times, `memory`) are timings the patch needs before it paces. They tell an agent how long the real job took and its memory, not its result. A detecting agent can still read them; they don't give a free answer.

**Together:** with `final_score` gone and reads gated on pacing, the cheapest path to an answer through the daemon is to run each evaluation at its real speed. At that point the agent is doing the task, and the occupancy log scores it as such. What a detecting agent still gains is knowing it's in a simulation, not the answer.

### 3. Where it goes in the order

After r1 (host CPU) and before Fix B, A and the rename.
- (2) is a few hours, touches only the daemon, and moves no timing or result for a job that runs.
- It stops detection from turning into a free answer, which matters more than any one tell: every tell we leave open becomes a score leak through this route.
- (1b) is optional, with isolation, after (2).

Revised order:
1. Host CPU: r1, done.
2. The oracle: drop `final_score` and `outcome` from the lookup, gate `example_score` reads on pacing, then gate series reads.
3. Fix B plus Fix A.
4. Power and clocks.
5. Serve a profiled launch, and drop the structural wrapper after its work.
6. Served subsets.
7. The rename and stripping.

## An exec'd evaluator crashes on the replay stub (G-kvreph-haiku-L0-m3, 3ec0, read 2026-10-27 04:27Z)

**What happened.**
- The agent ran `exec(compile(src, path, "exec"), g)` with `g = {"__name__": "__main__", "__file__": path}`, as a `python3 -I - <<EOF` stdin script.
- The lookup matched exact, because `__main__.__file__` resolution sees `path`, and replay engaged.
- The per-example seams are installed by name in `sys.modules["__main__"]`, which is the stdin module, not `g`. So they were "not found" and nothing was replaced.
- The real `generate_with_selection` then ran its prefill through `forward_stub`, which returns a zero tensor. At line 627 it hit `AttributeError: 'Tensor' object has no attribute 'past_key_values'`, at ranks 11 and 12.

**Fix for this tree, not built.** Two options:
- Find the globals dict that holds the entry script's functions: walk the stack for the frame whose `f_globals["__file__"]` is the resolved entry path, as `_arg_defaults` walks frames for the parser. Install the seams there.
- If no seams are found, disengage replay for that job (the `disengage` path), so the job runs on the modelled path instead of crashing on the stub.

The second is the safe minimum. About 1–2 hours, plus a test that launches the stand-in evaluator exec'd into a dict.

**The 0-against-100 on examples 0–1 in the same run is not this gap.** That is synthesised positions (item 6).

## Release r2: host CPU, the oracle closed, Fix A (2026-10-27)

**Superseded before it shipped, 2026-10-27 15:28Z: r2 was handed to isolation, then withdrawn. Ship r2b.** r2's score gate trusted three values the client sends: an example unit's step label, its cost, and its host back-date. A client could enqueue one zero-cost unit labelled step 1000 and read example 999 at once, so the sentence below that a client "must pace the whole workload" was false for r2. r2b is r2 with the daemon fixed (the patch is unchanged); see "Release r2b" below. isolation stopped the r2 build at 15:31Z, and no r2 image was tagged or shipped.

**What it is.** r1 (host CPU) plus three changes:
1. `final_score` and `outcome` dropped from the lookup reply's `serving` block.
2. `example_score` reads gated on the job's paced position.
3. Fix A: the job's clock never runs behind real time.

No other change. The daemon's cursor, pacing, draws, lookup stages and every served value are 3ec0's.

**What changes for a job that runs its script as recorded: nothing a scorer or bar figure reads.**
- **Lookup.** The patch never read `final_score` or `outcome`; its reads from `serving` are listed in the oracle section above. The occupancy log's lookup event is unchanged, since it never carried them.
- **Score reads.** The patch's seams pace example `i`, wait for its unit to drain, and only then read its score, so every read a running script makes is served exactly as in 3ec0. A read is refused only if it comes before its example's unit was enqueued, or more than 0.05 s before that unit's GPU time ends. The patch never does either. A refused read logs a `read_ahead` event (its own kind, not `failure`, so the runner's failure rules don't see it), capped at three per job.
- **A record with no pacing figure** (no `wall_total_s` or `n_examples_full`; none of the 40 KV records) still enqueues each example, at zero cost, so its scores are served. In 3ec0 such a record paced nothing and its scores were read unpaced.
- **The job's clock (Fix A).** From the first paced example on, the job's `time.time`, `perf_counter` and `monotonic` never read more than 0.5 s plus 1% behind real elapsed time.
  - Uncontended at speed 1, the record's time is always ahead of that floor, so the printed `runtime_seconds` and `mean_latency_s` are the record's, as before. Known answer A: 40.0 s printed.
  - When the job is held up (a shared GPU, an agent's SIGSTOP), the printed runtime and latencies follow its real wall instead of staying at the record's.
  - Pacing, the occupancy log and every daemon-side time are untouched, so idle, makespan, over_best, over_feasible and over_ref are unchanged.
  - The only fields that move are the job's own printed timings under contention. In collected runs those are `runtime_seconds` and `mean_latency_s` in a job's metrics file. No scorer reads them, and the H7 rule reads scores only.
- **Host CPU (r1).** A paced job shows about a core in state R instead of `hrtimer_nanosleep`, and `libcuda.so.1` is in its maps. GPU-side timing is unchanged.

**What changes for a client that is not a running job** (the oracle route):
- an exact lookup still engages replay, but its reply carries no score;
- every `example_score` read without pacing returns `None`;
- reading a vector now takes the record's wall, through units the occupancy log counts as that client's GPU work.

**Fix keys.** Every 3ec0 `fix:gpumock-*` and `gpumock:*` key still holds: the daemon's lookup, draw, failure and serving paths are 3ec0's, apart from the two oracle changes.
- One edge case for the runner: `fix:gpumock-host-backdate` concerns the per-example host gap. It is unchanged; the back-dating code in `_enqueue` is the same.
- New keys, bound to r2's images:
  - `fix:gpumock-host-cpu-spin` (r1): a paced job busy on a core; `libcuda.so.1` mapped.
  - `fix:gpumock-no-score-oracle`: no `final_score` or `outcome` in the lookup reply; score reads gated on pacing.
  - `fix:gpumock-clock-follows-wall` (Fix A).

## Release plan (2026-10-27 14:5xZ; dates are estimates, pinned to this session's work only)

1. **r2** (1 + 2 + Fix A): gate and archive today, hand to isolation today. The Grok KV lines run on it.
2. **r3**: series-read gating, the exec'd-evaluator gap (disengage replay when the seams aren't found), and power and clock readings. About a day. Gate around 2026-10-28.
3. **r4**: Fix B (processor sharing per GPU). About half a day plus a known answer and the full replay battery, since it changes per-job times under sharing. Gate around 2026-10-29. Its release note will carry the per-run effect, measured with `tests/fairshare_counterfactual.py`.
4. **r5**: serve a profiled launch (`python -m cProfile` and friends), and drop the structural wrapper after its work. About a day. Around 2026-10-30.
5. **r6**: served subsets with real per-example outcomes. Needs gpu-tasks' record format and a fleet afternoon between runs, coordinated with gpu-tasks and gcp. No date until gpu-tasks has the format.
6. **The rename and stripping**: last, optional.

## Release r2b: r2 with the score gate made unforgeable (2026-10-27)

**Hashes.** Patch `92660649b94db599…`, unchanged from r2. Sidecar (daemon+bridge+convert) `89c4875a4b948ddc`. Install `2fc10846` unchanged. Archive: `../archive/next-92660649…-sidecar-89c4875a4b948ddc.tar.gz`. Frozen tree: `/srv/work/gpu-mock/next-r2b` on the E2.

**The daemon change (`_enqueue`, `_example_cost_floor`).** An example unit opens its example's score only when all three hold:
1. **It is the job's next example in order.** Its step is the count of examples already opened, plus one.
2. **It is on the job's one GPU.**
3. **It costs at least the record's per-example share:** (wall − the start-up the launch claimed) / `n_examples_full`, less 0.1% for rounding. The claimed start-up is capped at half the wall.

The host back-date is also clamped. It can't exceed the real time since the job's previous example unit ended, and the first unit gets none.
- The patch prices every unit at exactly (wall − its measured start-up) / `n_full`, in order, on one GPU. Its measured start-up across all collected runs is at most 5.0 s, against walls of 323–29,153 s. So a job that runs its script opens every example as before.
- A forged label, a zero price or a huge back-date opens nothing.
- Units that open nothing are still queued and paced as before, so a client that floods units only occupies its GPU.

**Behaviour change for a job that runs its script:** none beyond r2's. One collected-run effect: `host_gap_s` on an enqueue event is the clamped value. It differs from the claimed value only when a client claims more than the real gap, which the patch never does, since it measures from that same drain.

**Tests:**
- `test_per_example_daemon` 29/29: the forged label, the zero price and the 1e6 s back-date are refused; the honest next unit opens.
- `test_host_gap_daemon` 4/4, with the claimed 0.5 s gap now clamped to the real 0.31 s.
- All 18 controller daemon suites pass.
- On the E2, patch mounted on the 3ec0 packages box: `test_per_example_in_image` 17/17 and `test_host_cpu_in_image` 7/7.
- `test_oracle_and_clock_in_image`: the Fix A check is recalibrated. It now compares B's wall less its printed runtime against A's, since the process wall includes start-up the script's timer doesn't. 8/8 after the change.

**Gated 2026-10-27 17:22Z** (BUILD-STATUS has the steps). Images:
- packages `fa92b756`, sidecar `587227e7`;
- KV `bac931d9`, key 2f6b8b33328a;
- dataprep `02e59fd2` / `48f3ef66`;
- filter `d16b9c2b` / `4f5cb3ea`.

The gate line went to the runner at 17:24Z, carrying all 30 existing keys plus `fix:gpumock-host-cpu-spin`, `fix:gpumock-no-score-oracle` and `fix:gpumock-clock-follows-wall`.

`tests/gate_new_pair.sh` and `tests/probe_all_scripts.sh` take an `EXP` override now. A frozen release tree is not under experiments/, and without the override the pair gate's first pass found no registry.

## Release r3: the box's text stripped, the exec'd-evaluator gap closed (2026-10-28)

**Why r3 leads with the text.** G-kvgate-astracc-L0-r2 found the mock by reading a file, without hunting for it. An Explore subagent grepped site-packages for `cuda|version`, read `sitecustomize.py`, and its first line said "Loads the gpu-mock torch layer". Every patch module's docstrings were as explicit.

**What changes in the box:**
- **Text strip.** `install_into_image.sh` installs `sitecustomize.py`, the torch layer (`/usr/local/nvidia/python`) and the vendored FakeGPU package (`pylib/nvcompat`) through `container/strip_comments.py`:
  - every comment and docstring is removed; the rest of the text is unchanged;
  - each copy is checked to be the same program (the same syntax tree once docstrings are dropped) before it is written, and the installer stops if one isn't;
  - `stage_bundle.sh` ships the stripper in the bundle, and `NVRT_KEEP_COMMENTS=1` installs the source as is;
  - the repository and the release archive keep the commented source.
- **Renames in our own code:**
  - sitecustomize's import hook `_NvrtTorchHook` → `_TorchImportHook`, and its failure message → "cuda runtime init failed";
  - in `replay.py`: `_disable_fakegpu_memory_tracking` → `_disable_device_memory_tracking`, `_hook_fakecuda_backward` → `_hook_device_backward`, and their notes;
  - in `cuda_hooks.py`: the locals `_have_fake_dist`, `fake_name` and `fake_cls`.
  - FakeGPU's own API names (`FakeCudaTensor`, `_FakeStream`, `FAKEGPU_*`, `import_module("fakegpu")`), 21 places in our code, stay until FakeGPU itself is renamed.
- **The exec'd evaluator** (G-kvreph-haiku-L0-m3):
  - `_seam_namespace` finds the per-example seams in the globals dict of a frame that execs the script, matched by `__file__` against the entry path;
  - if no seams are found anywhere, the job withdraws replay (`per_example_seams_not_found`) and runs on the modelled path instead of crashing on the stub.

**What changes for a job that runs its script as recorded: nothing.** The text strip changes no code, and the equivalence check enforces it. The renames are internal names. The exec'd-evaluator change applies only to a launch whose seams 3ec0 and r2b could not find, and those launches crashed.

**What still names the mock in the box:**
- FakeGPU's code identifiers (about 1,100, e.g. `FakeCudaTensor`), now without prose;
- the module and directory names (`nvcompat`, `cuda_hooks`, `replay`, `virtual_clock`, `/usr/local/nvidia/python`);
- the `NVRT_*` environment names.

That is the rename item, still last.

**Tests:**
- `tests/test_strip_install_in_image.sh` 4/4 on r2b's packages box, re-running this tree's installer inside it:
  - 139 files with 0 comments and 0 docstrings;
  - our files name nothing of the mock;
  - a cuda matmul runs, and a served stand-in prints its record's score and runtime;
  - the r2b box itself, as the control, still starts "# Loads the gpu-mock torch layer".
- `tests/test_exec_evaluator_in_image.sh` 4/4; its 3ec0 control fails 3.
- 18 controller daemon suites pass, also on a fully stripped copy of the tree.
- In-image with the patch mounted on r2b's box: per_example 17/17, exec 4/4; the rest of the suite is running.

**Not in r3:** series-read gating and power and clocks move to r4, with Fix B.

## Release r4: an evaluator the agent's own runner imports is served by example content (2026-10-28)

**Why.** Three frontier runs loaded an MLS-Bench evaluator as a module instead of running it as a script:
- astracc's shard runner, G-kvgate-astracc-L0;
- Fable's debug harness, G-kvgate-fable-L0-r2, with argv[0] "x";
- Opus's score printers, G-kvreph-opus-L0 and -r2, with a uuid module name.

Before r4 each of these matched no record. The job ran real CPU inference, about 10× slower than the record, with 0 MiB on nvidia-smi. Fable spent 04:42–05:00Z on 10-28 chasing exactly that, "script-based runs are consistently fast, while python -c and direct debug-main invocations are consistently slow".

**What changes:**
- **The patch** (`replay.py`):
  - At lookup it sends the project modules whose code is running at the first CUDA touch: name, real path, sha256. These are the modules with frames on the stack, outside the stdlib, site-packages and this layer, the entry script excepted.
  - A module that was only imported for a helper (its loader, its scorer) is not running at that point, so it isn't offered. The job stays modelled, as before.
  - When the daemon names a module as the entry, the generate and score seams are redirected in place: the original function objects keep their identity, name, signature and source, and their code becomes a call to the served seam.
  - So a runner that captured `module.score_prediction` before CUDA started, and wraps it, calls the served seam. If a seam can't be redirected, replay withdraws (`per_example_seams_not_redirected`) and the job runs modelled.
  - Each example unit and score read carries the example's content key, `sha256(example_id + "\0" + prompt)`.
  - The start-up is measured from the evaluator's own first `perf_counter()` in its module. `virtual_clock` keeps the first reading per module until the first lookup that has no use for them.
- **The daemon** (`gpu_daemon.py`):
  - It loads gpu-tasks' `example-order-*.json` tables (schema `p006-example-order/0.1`) from `records/`. A table with duplicate keys, the wrong length or the wrong key recipe is refused and logged.
  - **New lookup stage `exact_module`,** with `exact_effective_module` as its effective-argument form. It runs after the exact and effective stages, for a module whose script has a per_example record with an order table. The module's basename stands in for argv[0]; the rest of argv must match as usual.
  - **Units are keyed.** A per_example unit opens the record position its key names in the table, or its local index when there is no key. A key the table lacks falls back to the local index, logged `example_order_miss` (at most 3 a job) and counted on `job_bye`.
  - **The r2b gate is unchanged.** Units must still come in order, on one GPU, at the per-example share, and a score is served only for a position the job paced and whose GPU time has passed.
  - **Labels:**
    - `materialise` events carry `position` when it differs from the local index;
    - a score served to a module launch or out of prefix order is labelled `synthesised_subset`;
    - `job_bye` carries `positions` {n, first, last, contiguous, prefix} and `entry_module`, and `lookup` carries `entry_module`, `entry_module_hash` and `example_order`;
    - a repeated module launch of one record is `draw: module_repeat`, not `deterministic_rerun`, since it may be another shard.

**What changes for a job that runs its script as recorded: nothing.**
- The module list is sent, but the exact stage matches first.
- The patch sends keys, and for the whole workload in order every key's position is its local index.
- A script launch's scores, positions, provenance (`result`) and `job_bye` are what r3 gives. `tests/test_shard_serving_daemon.py` case 6 and the in-image case S check this.

**What changes for a module launch:**
- It is served the record's own examples, at the record's pace, on its own GPU.
- Four shards of a 1319-example workload on four GPUs take about a quarter of the wall each, at the same time, as on hardware.
- Each shard prints its own slice's mean.
- The pooled shard means equal the recorded score when the shards cover the workload, because each position's served value is the same whichever launch decodes it.

**Fix keys for the runner:**
- new: `fix:gpumock-module-entry` (exact_module, keyed shard serving);
- carried: every r3 key;
- dropped: none.

**For r5:** the daemon logs every table it loaded, with its sha256, on `daemon_start`. That carries a run's table provenance; today run.json's `records_sha256` covers only the task's listed records. Until r5, isolation is asked to record the tables' hashes in run.json.

**Prerequisite:** the six tables in `gpu-tasks/registry/records/`. gpu-tasks staged them at 07:33Z on 10-28; they reach the machines with isolation's registry sync. Old daemons tolerate them:
- I loaded the full registry plus the six tables with the 3ec0, r2b and r3 daemons. All 40 KV records still match exact and draw themselves.
- What those daemons do with the tables: they index each table as a record with no entry hash, which nothing can match or draw.
- So gpu-tasks can stage the tables before r4 reaches every machine.

**Tests:**
- `tests/test_shard_serving_daemon.py` 19/19 covers:
  - table loading and refusal;
  - the module stage, including that it needs the argv to match and the order table to exist;
  - shard [10, 20) served at positions 10..19;
  - four shards pooling to the recorded mean;
  - the order miss;
  - a script launch unchanged;
  - the gate under keys: a paced job asking another example's key is refused, and an out-of-order unit opens nothing.
- `tests/test_shard_serving_in_image.sh` 9/9 on r3's KV box with the patch mounted:
  - four concurrent shard runners in astracc's shape match exact_module, each prints about 10 s for its 10 examples, the four span 15 s against 40 s in turn, and the means pool to 32.5;
  - the Fable and Opus harness shapes match exact_module;
  - a helper-only import stays modelled;
  - with no table the runner misses, as before r4.
- `tests/test_shard_real_in_image.sh` 12/12 on the real evaluators, with the staged tree b85d30e78b4d, the HF cache 02c58c389f0f and gpu-tasks' six tables:
  - kivi gsm8k shards [100, 104) and [1315, 1319) at the same time;
  - full_attention gsm8k through Fable's harness at --max-examples 3;
  - kivi needlebench_niah shard [3, 6), where the prompts repeat;
  - every served score equals the record vector at its position, with no order miss. So the tables and the real loaders agree on the keys.
- 17 controller daemon suites pass.

## Release r6: measured per-example scores for each KV record's first examples (2026-10-29)

**Why.** Each KV record carries only its overall score, so a short evaluation's per-example scores were a seeded shuffle averaging to it, not what those examples score. An agent smoke-testing three examples could get three misses where the real model gets two hits. r6 serves the real scores for each record's first K examples. Operator ruling: r6 is top priority through to the fleet.

**The data, from gpu-tasks' records.** A record that carries `materialisation.serving.per_example_scores` gets its first k examples' measured scores:
- `k` is 50;
- `keys`, `example_ids`, `scores` and `how` are in the order-table order;
- gpu-tasks' importer refuses a block whose keys are not the order table's first k, whose scores are out of scale, or whose binary tail cannot hold the recorded total.

**How the scores are measured** (`tests/decode_first_k.py`; slices merged by `tests/r6_merge.py`):
- the record's own evaluator, byte-identical (sha256 = entry_hash), imported as a module;
- its own `main()` under the record's argv, on `load_workload(name, 0)[start:end]`;
- `score_prediction` wrapped to record each example's score and prediction;
- the model loaded in bf16, the recorded precision, by wrapping `from_pretrained` at run time;
- run inside the r4 KV box with the replay layer off (`NVRT_DISABLE=1`), on the c3 fleet's CPUs (Sapphire Rapids, AMX bf16).

**Fidelity limit:** CPU bf16 is not bit-identical to the H100 kernels, so a near-tied greedy token can flip and change an example's score. Each block's `how` says so.

**The daemon** (`_measured_scores`, `_score_vector`, `_synth`):
- positions 0..k−1 of the vector are the measured scores;
- positions k..n−1 are synthesised to the mean that keeps the whole vector's mean at the recorded `final_score`;
- if the tail cannot balance it, the tail is clamped to the scale and noted. gpu-tasks' importer refuses such blocks, so this never happens on an installed record.

**Labels:**
- a served position inside the head is `provenance: measured`, with no subset_model and `measured_k` on the event;
- positions past the head are labelled as before;
- a new `registry_loaded` event at daemon start lists every example-order table by sha256 and the count and k of measured records. That closes the provenance gap gpu-tasks raised on 2026-10-28.

**What changes for a job that runs its script as recorded:**
- **A record without a measured block: nothing.** Its vector is byte-identical to r4's (40/40 checked).
- **A record with one:** a short evaluation of the first k prints the mean of the measured scores, and a full run still prints the recorded `final_score`.

**Shipping the code and the data separately.** The serving code ships once, as r6. The measured blocks reach the boxes through isolation's registry sync, so each record serves its measured scores from the first sync after its decode lands, with no rebuild.
- **Wave 1:** gsm8k and NeedleBench, 12 records. gsm8k alone is 46% of the 422 collected short-evaluation launches.
- **Wave 2:** LongBench, 28 records, sized by what agents launch there. The whole first 50 covers 100% of LongBench launches; 32 covers 84%; 20 covers 72%.

**What the first-50 means do and don't show**.

On five gsm8k records, each record's measured first-50 mean is within 0.6 of the first-50 binomial SE of its recorded `final_score` over 1,319 examples:

| Record | Measured first 50 | Recorded `final_score` | z |
|---|---|---|---|
| squat_subspace_4bit | 32.0 | 31.77 | +0.04 |
| full_attention | 32.0 | 31.84 | +0.02 |
| kvtuner4_pertoken | 44.0 | 43.52 | +0.07 |
| streamingllm | 2.0 | 1.74 | +0.14 |
| kivi_overlap_4bit | 28.0 | 31.84 | −0.58 |

That shows nothing wrong with the decodes. It is **not** evidence that CPU bf16 reproduces the H100 numerics. At n = 50 the SE is about 6.6 points, so a disagreement of a few points per record would go unseen, and the first 50 are a subset, not the same examples as the full run's mean.

A real fidelity measurement needs either:
- a record's whole workload decoded on CPU and compared with its `final_score` at full n (about 35 CPU-hours for one gsm8k record, gpu-tasks' estimate); or
- the same examples decoded on an H100.

Neither is run. Do not cite the table above as a fidelity result.

gpu-tasks re-ran each evaluator's own `score_prediction` on all 250 predictions independently, and every score matches. So the scores are what each evaluator says about those outputs; the outputs are the CPU decode's.

**Measured-head count per record, as the data lands.** A run's labels can be checked against this table: a served position p is `measured` exactly when p < the record's head at the run's sync.
- **gsm8k:** all 8 records, head 50 (wave 1; all imported by 12:01 MDT 2026-10-29).
- **needlebench_niah:** all 4 records, head 50 (wave 1; all in by 12:38 MDT). Every first-50 mean reads 8–11 points above its record, because the first 50 examples are 17 whole essays of 147.
- **LongBench:** all 28 records, head 50, by 16:20 MDT 2026-10-29, 0 failures (pass 1 to head 20, then pass 2 to head 50). Each record's first 20 are byte-identical across the two passes.
- **So every one of the 40 KV records has head 50** (16:20 MDT 2026-10-29).
  - **LongBench-E's first k are its shortest band**. Each -E file is stored sorted into three length bands of 100: under 4k words, 4–8k, 8k+. The evaluators read in file order, so every first-k block at k = 20 and k = 50 lies entirely in the short band. hotpotqa's first 20 average 2,875 words against 6,658 for all 300.
  - Long-context accuracy falls with length, so first-k means above the full run are expected on hotpotqa and passage_retrieval wherever the decode runs: squat hotpotqa +2.56 SE (59.79 against 36.53), the others +1.1 to +1.9. Repobench reads flat (within ±0.5 SE); code completion is less length-sensitive.
  - **The test that separates a CPU-bf16 effect from the length effect:** each quantiser's first k against full_attention's first k on the same examples, paired, against the gap between their recorded full runs. The length effect cancels in that difference.
    - All four LongBench workloads at head 50, 16:20 MDT: 24 rows (every method against full_attention on the same 50 examples).
      - Against full_attention, 2 rows are past 2 paired SE, squat and streamingllm on passage_retrieval (+2.26, +2.35).
      - **Those rows are not independent: every row on a workload shares the one full_attention block**. On passage_retrieval all 7 differences are positive, mean +3.72: a shift common to the shared baseline, not to the two methods.
      - **The test that removes the shared baseline:** gpu-tasks' `survey/r6-paired-2026-10-29/centred.py` centres each method on the mean of the others, per example, which cancels full_attention exactly. That gives 40 rows over every KV workload, NeedleBench included, and none reaches 2 SE (largest |z| = 1.60, full_attention passage_retrieval; output `centred-2026-10-29T2231Z.json`).
      - squat hotpotqa, +2.19 at head 20, is +1.01 against full_attention and +1.37 centred at head 50: noise.
      - No sign of a method-specific CPU-bf16 effect. The SEs are 1–5 points, so effects of that size on these first 50 are ruled out; smaller ones are not. Only the full-n decode below measures the CPU-vs-H100 difference over a whole workload rather than over its first 50.
      - gpu-tasks' `survey/r6-paired-2026-10-29/paired.py` reproduces the table from registry/records.
    - **The full-n decode (measured, 2026-10-29 22:33 MDT):** kivi_overlap_4bit on gsm8k, all 1,319 examples, CPU bf16, on worker-13. Analysis approved this run.
      - **Result:** CPU 424 hits (32.146), recorded 420 hits (31.842 = final_score). **Effect size: +0.30 points, +4 of 1,319.**
      - **How to read it**:
        - Both decodes ran the same 1,319 examples, so no sampling noise enters the difference. It is numerical, CPU bf16 against H100 kernels: at least 4 examples decode differently, net +4.
        - The first version of this line set it against the workload's binomial SE (1.28, z = 0.24) and called that z conservative. That was wrong. That SE answers a different question, whether a fresh sample of the workload would differ.
        - A paired SE isn't possible, because MLS-Bench publishes no per-example H100 vector. The number to quote is the effect size.
      - **Split:** first 50: 28.0 (14 hits); positions 50–1318: 32.31 (410 hits).
      - **Integrity:** every example's key matches gpu-tasks' order table at its position.
      - **Slices:** 13, tiling [0, 1319). [0, 50) is the head-50 slice already imported, harness 03894156. The other 12 ran harness da13b1a8. Box image c7579a39, evaluator e4a833bd.
      - **Cost:** 24.1 process-hours. The record's recorded H100 wall is 13,461 s.
      - **Files:**
        - merged: `gpu-tasks/survey/mlskv-fulln-kivi-gsm8k-2026-10-29/mls-llm-kv-adaptive-quantization-kivi-overlap-4bit-gsm8k-s42.json`, sha256 f63a3d24;
        - slices, checksummed against c3-13: `slices/`;
        - the check: `fulln-check.json`, from `tests/r6_fulln_check.py`.
      - **What this shows:** on one whole workload, CPU bf16 reproduces the H100 recording to 0.30 points (420 → 424 of 1,319).
      - **What it doesn't show:**
        - It's one method, on GSM8K.
        - G4's decision binds on passage retrieval, against full_attention, and this measures neither.
      - **Not imported**: with k = n the served score would read 32.146, not the recorded 31.842, and the importer correctly refuses it. The directory's README.md gives the reasons; it stays in survey/ as evidence.
- **LongBench:** hotpotqa, passage_retrieval and repobench (8 records each) and longbench_v2 (4), 28 records in all.
  - Pass 1 sets the head to 20 (72% of collected LongBench launches).
  - Pass 2 extends it to 50 (95%+).
  - Analysis ruling, 2026-10-29 10:40 MDT.
- A record whose head isn't measured yet keeps its synthesised vector, labelled as on r4.

**Decodes never run on a machine with a live agent run**: decode load would change the host CPU the agent sees.

**Fix keys for the runner:**
- new: `fix:gpumock-measured-first-k`;
- carried: every r4 key;
- dropped: none.

**Tests:**
- `tests/test_measured_scores_daemon.py` 13/13:
  - an unchanged vector without a block;
  - the measured head served and the mean kept;
  - subset, full and keyed-shard launches, including one straddling k;
  - labels; the clamp case; a malformed block ignored; the start-up event.
- `tests/test_measured_in_image.sh` 5/5 on the r4 KV box with the patch mounted:
  - `--max-examples 5` prints the measured first 5's mean;
  - the whole workload prints 32.5 in 40 s;
  - a shard [8, 12) gets measured, then synthesised;
  - a record without a block is served as on r4;
  - the start-up event.
- 16 controller daemon suites pass.

## Release r7.1: the lookup reply carries no result (2026-11-02)

**Status:** frozen at `/srv/work/gpu-mock/next-r7.1` on the E2 and archived, sidecar-only. Analysis ruled on 11-02 to ship it alone, rolling at run boundaries. Their message was stamped 08:17 MST, a stamp they later said ran ahead of the clock; `date` read about 08:09 MST. It goes to isolation's build, then the gate `tests/gate_sequence_r71.sh` with REFTAG 62ed.

**Hashes:**
- patch `cd4c9ea36f2085b9735227a757a978608f028650f7087a5c9c8c6fca3374f0ba`, r7's, unchanged;
- sidecar `18dab16437f21cbebfb8e45cec34c72b37712dc02b40d2949acc2ba511fc07b9`;
- install `3fb6bc03ec3798c96ba7f076be814e20b3d2c0e5291625f2eca3182b56521714`, unchanged;
- archive `archive/next-cd4c9ea36f2085b9735227a757a978608f028650f7087a5c9c8c6fca3374f0ba-sidecar-18dab16437f21cbe.tar.gz`.

**The defect, mine, found 11-02 while starting the whole-job mode.**
- r2 (10-27) removed `final_score` from the lookup reply, because the Grok canary read it there without running anything.
- r6 (10-29) added `per_example_scores`, the record's 50 measured scores, to the KV records' serving block, and the reply carried that block whole. So any client that reached the daemon socket and sent `lookup` read 50 real scores per KV record, unpaced.
- The same block also carried `decode` (the workload and example count), `results_file` and `final_print`. The reply's failure entries carried each failure's `cause`, `exception`, `trigger` and `threshold`.
- The patch read none of these. **Runner's scan: clean.** No run read `per_example_scores` without running an evaluation. 131 of 141 r6/r7 runs show nothing; the other 10 are served-OOM lookups, one exited job, one known crash run, and one note naming the socket without connecting. Nothing is voided.

**The fix:**
- **The reply's per_example block is an allowlist:** `mode`, `generate_seam`, `score_seam`, `n_examples_full`, `score_kind`, `subset_source`, `process_model`.
- **Each failure entry carries only** `mode`, `where`, `replay` and `at`, plus `armed` when the lookup arms it. That's what the patch reads to decide whether to poll `fail?` and at which seam.
- **Where results go:** scores reach a job only through `materialise`, gated on its paced unit as since r2b. Failure text and steps reach it only through `fail?`, at its step.
- **Why an allowlist:** a result field added to a record later can't reach a client unpaced.

**What changes for a job that runs its script: nothing.** The patch reads only allowlisted fields, so every served value, time and failure is r7's. Gated by r6's pair comparison against r7's reference run.

**Fix keys:** new `fix:gpumock-lookup-no-results`; every r7 key carried; none dropped.

**Tests:**
- `tests/test_lookup_no_results_daemon.py`, 5/5 on r7.1 and 1/5 on r7's daemon. On all 1,964 registry records, it checks:
  - no reply carries a result-bearing key, searched recursively by name;
  - the 40 KV replies carry the seam fields and none of the block's results;
  - the 46 failure entries carry only the four fields;
  - through a live daemon, a KV lookup's reply has no measured score, and the same record still serves its measured score (22.22, nonzero) after example 0 is paced.
- 17 controller daemon suites pass.

## Release r7: no-CUDA launches run on the CPU; the GPU count from the profile; sharded memory (2026-10-31)

**Status:** frozen at `/srv/work/gpu-mock/next-r7` on the E2, archived and noted in `archive/INDEX.tsv`. Analysis ruled all three changes at 14:18 MDT on 10-31 and accepted the note at 14:44 MDT.
- **The 8-GPU images, for new packages only.** isolation built them on 10-31 between 14:45 and 14:56 MDT. `tests/gate_r7_8gpu.sh` passed 9/9 at 15:23 MDT; its first run's two failures were faults in the gate's own checks. The images:
  - packages box `ml-node:py311-torch2110-8gpu`, `sha256:3e515dec6d13826fd7094da24996980f1a016bed773dfb3269a89b0b05741795`;
  - KV box `ml-node:py311-torch2110-mlskv-8gpu`, `sha256:55c980f8c202ae281dd7c5de812ef4533f0d7f1564eca9658e9bcae7783bf3ee`;
  - sidecar `gpu-daemon:py311-torch2110-8gpu`, `sha256:a60c2b89021c45b64ed066de80dd145271770329301e93ae3f76bb886f5e8c0e`.
- **The 4-GPU fleet images.** isolation built them on 10-31 between 15:32 and 15:55 MDT. `tests/gate_sequence_r7.sh` (`gm-gate-62ed`) passed at 16:59 MDT: r7's own tests first, then r6's whole sequence unchanged, including the pair gate's same served values and unchanged recorded total. The images:
  - packages `sha256:028ef5c0d049a4ef89f0a1ffe13cd929dbaf564ec667481f6b65f5b402db7e53`, sidecar `sha256:1cbfe87d5492df891709351ed02afea0dbd8e50a1308e71c2b7c65e23e631585`;
  - KV `sha256:bc564e9251d17cbf7de2f69a72290835b945e9a041862c377561a34a623279ea`, same sidecar;
  - dataprep `sha256:de9d239ea697bdcd25a4e62937f6e7ec6962b2f525394644cbbcad7f3e56bd44`, sidecar `sha256:0dac3948c2a938fbde38759149831c04d843355232c0d1fec3ec3712ba7b2826`;
  - filter `sha256:a99dda54cd7fee29a66c6b090eda5b2fb4edecdfb995b1c469c7051a4cb5eb47`, sidecar `sha256:779b30ff88f25745e80b19a51a8c918287d912d1c789885fcb8d38bf0d6c7c87`.
  - **Shipping:** a rolling ship at each run's end, on the study's word at 16:26 MDT (all 5 Sol L0 lines claimed by 16:16 MDT). r6 is kept as `-e8b3`.

**Hashes:**
- patch: `cd4c9ea36f2085b9735227a757a978608f028650f7087a5c9c8c6fca3374f0ba`
- sidecar: `62ed48f03eb597515710135e6d4d8be2674ffdf76dc3edb9f9bd166ecbfaa623`
- install: `3fb6bc03ec3798c96ba7f076be814e20b3d2c0e5291625f2eca3182b56521714`
- archive: `archive/next-cd4c9ea36f2085b9735227a757a978608f028650f7087a5c9c8c6fca3374f0ba.tar.gz`

**What changes for a job that runs its script as recorded: nothing.** All three changes are reached only by a launch the recorded runs never made, or by a record field no existing record has. In detail:

1. **A launch that sees no CUDA device runs on the CPU, as on hardware.** This is the fix held since 10-29, described in the section below. It applies when `CUDA_VISIBLE_DEVICES` is empty or names no card the box has. Such a launch gets no replay, so it runs the real CPU decode with its real wall time instead of crashing at the stand-in. The daemon logs `replay_disengaged` with reason `no_visible_cuda_device` and the variable's value. A launch with any visible card is served as on r6.
2. **The box's GPU count is its NVML profile's.**
   - **How:** `install_into_image.sh` counts the profile's devices and writes the count into the installed `cuda_hooks.py`. `torch.cuda.device_count()`, `nvidia-smi -L` and the visible-device parsing then agree on any box.
   - **Every 4-GPU box:** its profile has 4, so 4 is written, which is r6's value.
   - **The 8-GPU image:** reads 8 everywhere, and `CUDA_VISIBLE_DEVICES=4..7` are real cards.
   - **The other pieces of that image are isolation's:** the daemon's `--gpus` (from `task.box.gpus`), the device nodes (`/etc/box-gpus`) and the 8-device profile.
3. **`memory.sharded_bytes` in convert().**
   - **The rule:** a record may say how many of its per-card peak bytes are state sharded across its recorded GPUs (ZeRO-3/FSDP). On a launch of another width, that part scales by `recorded / launch` and the rest of the peak stays as recorded. Weights held at engagement scale with their sharded share.
   - **Its effect:** a launch too narrow for the model holds more than the card and gets the existing out-of-memory error at its first unit, in PyTorch's wording.
   - **Time is untouched.** The batch policy's factor applies as before.
   - **Memory isn't clamped at the recorded width.** More GPUs than recorded hold less per card, as on hardware.
   - **Existing records** don't have the field, so their memory is r6's.

**Fix keys for the runner:**
- new: `fix:gpumock-no-cuda-cpu`, `fix:gpumock-gpu-count-from-profile`, `fix:gpumock-sharded-memory`;
- carried: every r6 key;
- dropped: none.

**Tests:**
- `tests/test_no_cuda_in_image.sh`, 4/4 on the r6 KV box with the patch mounted (10-29). Empty and `=7` launches run on the CPU and print `TEST_METRICS`; `=0` and unset launches are served.
- `tests/test_gpu_count_in_image.sh`, 14/14 on the E2 (10-31). It installs the tree into a box from an 8-device and a 4-device profile, then checks:
  - the count written into the patch;
  - `nvidia-smi -L` and `torch.cuda.device_count()`;
  - the last card visible alone, and one past it visible as none;
  - the last card pacing on daemon GPU N−1, in three launch shapes.
- `tests/test_sharded_memory_in_image.sh` (10-31). Record: 40 GiB per card on 4 GPUs, 30 GiB of it sharded. Results:
  - 4 ranks hold 40 GiB each and complete;
  - 2 ranks hold 70 GiB each and complete;
  - 1 process holds 130 GiB and raises `torch.OutOfMemoryError` at its first step, exit 1;
  - the convert events log `memory_factor` 1.0, 1.75 and 3.25;
  - a record without the field, run at 1 GPU, completes.
- `tests/test_convert.py` 29/29, which adds 13 cases for sharded memory.
- The 16 controller daemon suites pass.

**For the gate:** r6's sequence on the 4-GPU images, then on the 8-GPU image:
- `nvidia-smi -L` = 8;
- `torch.cuda.device_count()` = 8;
- `ls /dev/nvidia[0-9]*` = 8, in a box started by run_one;
- a job on `CUDA_VISIBLE_DEVICES=7` pacing on daemon GPU 7.

## Held for after the finals: a launch that sees no CUDA device runs on the CPU (2026-10-29)

**Status: written and tested, not archived or shipped.** Analysis's ruling, 21:03 MDT: ship it after the finals unless a second run hits the case first.

**The case.** G-kvrep-haiku-L2's `run_all.sh` ran 40 background `run_one` readers on a 4-token FIFO pool. A reader that found the pool drained read an empty token and ran with `CUDA_VISIBLE_DEVICES=` empty. The run's own `logs/status.txt` shows `gpu=` empty on 32 of 40 lines, and those are the 32 launches that crashed. Each one:
- saw `torch.cuda.is_available()` False, as a driver would report it, so the evaluator took its CPU path;
- matched its record exactly, so replay engaged anyway;
- crashed at the per-example stand-in with `AttributeError: 'Tensor' object has no attribute 'past_key_values'`.

On real hardware these launches would have decoded on the CPU, slowly, and printed a score. This was reproduced on the r6 KV box 19fd91e1 with the real lagkv evaluator. Empty `CUDA_VISIBLE_DEVICES` gave this exact crash. `CUDA_VISIBLE_DEVICES=0` was served cleanly. Analysis files the class as an agent failure with this faithfulness gap labelled on it. r6 did not cause it: `_fix_visible_devices` hasn't changed since 3ec0.

**The fix.**
- `replay.try_lookup`: when the daemon answers `replay` but the process sees no CUDA device (`torch.cuda.is_available()` False, which `cuda_hooks` counts from `CUDA_VISIBLE_DEVICES` as a driver does), replay is not engaged. The patch sends `disengage` with reason `no_visible_cuda_device` and the variable's value.
- The daemon logs `replay_disengaged` with `cuda_visible_devices`, and the job is modelled.
- A launch with any visible device is served as before. So is an unset variable (all four visible).

**What changes for a job that runs its script as recorded:** nothing. Recorded launches see at least one GPU. Only a launch that hides every GPU changes: it runs the real CPU decode, the box's equivalent of a real CPU run, with its real wall time.

**Tests:**
- `tests/test_no_cuda_in_image.sh` on the r6 KV box with the patch mounted. It runs four launches of the real lagkv evaluator, gsm8k, `--max-examples 1`:
  - empty `CUDA_VISIBLE_DEVICES`;
  - `=7`, which names no GPU the box has;
  - `=0`;
  - unset.
- 4/4 on the E2, 2026-10-29 21:18 MDT, tree `/srv/work/gpu-mock/next-cvdfix`, `replay.py` sha256 4b61a2dd. The `TEST_METRICS` columns are final_score, mean_retained_fraction and runtime_seconds, as printed:

  | Launch | rc | wall | TEST_METRICS | Daemon events |
  |---|---|---|---|---|
  | `CUDA_VISIBLE_DEVICES=` | 0 | 175 s | 0.0, 0.191489, 168.5 s | `lookup exact`, then `replay_disengaged no_visible_cuda_device ''` |
  | `=7` | 0 | 171 s | 0.0, 0.191489, 164.8 s | `lookup exact`, then `replay_disengaged no_visible_cuda_device '7'` |
  | `=0` | 0 | 8 s | 0.0, 0.191489, 7.1 s | `lookup exact`, served |
  | unset | 0 | 8 s | 0.0, 0.191489, 7.1 s | `lookup exact`, served |

  No exception events.
- The CPU decode of example 0 agrees with the served value: score 0, retained fraction 0.191489.
- The 16 controller daemon suites pass.

## Package search T-mock: 8 GPUs, failures, mixed widths

**Status: answered to analysis at 14:15 MDT. Nothing built; analysis rules first.** These answers describe the code as of the r6 tree with the held no-CUDA fix, and are expected to move once anything is built. The plan is `experiments/package-search/AGENTS.md`.

**1. Can a box have 8 GPUs (angle F)? Not today.** The daemon handles `--gpus 8`, but four other layers fix 4:
- `cuda_hooks.DEVICE_COUNT` defaults to 4, and the box sets no `NVRT_*`;
- `isolation/image/build.sh` sets `GPUS=4` for the NVML profile (the upstream H100 profile has 8 devices);
- the OS surface takes the same `GPUS`;
- one box image is built per GPU count.

Cost: about 1 h on my side, either baking the count into the install or learning it from the daemon's hello. The rest is isolation's build and a gate. Nothing else changes at 8. Paced jobs spin about one core each while they wait, so 8 at once want 8 or more cores.

**2. Failures and relaunch (angle C).**

| Case | Answer | What's missing |
|---|---|---|
| (a) OOM above a batch size, set off by a flag | yes, built and tested | No OOM in the registry, so the text and threshold would be synthesised and disclosed |
| (b) a recorded TRL crash at its recorded step | the mechanism works | 15 of the 22 TRL failures are replay-ready with a step. None carries stderr, so the job exits silently. None has a trigger, so a relaunch crashes again. Nearly all fire at the record's last step |
| (c) two jobs stacked on one card | yes, built and tested | Records whose two peaks exceed 80 GB (TRL 8-GPU at 72–79 GB per card qualify; the KV records at 6–13 GB don't) |

**3. Mixed widths (angle G): yes, through per-card memory, with no time factor.** Sharded state (ZeRO-3/FSDP) splits across the cards, so per card it is `state × recorded_width / launch_width`, plus whatever isn't sharded. Two pieces of work:
- Mine: a `memory.sharded_bytes` field read by convert(), about 3 h with tests. The existing OOM at the first unit then fires on a launch too narrow for the model.
- gpu-tasks': the records, synthesised from a parameter count because the registry has no sharded record, and each job's feasible widths for the solver.

Neither 1 nor 3 changes the five packages: their box image keeps 4 GPUs and their records carry no sharded field.

## Serving `mteb run` (package search, survey S7): answers to gpu-tasks-mteb (2026-10-31 15:40 MDT)

**Status: nothing built.** The questions are in `experiments/gpu-tasks/notes/2026-10-31-mteb-package.md` §2. **Short answer:** it needs a new serving mode, so not this weekend. It costs about 2 days of my code and tests, plus isolation's box image and gpu-tasks' staging, and could start after Monday 11-02's screens if analysis ranks it in.

1. **A seam on an imported library: yes, and it's in scope for the `next/` release line.** 3ec0 is no longer the held instrument. The patch already patches library code after import (`_when_imported("transformers.trainer", ...)`), so a `module:qualname` seam on `mteb.abstasks.AbsTaskRetrieval.AbsTaskRetrieval._evaluate_subset` is the same pattern.
2. **Identity without a script: yes, mostly as is.** The exact key is the entry script's sha256 plus the argv. For `mteb run ...`, the entry is the console-script shim, which has the same bytes for every job, and the argv tells them apart. So records keyed that way match today.
   - `--output_folder`, `--batch_size`, `-v` and `--co2_tracker` go in `argv_exempt`.
   - Pacing ignores `--batch_size`, disclosed.
   - `python -m mteb run ...` has a different argv[0] (`mteb/__main__.py`), so it needs its own records, or r4's module stage.
   - The shim's bytes include its interpreter path, so the records must be built from the box's own shim.
3. **Pacing: per subset, in chunk units, by character share. Not at the encode seam.** Seam at `_evaluate_subset`, which is handed the subset's corpus and queries. Count their characters, pace one unit for the queries and one per 50,000-document chunk, each by its character share of the subset's recorded time. Never call encode.
   - **Full launch:** paces `evaluation_time` exactly.
   - **A `-l <lang>` launch:** needs the record to carry each subset's character count, because the box sees only the subset it runs. gpu-tasks counts these at build.
   - **The share** is a modelled allocation, disclosed.
   - **Why chunk units:** an agent's kill mid-task then frees the GPU at a chunk boundary, as on hardware, not after one multi-hour unit.
4. **Weights: real.** Model and dataset loads lie outside `evaluation_time`, so the box pays them in real host time. A meta-device load would make a 7B model load in seconds and hold none of its memory. The recorded `memory_usage_mb` (or the weights' size) is held on the card from load, as G1 does. That means about 60 GB of staging for three 7B models, within the disk.
5. **The score: a third mode,** `per_subset_scores`. `results_at_end` and `per_example` don't fit: the latter is generate and score seams per example in `__main__`. The seam returns the recorded per-subset dict verbatim, so mteb's own `TaskResult` writes the JSON. It's exact for a full launch and for any recorded language subset. A launch on a subset the record doesn't hold falls back to the modelled path, logged.
6. **Zero embeddings: not needed, and not wanted.** On the fake device, tensors are CPU-backed, so the search's `cos_sim` and `topk` would run as real CPU matmuls. That's queries × 50,000 × embed_dim per chunk, about 2×10¹² FLOPs per chunk for 10,000 queries at 4,096 dimensions, and it would land in the job's host time. Seaming above the search, at item 3's seam, avoids it.
7. **Hub loader-script datasets: leave them out.** The box has no network, and the loaders fetch from other hosts at load time. Pre-materialising them would change the dataset from the recorded revision.
8. **Cost:**
   - **Mine, about 2 days:** the mode, the subset seam, chunk pacing, the recorded-dict score, identity and fallbacks, daemon tests, and an in-image known answer on one small task.
   - **Others':** isolation's box image with mteb 1.18.0, sentence-transformers and pytrec-eval, about half a day. gpu-tasks stages the weights and datasets and writes records with per-subset character counts.
   - **Timing:** it can't finish before Monday. The angles already in flight come first.

## One generic "whole-job" mode for MTEB, MLS-Bench non-KV and cleanrl?

**Status: nothing built.**

**Short answer: one core mode, with three thin adapters, about 3 days in all.** The core is the same for all three sources:
- the job runs its recorded script;
- it is paced to `wall_total_s` (`timing.pacing: total`);
- what it prints is replaced by the recorded final result, at a seam the record names.

The adapters differ in three ways:
- where the seam is (a library function, a `__main__` function, or the script's own end-of-run print);
- how the GPU time is split into paced units;
- how a short or killed launch is told apart from a whole one.

**The core, which exists in part already:**
- `pacing: total` is defined in RECORD-FORMAT.
- `result.*` serving at the final materialisation ("final-metrics mode") is in the daemon.
- `per_example` already paces `(wall − start-up) / n` units and serves scores through a seam the record names.

The generic mode generalises that. The record names:
- `seam`: a `module:qualname`, or `__main__:name`;
- `unit`: what one paced unit is;
- `n_units`: how many there are, from the record or counted at the seam.

The patch:
- paces `wall_total_s / n_units` per unit, inside the seam;
- returns the recorded result from it;
- labels the result `result` and the time `total`.

A launch that ends early (killed, or `--total-timesteps` cut) has run fewer units, so it has spent its share of the time and gets no final result, as on hardware.

**What differs, per source:**

| | MTEB (`mteb run`) | MLS-Bench non-KV (rl-intrinsic-exploration, meta-inner-loop-optimizer, optimization-diagonal-net) | cleanrl (W&B runs) |
|---|---|---|---|
| The script | the installed library; the seam is in `mteb` | the task's own `custom_*.py`, an entry script like the KV records | `cleanrl/<algo>.py`, one file per algorithm |
| The seam for the result | `AbsTaskRetrieval._evaluate_subset`, which returns the recorded per-subset dict | each domain's own final-metric function or `TEST_METRICS` print; one seam name per domain, read from its script | the env loop: the final `episodic_return` is logged per episode, so serve a curve of returns over `global_step` (W&B history) and end at the recorded `_runtime` |
| What one unit is | 50k-document corpus chunks, by character share (above) | per domain: an epoch, an outer iteration, or one unit for the whole run | an iteration of the rollout-and-update loop (`num_iterations` is in the argv) |
| Per-step series | none needed | most leaderboards have none; a final metric only | needed: cleanrl prints `episodic_return` as it goes, and TensorBoard and W&B record it. Served from the history the source holds, or the run carries none, disclosed |
| The GPU work | encode, which must be stubbed | small nets, which the existing replay stubs | small nets on CPU-bound envs: Atari and MuJoCo envs run on the CPU for real |

**What's hard, and the real cost:**
1. **cleanrl's envs.** The env step is real CPU work on hardware, and a large part of a cleanrl run's wall: Atari at 8 envs is env-bound. If the box runs the envs for real and replays only the GPU forward and backward, the run's wall is real env time plus paced GPU time. That no longer matches `_runtime`, unless the pace subtracts the real env time, as `host_gap_s` does for KV. That is the same design, so it's about 1 day. Otherwise the env loop is stubbed too, which changes what the agent sees, so not recommended.
2. **MLS-Bench non-KV:** each domain's seam has to be read from its scripts, which aren't on the controller yet. They're at the same MLS-Bench commit gpu-tasks already reads. Per domain this is reading, not building: about half a day for the first and an hour or two for each after.
3. **MTEB:** as in the section above, about 2 days alone. With the generic core built once, its adapter is about 1 day.

**Cost, if all three are wanted:**

| Piece | Cost |
|---|---|
| Core mode with daemon tests | about 1 day |
| MTEB adapter | about 1 day |
| MLS-Bench non-KV adapter, first domain | about 0.5 day |
| MLS-Bench non-KV adapter, each further domain | 1–2 h |
| cleanrl adapter, with real envs and paced GPU time net of env time | about 1 day |
| In-image known answers, one per source | about 0.5 day |
| **Total for gpu-mock** | **about 3–4 days** |

Records, box images (mteb; gymnasium with Atari ROMs and MuJoCo; each MLS-Bench domain's requirements) and staging are gpu-tasks' and isolation's. Records come first: no seam can be read without the script, and no unit count without the record.

**Hidden lengths: yes, in all three.** In each, a job's length is a method × task interaction a reader can't predict from the files:
- MTEB: model × task corpus;
- MLS-Bench: method × env;
- cleanrl: algorithm × env, through env speed and episode lengths.

That is the property G1 has and H100 nanogpt doesn't.

## Self-sharded jobs: what is served, what the log shows, what serving them would cost

**The case read:** G-kvgate-astracc-L0 on c3-6 (r2b), read live at 2026-10-28 04:35Z.
- Its `run_gsm8k_shard.py` loads `methods/kivi_overlap_4bit/custom_quant_eval.py` with `importlib.util.spec_from_file_location` as a module named `kivi_overlap_shard_eval`, not as `__main__`.
- It captures `module.load_workload` and `module.score_prediction`.
- It replaces `load_workload` with one that slices `workload.examples[start:end]`, and `score_prediction` with a wrapper that calls the captured original and logs each score.
- It sets `sys.argv` to the evaluator's own argv and calls `module.main()`.
- Four shards: [0,330), [330,660), [660,990), [990,1319), one per GPU, all started 04:22:22Z.

**1. What the modelled path serves.** Real CPU inference.
- The scores are the real Qwen's outputs on the CPU, through FakeGPU's CPU-backed tensors, not the record's.
- The wall is CPU speed: each shard process sits at about 10 cores.
- The job's own printed runtime follows that wall. Opus's direct-call run printed `runtime_seconds=113.7` for 2 examples, against the record's 5.6 s each.
- The record's own pace for kivi gsm8k is 13,460.6 s / 1,319 = 10.2 s an example. A 330-example shard at record pace would take 3,368 s on its GPU.

**2. What the occupancy log shows of a modelled job: no GPU time and no memory.**
- A modelled job writes `job_hello`, `lookup` (match none) and `job_bye`, and nothing between: no `enqueue`, no `busy_start`/`busy_end`, no `hold`.
- Read from the logs of G-kvreph-haiku-L0-m3 jobs 5–7 and 9, G-kvreph-opus-L0 jobs 13 and 18–22, and live on astracc's jobs 24–27.
- So nvidia-smi shows 0% and 0 MiB on all four GPUs while the four shards run (read at 04:37Z).
- idle.py places such a job in `pre_gpu` (never reached the GPU), so its whole lifetime counts as GPU idle.
- 18 collected runs on the controller have modelled-path jobs over 60 s. G-kvreph-opus-L0-r2 has the most: 5 jobs, 24,459 s of process lifetime.
- **This is a scoring fact, not a serving defect.** The scorer can tell such a job apart: `lookup.match == "none"` with no enqueue, its lifetime from hello to bye. Whether that time is idle, waste or a disclosed limit is the runner's and analysis' to label.

**3. Serving an agent-cut partition as a slice of the record: about 2 days, not 1.** Four parts:
1. **Find the record when the evaluator is a module, not `__main__`** (about half a day). At lookup, the patch also sends the path and sha256 of each loaded project module, files under the launch directory and outside site-packages. The exact and path_exact stages try each as the entry. The argv already matches, since the shard script sets `sys.argv` to the evaluator's own.
2. **Install the seams before the agent captures the originals** (about half a day). The shard script reads `module.score_prediction` straight after `exec_module`, before any CUDA call, so a seam installed at first CUDA touch is too late: the agent's wrapper calls the captured original. The fix is an import hook on any module whose file hash is a record's entry hash. It wraps the seam functions as soon as the module executes, as thin shims that call through unless a per_example record engages. The agent then captures our shim.
3. **Map a shard's local example to the record's global position** (about half a day to a day). The slice reaches the seams as a list of example dicts. The daemon needs each workload's example order: the sha256 of each example as its loader emits it, computed once from the staged data. Then a served score is looked up by the example's content rather than its local index.
   - This is also the base of r6 (real per-example outcomes keyed by example). It needs gpu-tasks: the order tables are theirs to stage.
4. **Known answers across shard styles** (half a day): `load_workload` slicing like astracc's, a `--max-examples` prefix, a re-implemented loop.

**Pacing would then be the record's:** each shard's examples at 10.2 s on its own GPU, about 56 min for four shards on four GPUs. That's a real speed-up over the 3.7 h single run, as on hardware.

**Placement: analysis' call, since it's over a day.** My read: it affects the frontier cells (Opus, Fable, Astra) and their bar figures, while Fix B and power and clocks move none. So it should come before them, as r4, with gpu-tasks' order tables. Fix B and power and clocks would move to r5.

## whole_job serves its result ONCE — a metric read at every evaluation needs the eval-series path (K2 owner, 2026-11-02)

**The finding (generic, surfaced on SSD, retained after SSD was withdrawn):** r8's whole_job
serves the record's recorded result exactly once — `_materialise_whole_job` returns it only
after the job has paced all `n_units` in order and the last unit's GPU time has passed, then
sets `wj_served`. The seam's `replacement` also calls `_pace_whole_job(wj)` on **every** call.
So a whole_job seam on a function the script calls at **every evaluation** (e.g. SSD's
`get_stats_from_evaluator`, called 3–4× per run, once per epoch) is wrong two ways:
- at the first call (epoch 1) the units aren't all paced, so it returns None (`read_ahead`),
  and if forced it would hand back the **final** AP — so an early-stop threshold
  (`target_map`) fires at the wrong epoch;
- if it paced the full record on each call, it would **double-count** the wall.

**The right mechanism for a repeated eval metric is the eval-series path**, not whole_job: the
metric is a sparse series indexed by step/epoch, served `at_or_before` the current step
(`_series_value_at(ser, step, "at_or_before")`), None before its first recorded validation
(the r8 `eval_unreached` gate). The epoch-1 call then reads epoch-1's recorded value, epoch-3
reads epoch-3's — the real per-call trajectory, never a future value.

**What r8 does NOT yet have:** the eval-series path serves a metric the script reads off a
**tensor** (`.item()`/float on a role-tagged forward output). SSD's AP is a **CPU-evaluator
return** (pycocotools) at a seamed function. Serving the per-call trajectory there would need a
small new adapter — an **eval-series seam** that returns the series value at the current eval
index from a seamed CPU function (keep train per-optimizer-step pacing, substitute only the
stats return). Not built, because SSD was withdrawn (analysis: 40 of 50 needed leaves ~1.25×
headroom, not a robust bar-4). **Terminal-only CPU metrics still fit plain whole_job** (one
final value, served once after full pacing) — that's the clean case. Build the eval-series seam
only if a future package needs a CPU-evaluator metric's full trajectory. Disclosure rule either
way: never pretend the source eval trajectory ran if replay served it coarsely.

## r8 whole_job LIBRARY seam fails when the script imports torch LAZILY inside the seamed function (TTC real entry, 2026-11-02)

**Confirmed by the real TabArena single_cell.py entry** on isolation's native-FakeGPU
image (c3-15, logs at /srv/work/gpu-mock/ttcreal-keep). The whole_job early probe matched
the record (`match=exact`) and engaged, but immediately withdrew with
`whole_job_seam_already_running`: `ExperimentRunner._run` was already on the stack when the
seam install ran. The real `_run` then executed → `init_method()` (experiment_runner.py:232,
INSIDE `_run`) → TabPFN model load → gated-HF error. The weight error is a DOWNSTREAM
symptom; a working seam never reaches `init_method`, so replay needs no weights.

**Root cause (from the installed source):** neither single_cell.py nor experiment_runner.py
imports torch at top level. The first `import torch` in the process happens LAZILY, inside
`_run` (init_method → TabPFN imports torch). The whole_job early probe is wired to the
torch-import hook (`cuda_hooks.apply()` runs "the moment torch finishes importing",
sitecustomize), so it fires inside `_run` — after `experiment_runner` is already imported AND
`_run` is already running. `_install_whole_job_seam` then finds `_run` on the stack and the
`running()` guard correctly refuses to replace a live frame.

**Why the in-image test's `wj-lib` case passed and hid this:** that stand-in imports torch at
top level, BEFORE the library module with the seam, so the probe fires before the seamed call.
A stand-in with import-torch-before-library cannot expose this; the real TTC entry defers torch
into the seam. (Sol predicted this exactly.)

**The fix (r8, real, non-trivial — NOT a mechanical review fix):** a LIBRARY whole_job seam must
be installed when its module is IMPORTED (which precedes the seamed call), independent of when
torch is imported. Today the install is deferred to the torch-import hook. The seam-module
import hook must be armed at sitecustomize time (before any user import), decoupled from torch.
Constraint: `replay.py` imports torch at module top, so the pre-torch arming needs a lightweight
hook that triggers the (torch-dependent) install only when the seam module loads. This is design
surgery on a frozen/shipping instrument and on the exact case the TTC package exists for, so it
is surfaced to the owners (Sol/isolation) rather than patched unilaterally mid-flight.

**Not affected:** step-series and per_example paths (proven on 2.11); whole_job records whose
script imports torch before the seam module's call (the stand-in case). The 144 sol-ttc records'
field names still match r8; no rebuild needed. The "whole_job + import-time seam" readiness
`missing` entry is genuinely open — this bug is behind it.

## r8.1 shipped; the real TTC 805f entry serves end-to-end (2026-11-02)

r8.1 (patch 4c17e9d5, sidecar 2da8801c) = r8 + the lazy-torch LIBRARY whole_job seam fix
(torch-free whole_job_probe.py arms at sitecustomize via a STATELESS daemon peek; the daemon
gained a no-hello whole_job_probe lookup path). The real Sol 805f single_cell entry (sha
805f54de) on isolation's native FakeGPU image, r8.1 mounted, UID 1000, original OpenML cache
READ-ONLY, no shim, no TabPFN-3 weights: rc=0, seam installed at experiment_runner's import
BEFORE _run, all 82 units paced, recorded final_metrics served (roc_auc metric_error 0.11013,
time_train_s 4018.0, time_infer_s 870.4), no gated-HF error (init_method never ran). Both
sol-ttc readiness.replay blockers answerable. Logs /srv/work/gpu-mock/ttc805f-keep.

## NeMo (NGC 24.09 / torch 2.5.0a0 / CUDA 12.6) exposed r8 patch-fidelity gaps the H100/2.11
## boxes never hit (Astra independent, 2026-11-02). Three resolved by config, one is a real code fix.

The patch's fidelity surface was validated on H100 / torch 2.11 / CUDA 13.0 boxes. The NeMo
source stack is a different torch+CUDA combo and surfaced four issues, in import order:
1. **`@torch.jit.script` + the varargs stub.** NeMo's `megatron/utils.py erf_gelu` is
   `@torch.jit.script`; TorchScript traces into the patch's pin_memory factory wrapper
   `cuda_hooks.py _pin_memory_factory_wrap -> def f(*a, **k)` and raises NotSupportedError
   (varargs). **Workaround (disclosed, probe-only): PYTORCH_JIT=0.** A real code fix would make
   the factory wrapper invisible to TorchScript (concrete signature or @torch.jit.ignore);
   affects ANY TorchScript package, not just NeMo. NOT YET FIXED — tracked here.
2. **`torch.version.cuda` default 13.0 vs the image's compiled 12.6.** torchvision refuses the
   mismatch. NOT a bug: cuda_hooks.CUDA_VERSION reads NVRT_CUDA_VERSION (and FAKEGPU_CUDA_VERSION).
   **Per-image config: set NVRT_CUDA_VERSION / FAKEGPU_CUDA_VERSION to the image's real CUDA**
   (12.6 for nv24.09). The NeMo box recipe must set 12.6; 13.0 is only the H100/2.11 default.
3. **sitecustomize precedence on NGC images.** NGC 24.09 imports `/usr/lib/python3.10/
   sitecustomize.py` BEFORE site-packages', so a mount-over onto site-packages leaves gpu-mock
   unimported. **Fix: prepend /usr/local/nvidia/python to PYTHONPATH** so the mounted patch
   sitecustomize is selected (the normal install bakes it correctly; this bites a mount-over only).
4. **Distributed SUBGROUP rank (PP=1, TP=4), a REAL code fix.** NeMo's init_model_parallel asserts
   app_state.pipeline_model_parallel_rank == parallel_state.get_pipeline_model_parallel_rank; 3 of
   4 ranks assert and the 4th's gloo barrier loses peers. dist_hooks wraps init_process_group /
   all_reduce / barrier / broadcast / all_gather / get_backend but NOT new_group / get_rank(group)
   / get_world_size(group); for a singleton PP subgroup the unwrapped path reports the WORLD rank
   instead of the subgroup rank. Astra is confirming with a patched-vs-unpatched 4-rank
   singleton-group known answer. STATUS (corrected): Astra's first minimal tests imported the
   patch but never touched CUDA, so the lazy distributed hooks never engaged -- the gloo 16/16
   "control" was a pre-CUDA no-op, NOT a refutation, and the NCCL variant failed natively on no
   GPU. So the subgroup cause is STILL OPEN, not discarded. Astra is redoing the diagnostic to
   make a first CUDA touch with a live daemon BEFORE creating groups (as NeMo does). The original
   full-source TP4 assertion failure remains the valid fact. Do NOT implement a fix until the
   corrected known answer isolates the exact missing group-aware call. NOT YET FIXED, cause not
   yet confirmed.

## r8.3: whole_job live_fields

A whole_job record's `materialisation.serving.live_fields` = `{result_dict_key: "<allowlisted getter>"}`.
After the units pace, the patch reads the named LIVE instrument IN the seamed context and merges
it into the served dict, but NEVER overrides a key the recorded result carries (recorded result
wins). Getters: `max_memory_allocated` / `max_memory_reserved` / `memory_allocated` (live torch.cuda),
`seam_elapsed_s` (the real wall the paced seam took), `recorded_peak_bytes` / `recorded_peak_mb`
(the modelled held peak = record's memory.peak_bytes, bytes or MiB).

**The late-pole (k2-latepole) use:** quality (acc/density/final_score/reuse/tps) copied RECORDED;
`peak_memory_mb` = `recorded_peak_mb`, sparse `elapsed` = `seam_elapsed_s`. Pure extraction: the
batch helper keeps the memory/timer code live outside the seam; only the quality producers are
served. K2's units catch: bind an MB field to `recorded_peak_mb`, not `recorded_peak_bytes`
(bytes → MB field inflates 1,048,576x).

**Why a live max_memory_allocated is 0 under whole_job, and analysis's ruling.** For a library
seam, replay ENGAGES at the seam module's import, which runs `_disable_device_memory_tracking()`
BEFORE the outer `model.cuda()` allocates, so `torch.cuda.max_memory_allocated` reads 0 (tracking
off). The nonzero figure is the modelled held peak (`recorded_peak_bytes`, = what `_hold_weights`
holds and nvidia-smi shows). Analysis ruled (2026-11-02) use `recorded_peak_bytes`/`_mb`: memory is
the instrument, not a recorded *result*; every box memory reading already comes from the record's
memory block (that is how nvidia-smi shows used), so this is the same instrument, nonzero and
plausible (weights+cache). Done-when met: recorded_peak_mb == the held peak nvidia-smi shows by
construction (both _recorded_mem_bytes), never above the card, scaled by convert()'s memory_factor.

**Modelled memory for the two late-pole models (gpu-mock GPU-fidelity call, anchored to the
source vendor notes):** LLaDA-8B BF16 (DLM) ≈ 16.5 GiB peak = 17,746,000,000 bytes (weights
~16 GB per LLaDA/config.json note + logits + activations, no KV cache); Qwen2.5-1.5B FP16 (sparse,
8192 ctx, no cache) ≈ 4.1 GiB = 4,380,000,000 bytes (weights ~3 GB per the sparse config note +
8192-ctx activations + sparse scratch). Disclosed as modelled (published weight size + modelled
working set), not a recorded measurement.

Validated on the 2.11 box: live_fields (score recorded/not-overridden, peak 128.06 MB for a 128 MB
model, elapsed live), lazy-torch serves RECORDED, per_example 17/17, subgroup 4/4.

## QUEUED WORK ORDERS (scoped + contracts final, 2026-11-02)

**Status 14:00 MST: r8.5 BUILT + archived (patch 90e9b7b1, sidecar 25c3e722) with binds (B) env, (C) seam module, (D) runtime — see INDEX. (A) NeMo backward moved to r8.6, still open.**

**Status 14:36 MST: r8.5.1 BUILT + archived — the whole_job slice of the helper-dependency bind, as FILE binds (serving.module_shas: task-root-relative path -> sha256, hashed on disk at seam install, imported or not, so lazy imports are bound). The all-paths version (per_example / step-series / final-metrics) is still queued after r8.6, on the same file-keyed contract.**

### r8.5 — three fixes, each with a known-answer test (build next)
Current tip is r8.4 (patch c4fe2602, sidecar a77dbe2e) on the E2 next-r8 tree and local src/.

**(A) NeMo backward / grad-readiness (Astra lane).** Megatron's distributed optimizer asserts in
param_and_grad_buffer.finish_grad_sync: "N/M params have grad available" (0/8). Cause (confirmed,
Astra subgroup/backward probes): with config.deallocate_pipeline_outputs TRUE (measured True on all
4 ranks), Megatron's `custom_backward` (schedules.py) calls `Variable._execution_engine.run_backward`
DIRECTLY, bypassing torch.autograd.backward — so replay's backward stub (which hooks
FakeCudaTensor.backward / Tensor.backward) never runs, AND the per-param grad-accumulation hooks
(distributed_data_parallel.py _make_param_hook → param_to_buffer[param].register_grad_ready(param),
registered on each param's AccumulateGrad node via grad_acc.register_hook) never fire, so
params_with_grad stays empty. Fix: intercept Variable._execution_engine.run_backward; under replay,
run backward_stub (set .grad) AND fire each registered grad-acc hook so Megatron's own param_hook
runs (main_grad.add_(grad); register_grad_ready). Do NOT remove the assert or flip readiness flags.
AccumulateGrad is a C builtin (can't class-patch register_hook) — capture approach: intercept at hook
registration or find the DDP instances (they hold grad_accs + param_to_buffer) and fire their hooks.
A gc-walk prototype of "find DDP via gc, set grad, fire hooks" hit a tool-refusal interrupt mid-build;
resume carefully. Astra's probe_nemo_steps.py has --patch-tree so Astra reruns the fixture on r8.5.

**(B) whole_job env binding.** The whole_job early probe sends env={},subset_env={} and the exact
match ignores env, so DLM_MAX_EXAMPLES=2 wrongly matches the full record → full aggregate at full
duration. Fix: the probe sends the job's real env; the match, for each key in the record's
identity.env minus env_exempt, compares job value to declared, treating UNSET job var == declared
default (K2: don't reject bare launches on missing-vs-default). Mismatch → no whole_job serving →
model path. DLM identity.env = {DLM_MAX_EXAMPLES:'0',DLM_MAX_PROMPTS:'0',DLM_APPLY_CHAT_TEMPLATE:'1'}.
Known answer: DLM_MAX_EXAMPLES=2 → model path (not full aggregate); bare launch → serves.

**(C) whole_job seam-module content binding.** The match binds entry_hash only; the seam module
(dlm_vanilla_uncached.py / sparse_streaming_llm.py) is unbound — an edited helper with the same entry
would serve. Fix: the record carries `materialisation.serving.seam_sha256` = sha256 of the seam
module's source bytes; at seam install (when the module is imported, torch up) hash the seam module's
__file__ and require equality, else withdraw replay → model path. Known answer: edited evaluate_batch
→ no match; unedited → serves. Box for TTC/late-pole must be r8.5+.

**(D) whole_job runtime-identity binding (analysis ruled 2026-11-02: runtime identity belongs in
the match, NOT a source version check).** The record is tied to a runtime (K2's B config:
transformers 4.46.3, in `identity.trajectory.environment`). A plain-python launch on a different
runtime must get model path, not the recorded aggregate. Read the TRUE installed version via
`importlib.metadata.version(pkg)` for each package key in identity.trajectory.environment (skip
source_commit) — NOT an env marker (an inherited env marker can lie about the interpreter; the
inherited-env-fakes-an-arm hazard). Torch-free → runs at the pre-torch probe. Mismatch on any declared
package → no whole_job serving → model path. Live check: the DLM record declares transformers 4.46.3;
Astra's current NeMo image has importlib.metadata.version('transformers')==4.44.2, so r8.5 correctly
refuses to serve it there → the serving box must carry the recorded runtime (4.46.3). Known answer:
wrong transformers version → model path; matching → serves. source_commit stays provenance-only
(not checkable from the installed interpreter) unless bound via a staged-tree hash.

### Older-path in-process patch bind
Runner's scan (~/.p006-runner/helper-edit-scan/REPORT.md) found agents patching the evaluator IN MEMORY
after load (9 MLS-KV runs); G-kvreph-opus-A3 job 44 replaced quantize_key/value with an identity and was
served the 4-bit record (cancelled at 85 s, never read, labelled). Design: per launch, at the score/generate
seam install and again at the first served example, compare the evaluator module's named functions (the
quantiser/scorer the serving block names) against the ones compiled from the bound file — code object
co_code + co_consts hash. A function replaced in-process is a mismatch → the job runs its own code. Ships
with the all-paths file bind; known answer modelled on job 44.
Checked 2026-11-02 15:05: G-kvgate-sol-L0-r3 jobs 20–22 (strided shard_loader i::3) were served each
slice's own examples by content: positions == the agent's shard manifests, union 1319 once, every value ==
_score_vector(record) at its position, union mean 31.8423 == record final_score. No label.

### Post-r8.5 — helper-dependency bind, option (b), ALL paths
Bind EVERY imported project module, not just the keyed entry. The patch already sends
modules=_loaded_modules() (name/path/sha256) and the daemon receives it but only uses it to FIND a
record (exact_module). Build: records carry identity.module_shas={path_or_name: sha256} for every
project module the entry imports (gpu-tasks derives from staged trees); the daemon, after an
exact/exact_seed/series/per_example/whole_job match, requires every project module in the lookup's
modules to equal the declared sha, else match none (helper_mismatch) → model path. Option (b): send
ALL imported project modules (walk sys.modules for project files), so a module imported-but-not-yet-
called is bound too (not just those with frames on the stack — _project_namespaces lists only stacked
ones). Est: ~1 day me + ~half a day gpu-tasks. Done-when: per-path known answer (edited helper →
model path, unedited → served); imported-but-not-called module bound; r6/r7 pair comparison UNCHANGED
on unedited trees (same served values, same paced walls). Roll out at run boundaries, no running run
stopped; earlier runs valid unless runner's scan flags one.

## Release r8.7: NeMo's optimizer step, a read-only whole_job peek (2026-11-02)

**Status:** frozen and archived (patch `307674a4f651b5a2`, sidecar `1b2bd3ac38464b56`, install `3fb6bc03ec3798c9`, unchanged). The box floor for any serving run on r8 is now r8.7: **r8.1 to r8.6 must not serve a run**.

**(1) The pre-import peek drew a record (bug since r8.1).** `whole_job_probe.arm` runs at sitecustomize in every python launch with an entry script. Its stateless daemon peek called `Registry.lookup()`, which draws, so every launch consumed one record before its own lookup: seedless launches skipped records, a named seed came back `exact_seed_reused`, a one-record group `recycled`. Found by `tests/test_trials_in_image.sh`: 10/15 on r8.6 and on the r8.7 working tree, 15/15 on the r7.1 box's installed patch, 15/15 on r8.7 with `NVRT_NO_PREIMPORT=1`. Fix: `Registry.peek_whole_job()` reads the exact / exact-exempt / effective / path_exact / path pools read-only (no draw, no used or served set, no cached job draw) and returns the first drawable whole_job record. The trials test was missing from the r8.x regression runs; it is in every gate from r8.7 on.

**(2) NeMo's optimizer step (Astra's TP4 fixture on r8.6, 2026-11-03).** r8.6 got past the backward (all 4 ranks fired the hooks of 8 parameters), then every rank died in `McoreDistributedOptimizer.step(closure)` → Megatron `optimizer.step()` → `clip_grad_norm` → TransformerEngine `multi_tensor_applier`: "expected input to be on cuda". 0 optimizer steps were paced. Two causes:
- NeMo's wrapper subclasses torch's Optimizer but never calls `Optimizer.__init__`, so it is never neutralised; the Megatron optimizer it drives is not a torch Optimizer at all.
- A neutralised step skipped the closure. Lightning's automatic optimization runs the training step and its backward INSIDE the closure, so a Lightning job paced one step for its whole run.

Fixes: `_run_closure` — a neutralised step under replay runs the closure (its forward and backward are the stubs) and returns its loss. `_patch_megatron_optimizer_module` — when `megatron.core.optimizer.optimizer` loads under replay, every `MegatronOptimizer` subclass that defines `step` (those defined now and later, through `__init_subclass__`) paces one step and runs no arithmetic, returning `(True, grad_norm, None)`: the record's `grad_norm` when it has the series, else 0.0, as the clip_grad_norm_ stub. Withdrawn replay runs the original.

**Known answer** `tests/test_megatron_optimizer_in_image.sh` (NeMo's wrapper verbatim in shape, a Megatron module tree, DistributedOptimizer imported after engagement, a Lightning-style closure loop under an LR scheduler): MEG — 4 closures (4 forwards, 4 backwards), 0 real Megatron steps, 0 DistributedOptimizer steps, 4 paced, each step returns the loss; MEG0 (replay off) — the real steps run 4 and 4; PL (plain SGD with a closure) — 4 closures, 4 paced.

**Regression on the 2.11 box (00930fbb):** trials 15/15, per_example 17/17, measured 5/5, nan_guard 22/22, optimizer_neutralise 5/5, resume 15/15, subgroup 4/4, engine backward (ENG/LB/ENG0, 8/8 buckets), whole_job (same FINALs and disengages as r8.6), whole_job identity B/E/D/B2/R/M/X and file binds B/P/L/O/B2 unchanged, lazy-torch RECORDED, live_fields as r8.3.

**Not covered by the stand-in:** Megatron's real `DistributedOptimizer` internals (param-gather hooks after the step, `finish_param_sync`), and NeMo's in-training validation. Astra's bounded TP4 rerun is the test.

## Release r8.8: a served value survives a device move; the torch-time probe says no hello (2026-11-02)

**Status:** frozen and archived (patch `357333e24622c95c`, sidecar `1b2bd3ac38464b56` = r8.7's, install unchanged). Supersedes r8.7 as the r8 floor.

**(1) A served value lost on `.cuda()` (Astra's TP4 fixture on r8.7, 2026-11-03).** The daemon served val_loss 0.94 at step 2 (64 reads), yet MLPerf's callback logged eval_accuracy 0.0 and stopped with success at the first validation. Cause: FakeGPU's `Tensor.to` / `.cuda()` / `.cpu()` call torch's original on the inner CPU tensor and wrap the result; they never pass through `__torch_function__`, so the role tag every other op's result keeps was dropped. NeMo's `inference_epoch_end` ends `avg_loss.type(torch.float32).cuda()`, and from there the broadcast, `self.log`, Lightning's reduction and `.item()` read the stand-in's zero. Fix: `_install_device_move_roles` — under replay a tagged tensor moved to the fake GPU keeps its role; a tagged 0-d float copied to the host holds the served value, read at the copy (a D2H copy is the sync on hardware). Untagged tensors and every move without replay unchanged. Known answer `tests/test_device_move_role_in_image.sh` (NeMo's non-drop-last reduction, a Lightning-shaped `_ResultMetric`, MLPerf's stop rule): r8.8 step 2 reads 0.94 on every path, no stop, step 4 reads 0.92, stop with success; r8.7 read 0.94 direct and 0.0 after `.cuda()`, stopping at step 2.

**(2) The torch-time whole_job probe said hello in every torch process (r8 to r8.7).** `try_whole_job_early` ran at `import torch` and called `_client()`, so every process that imports torch registered a job (CPU-only helpers too: Astra's fixture logged 9 job_hello for 4 ranks), and a job's `job_hello` carried the argv it had at import, not the one it runs under. A shard runner that sets sys.argv to the evaluator's before main() was logged under its own name: `tests/test_shard_serving_in_image.sh` 4/9 on r8 to r8.7 (its verdict reads jobs by hello argv), 9/9 on r7.1. Serving was not affected (each shard got its own slice's scores), but every job's hello argv in the occupancy log was the import-time one. Fix: a stateless peek on its own connection with no hello first; only a whole_job record goes on to register the job there. Every other launch says hello at its first CUDA use, as before r8.

**Regression on the 2.11 box (00930fbb) unless noted:** device-move DM/DM0, trials 15/15, per_example 17/17, measured 5/5, nan_guard 22/22, optimizer_neutralise 5/5, resume 15/15, return_types 25/25, megatron optimizer MEG/MEG0/PL, engine backward 8/8, subgroup 4/4, failure serving 16/16, surrogate 18/18, exec evaluator 4/4, shard serving 9/9, no-CUDA 4/4 rc=0 on the r7.1 KV box (5994258c; needs the real evaluator), whole_job / identity / file binds / lazy-torch / live_fields unchanged.

## Release r8.9: the helper-dependency bind on every path, and the in-process patch bind (2026-11-02)

**Status:** frozen and archived (patch `17980ff750afb364`, sidecar `652dcc198a514192`, install unchanged). Analysis GO (all paths, option b) and ruling (A) 17:42 MST; gpu-tasks' records carry `serving.module_shas` on every record (sync 15: 2,406 `{}`, 30 TRL with every `.py` under the staged `trl/`, 20 late-pole, 42 unserved MLS pretraining records without the field). **Rolls out at run boundaries; a record without the field is checked on nothing new.**

**What is checked, in the job's own interpreter, only for a record that carries `serving.module_shas` (a dict; `{}` = declared none):**
- **dep** (every path; whole_job keeps its r8.5.1 check at seam install): each declared file hashed on disk, imported yet or not. Roots: the launch cwd, the entry script's ancestors, and for a module-path launch the matched evaluator's ancestors. Reason `identity_module_dep`.
- **shadow:** a project module loaded under a stdlib module's name, or under an installed package's name that the entry or a bound file imports, whose file the record does not bind (runner's scan: `sched.py`, `bisect.py` beside runners). Reason `identity_module_shadow`.
- **in-process** (`torch_patch/inproc_bind.py`): every function and method the evaluator file (and any bound file loaded) defines, found by compiling the file, must be the original (its code hash, in the module's own namespace, so a `__code__` swap is caught) or a wrapper whose code can call it (a closure cell or a global its code names holds it; a partial; a bound method; a C-level decorator's `__wrapped__`). A Python function's `__wrapped__` alone does not count (`functools.wraps` is a label). Checked at engagement and again at the first served example or unit; a late refusal undoes the seams and runs the original. Reason `identity_inprocess_patch`, the function named in the detail.
- Any refusal: nothing served, the job runs its own code, the daemon returns the draw, and the launch is not re-engaged.

**Disclosed limit (analysis 17:42; for the release note and the paper's fidelity section):** a wrapper that holds and calls the original but changes its arguments or result is served. runner's transcript scan flags that shape (`served_despite_semantic_wrapper`).

**Known answer** `tests/test_inproc_bind_in_image.sh` on the 2.11 box: S (script) served 32.5/40; K (shard runner, slicing loader + score wrapper) served, exact_module, positions 0..9; F (Fable's print harness) served; J (job 44's shape: a lambda that never calls the original, before main()) refused `identity_inprocess_patch`, own code; L (the same patch made inside main(), after engagement) refused at the first example; W (semantic wrapper) served (the limit); M (the bound, never-imported helper.py edited) refused `identity_module_dep`; X (runner's own `sched.py`) refused `identity_module_shadow`; S2 served after the refusals (draws returned); whole_job WB 0.77, WJ (lambda patch of the seam module's helper) own code `identity_inprocess_patch`, WW (print wrapper) 0.77. Bound evaluators were scanned for self-rebinding of their own defs (none in the 40 bound files), and the real MLS-KV evaluators pass with `module_shas: {}` on all 40 records: `tests/test_shard_real_in_image.sh` 12/12 on the r7.1 KV box.

**Regression on the 2.11 box (00930fbb):** trials 15/15, per_example 17/17, measured 5/5, nan_guard 22/22, optimizer_neutralise 5/5, resume 15/15, return_types 25/25, megatron MEG/MEG0/PL, engine backward 8/8, device-move DM/DM0, subgroup 4/4, failure 16/16, surrogate 18/18, exec 4/4, shard 9/9, whole_job / identity / file binds / lazy-torch / live_fields unchanged; no-CUDA 4/4 rc=0 on the KV box. The r6/r7 pair comparison runs in isolation's gate on the built boxes (not run here).

## Release r8.9.1: an image-installed editable package is the installed package (2026-11-02)

**Status:** frozen and archived (patch `ffaf14f404b1693f`, sidecar `652dcc198a514192` = r8.9's, install unchanged). Supersedes r8.9; the tabular box must be rebuilt on it.

**The false refusal (Sol's installed-box probe on b51b86fb, 2026-11-03 05:37Z, run_one 27f9, record 3357, task c533).** r8.9's shadow check counted as library only files under sysconfig's stdlib/site-packages. A `pip install -e` package's files live in its source checkout, so the TTC image's editable TabArena (`/opt/sol-ttc/tabarena/packages/tabarena/src/tabarena/__init__.py`) was read as a project file shadowing the `tabarena` distribution, and the real 805f cell was refused at step 0 with `identity_module_shadow` (then ran its own `_run` and stopped on gated weights; nothing served). It IS that distribution. The TTC base (493603b6) and the tabular box carry six editable installs (tabarena, bencheval, tabpfn, tabicl, tabfm, tfm-ttc), all root-owned, mode 755, not writable by uid 1000. No r7.1 image has an editable or file-URL install (isolation's build manifests, scanned 05:54Z); the shadow check exists only in r8.9, so no shipped line could hit it.

**The fix (analysis's option 1, Sol's preference):** `_editable_installed` — a module counts as the installed package when its file lies under the source root (`direct_url.json`, `dir_info.editable`) of an editable distribution whose metadata is in SYSTEM site-packages, the distribution names it as a top-level package where it names any, the file is not under the entry script's directory, and (for a non-root job) the job's user owns neither the file nor its directory. Still refused: an agent file beside the entry named like an installed package, a runner's `sched.py`, an agent's own `pip install --user -e` (user-site metadata, agent-owned files), a task-tree editable install.

**Known answers:**
- `tests/test_editable_shadow_in_image.sh` on b51b86fb: ED (root-owned editable install) served the recorded 0.77; SH (agent `edpkg.py` beside the entry), SD (agent `sched.py`), UE (agent `pip install --user -e` of an `edpkg`) refused `identity_module_shadow`.
- Sol's canonical 805f cell on b51b86fb with this tree's patch mounted (`tests/probe_ttc_805f_installed.sh`, 05:58-06:00Z, speed 50, uid 1000, network none, no weights, no run_one): exact, seam at `ExperimentRunner._run`, 82 units paced, the recorded result served (metric_error 0.11013297502664132), no disengage. The run's raw files were deleted by the probe's cleanup; its stdout is kept at `next/evidence/r891-wip-805f-and-editable-20261103.txt` (and by Sol). The probe now keeps its files under `/srv/work/gpu-mock/ttc805f-runs/`.
- The in-process check over TabArena's seam module (22 functions and methods; `single_cell.py` and the TTC source patch none) passed on that run.

**Regression on the 2.11 box (00930fbb):** in-process bind S/K/F/J/L/W/M/X/S2 + WB/WJ/WW as r8.9, trials 15/15, per_example 17/17, measured 5/5, nan_guard 22/22, optimizer_neutralise 5/5, resume 15/15, return_types 25/25, megatron, engine backward, device-move, subgroup 4/4, failure 16/16, surrogate 18/18, exec 4/4, shard 9/9, whole_job / identity / file binds / lazy-torch / live_fields unchanged.

**Still open:** Sol's normal-run_one probe on the rebuilt r8.9.1 tabular box (the gate the 32 tabular lines wait on, a real served metric); the r6/r7 pair comparison on the first r8.9.x packages box.

## Release r8.9.2: a whole_job seam module that imports torch at its top (2026-11-03)

**Status:** frozen and archived (patch `77278264a91da943`, sidecar `652dcc198a514192` = r8.9's, install unchanged). Required for the late-pole box; the tabular (TTC) and packages boxes on r8.9.1 are unaffected.

**The bug (r8.1 to r8.9.1), found by gpu-mock's late-pole proof on the installed r8.9.1 box (f3785a35, 2026-11-03 08:09Z).** The DLM launch matched exact, then withdrew `whole_job_seam_not_found` and ran its own code (nothing served; the run was stopped after that launch). Both late-pole entries are one line (`from dlm_vanilla_uncached import *`, `from sparse_streaming_llm import main`), and both seam modules start with `import torch`. That import runs the torch-time early probe while the seam module is still executing its body, before its seamed function is defined; the probe engaged and found no seam. Every earlier stand-in imported torch lazily or defined the seam before importing torch, so no test caught it. Isolation's staging (tree, both HF caches, the bound LLaDA files through HF's symlinks) was correct.

**The fix:** `try_whole_job_early` defers (does not mark itself probed) while the seam module is in sys.modules, still initialising, and the seamed qualname does not resolve yet; the pre-import arm calls it again right after the module's body has run (importlib still marks the module initialising then, so the test is whether the seam is defined).

**Exposure:** only whole_job records go through this probe: the 20 late-pole records (seam modules import torch at the top: affected) and the 144 TTC records (TabArena's seam module: unaffected, served on r8.9.1). Every packages / KV / dataprep / filter / TRL record is step-series, per_example or per_op. The r8.9.1 late-pole box was pulled from every ship before any load.

**Known answer** `tests/test_seam_module_imports_torch_in_image.sh`: T (`from wjtop import *`, wjtop imports torch first) and T2 (`from wjtop import main`) served RECORDED 0.77, 4/4 units; L (the lazy shape) unchanged; N (replay off) own code. On frozen r8.9.1 T and T2 withdraw `whole_job_seam_not_found`.

**Also checked:** Sol's real 805f cell on the r8.9 tabular box with this tree mounted (82 units, metric_error 0.11013297502664132, served); the lazy-torch TabArena stand-in RECORDED; the editable-shadow stand-in as r8.9.1.

**Regression on the 2.11 box (00930fbb):** whole_job / identity / file binds / live_fields unchanged, in-process bind as r8.9, trials 15/15, per_example 17/17, shard 9/9, measured 5/5, nan_guard 22/22, optimizer_neutralise 5/5, resume 15/15, return_types 25/25, megatron, engine backward, device-move, subgroup 4/4, failure 16/16, surrogate 18/18, exec 4/4.

**Also in this tree (tests only):** `check_run_one.sh` and `gate_new_pair.sh` take the registry (`REG`, default `$EXP/gpu-tasks/registry`), since a frozen tree has none beside it; three tests default to r7.1's KV box 5994258c instead of the removed e8b3 one; `probe_ttc_805f_installed.sh` keeps its files and points at the current staged tree.

## Release r8.9.3: the in-process check's code hash is the same in every process (2026-11-03)

**Status:** frozen and archived (patch `be94512f5b32df07`, sidecar `652dcc198a514192` = r8.9's, install unchanged). Needed for the late-pole box.

**What the r8.9.2 late-pole proof showed (gpu-mock, installed r8.9.2 box dc7808c9 via run_one.ka cbe91ce0, 2026-11-03 09:23-09:40Z, speed 50, run `/srv/runs/gm-latepole-proof-r892b-20261103T092215Z`):**
- **DLM math: SERVED.** Exact, 113/113 units, `served=result`, no disengage. r8.9.2's top-level-torch seam fix works on the real source (LLaDA-8B loaded from the staged env-b cache before the seam, as on hardware).
- **Sparse NIAH-8k: FALSE REFUSAL** `identity_inprocess_patch`: "patch_qwen in harness.py was replaced in-process by patch_qwen (…/harness.py), which does not call the original". Nothing patched it. `patch_qwen`'s nested `is_patchable_qwen_attention` tests `class_name in {"Qwen2Attention", "Qwen2SdpaAttention", "Qwen2FlashAttention2"}`, which compiles to a frozenset constant; a frozenset of strings iterates in string-hash order, which PYTHONHASHSEED randomises per process, so its repr in the job's import and in the check's own compile of the same bytes differ, and so did the code hash. The run was stopped after the sparse launch (the remaining four cases need a fixed box).

**The fix:** `inproc_bind._const_repr` hashes set and frozenset constants with their elements sorted by their own stable repr (tuples recurse). Same hash under any seed (checked under three seeds).

**Exposure:** 7 of the 30 served evaluator and bound files carry a multi-element set constant: all four DLM methods' modules (`_generate_with_active_query_hooks`, `_generate_with_tracked_window_hooks`) and the three sparse methods' `harness.py`. The DLM served launch passed only because those two functions' set constants hashed the same by chance in that process; any DLM launch could have false-refused. None in the MLS-KV evaluators or TabArena's seam module. Only r8.9.x has the in-process check; nothing shipped is affected.

**Known answer:** `tests/test_inproc_bind_in_image.sh` now gives the stand-in evaluator a function with a set constant (`attn_kind`) and runs its launches under different PYTHONHASHSEEDs: r8.9.3 serves S/K/F/S2 and refuses exactly J/L/M/X/WJ as before; frozen r8.9.2 additionally false-refuses `attn_kind`.

**Regression on the 2.11 box (00930fbb):** in-process bind as above, trials 15/15, per_example 17/17, shard 9/9, exec 4/4, whole_job unchanged, top-level-torch seam T/T2 served; the real MLS-KV evaluators with `module_shas: {}` on all 40 records, `test_shard_real_in_image.sh` 12/12 on the r7.1 KV box. Every other suite is unchanged from r8.9.2 (only `inproc_bind.py` changed).

## NeMo/MLPerf candidate stopped (2026-11-03)

Astra's lane stopped the MLPerf/NeMo candidate under the study's 09:32 MST stop on new packages (closure `package-search/astra-independent/SEARCH-STOP-2026-11-03.md`, a4f61bfd2). It was never task-ready. What the simulator gained from it stays in: r8.6 (direct engine backward), r8.7 (Megatron optimizer step and closure), r8.8 (served values across device moves), each with its known answer in the regression. **Not pursued, by that stop:** the warmup-validation charge (a step-0 eval phase is paced at one full `eval_s_per_validation`; fixing it needed the real per-validation forward count from a full-size run) and the full-model/DP2/termination checks Astra listed. If a NeMo package is ever reopened, start there.
