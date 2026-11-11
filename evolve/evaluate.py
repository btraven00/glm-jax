"""Score one candidate glm_jax/<target>.py: static ban, correctness gate, pinned timing.

    [EVOLVE_TARGET=dispersion|irls] pixi run -e test python evolve/evaluate.py CANDIDATE.py [--no-tests]
    -> one JSON line

1. Regex ban on monkeypatching and I/O (``irls.x = ...`` for any *other* module, setattr,
   importlib, os., subprocess, open, sys.modules) - the brief says only this file may change.
2. The target's tests (TARGETS[t]["pytest"]) in a scratch copy of the package with the
   candidate swapped in.
3. On each timing workload (WORKLOADS: synthetic 5000 genes x 12 samples, and the first 500
   genes of the Hao 10k-cell slice) the candidate must match a reference computed once from
   the repo code (evolve/reference_<target>_<workload>.npz): dispersion -> apl_grid on a
   20-point grid vs steps=None (finite, every |dAPL| < 5% of that gene's APL gap to the nearer
   neighbouring grid point + 0.01, identical per-gene argmax);
   irls -> fit_glm with phi=0.1 vs max_iter=200 (every gene the reference converged is
   converged, max|dbeta| < 1e-6 on those).
Fitness: geometric mean over workloads of the median of 5 timed runs, f64, one core
(EVOLVE_CORE, default 0), minimal environment (no credentials). Timeouts and crashes are
reported, never raised.
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TARGET = os.environ.get("EVOLVE_TARGET", "dispersion")
HAO = os.environ.get("EVOLVE_HAO", os.path.join(REPO, "scratch", "data", "hao_10000.npz"))
# EVOLVE_REMOTE=<ssh host with sbatch>: timing scripts run there as one GPU SLURM job each
# (tests stay local); ~ is shared with the GPU node.  Empty = local pinned core.
REMOTE = os.environ.get("EVOLVE_REMOTE", "")
REMOTE_DIR = os.environ.get("EVOLVE_REMOTE_DIR", "glm-jax-evolve")  # relative to remote ~
REMOTE_PY = os.environ.get("EVOLVE_REMOTE_PY", "glm-jax/.venv-gpu/bin/python")
# EVOLVE_REMOTE_SBATCH=0: the remote host has the GPU itself, run scripts over plain ssh
SBATCH = os.environ.get("EVOLVE_REMOTE_SBATCH", "1") == "1"
DEVICE = os.environ.get("EVOLVE_DEVICE", "one GPU (NVIDIA L4)" if REMOTE else "one CPU core")
SC = os.environ.get("EVOLVE_SC", "")  # npz (Y, X, offset) for the pbmc100k_* workloads, on the timing host


def _rp(p):  # remote paths: absolute as given, else relative to the remote ~
    return p if p.startswith("/") else f"~/{p}"

HEAD = """
import sys, time, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import glm_jax as gj
grid = np.logspace(-3, 0.5, 20)
"""
# name -> (code defining Y, X, offset; one-line description for the brief)
ALL_WORKLOADS = {
    "synth": ("""
G, n = 5000, 12
rng = np.random.default_rng(0)
X = np.column_stack([np.ones(n), np.repeat([0, 1], n // 2), rng.normal(size=n)])
offset = np.log(rng.uniform(0.5, 2.0, n) * 1e6)
beta = np.column_stack([rng.normal(-11, 1, G), rng.normal(0, 1, G), rng.normal(0, 0.3, G)])
mu = np.exp(beta @ X.T + offset)
Y = rng.negative_binomial(5.0, 5.0 / (5.0 + mu)).astype(float)
""", "synthetic bulk, 5000 genes x 12 samples (3 coefficients)"),
    # ponytail: first 500 of 2000 genes keeps one evaluation ~1 min; take more if timing gets noisy
    "hao": (f"""
d = np.load({HAO!r})
Y, X, offset = d["Y"][:500], d["X"], d["offset"]
""", "real single-cell, 500 genes x 10000 cells (10 coefficients, mostly low counts)"),
    # GPU-shaped: same generator as examples/sc_bench.py and all 2000 Hao genes
    "synth_sc": ("""
G, n, PHI = 2000, 10000, 0.1
rng = np.random.default_rng(0)
cond = rng.integers(0, 2, n); ctype = rng.integers(0, 4, n)
X = np.column_stack([np.ones(n), cond] + [ctype == k for k in (1, 2, 3)]).astype(float)
offset = np.log(rng.lognormal(np.log(5000), 0.5, n))
beta = np.column_stack([rng.normal(-8, 1.5, G), rng.normal(0, 0.5, (G, 4))])
mu = np.exp(beta @ X.T + offset)
Y = rng.negative_binomial(1 / PHI, 1 / (1 + PHI * mu)).astype(float)
""", "synthetic single-cell, 2000 genes x 10000 cells (5 coefficients)"),
    "hao_full": (f"""
d = np.load({HAO!r})
Y, X, offset = d["Y"], d["X"], d["offset"]
""", "real single-cell, 2000 genes x 10000 cells (10 coefficients, mostly low counts)"),
    # dispersion_sc: DT is the compute dtype, TOL the max f64 APL a gene may lose vs the reference
    "pbmc100k_f64": (f"""
d = np.load({SC!r})
Y, X, offset = d["Y"].astype(np.float64), d["X"], d["offset"]
DT, TOL = np.float64, 0.05
""", "real single-cell (Hao PBMC), 256 genes x 100000 cells (10 coefficients), float64"),
    "pbmc100k_f32": (f"""
d = np.load({SC!r})
Y, X, offset = d["Y"].astype(np.float64), d["X"], d["offset"]
DT, TOL = np.float32, 0.1
""", "the same data in float32"),
}
WORKLOADS = {w: ALL_WORKLOADS[w][0] for w in os.environ.get("EVOLVE_WORKLOADS", "synth,hao").split(",")}
FITNESS = (f"Fitness is the geometric mean of the median wall times on {len(WORKLOADS)} workload(s), "
           f"{DEVICE}, f64: " + "; ".join(f"({chr(97 + i)}) {ALL_WORKLOADS[w][1]}" for i, w in enumerate(WORKLOADS))
           + ".")
HINTS = os.environ.get("EVOLVE_HINTS", "")  # free text appended to the brief (seed ideas / findings)
TAIL = """Propose ONE idea per attempt.  Replace the
previous idea comment with yours: exactly one line starting with "# Idea:" at
the top of the file.  Return the complete new file in a single ```python block
and nothing else.  Prefer changes that reduce total floating-point work or
memory traffic; micro-rewrites that XLA already fuses are wasted attempts."""
BRIEF_DISPERSION = f"""You optimise one file of a small JAX library for speed on {DEVICE}.

The file is glm_jax/dispersion.py: negative-binomial dispersion estimation by
Cox-Reid adjusted profile likelihood on a grid, plus a Newton polish.  It may
import these private helpers from glm_jax.irls, whose code you cannot change:
_fit_single(y, X, offset, w, phi, beta0, use_beta0, *, family, max_iter, tol,
beta_tol, ridge, max_halving) -> GLMResult(coef, deviance, mu, cov_unscaled,
n_iter, converged); _working(family, y, eta, w, phi) -> (W, z, mu);
_wls(X, W, z, ridge); _small_chol(A) -> lower-triangular L as nested lists;
_broadcast_inputs(Y, X, offset, weights, phi, beta0) -> (Y, X, offset, weights,
phi, beta0, use_beta0); _xtwx(X, W, ridge) -> X^T diag(W) X + ridge I; _SMALL_P; fit_glm.

Hard constraints (an automatic evaluator enforces them and discards violators):
- keep the public functions and signatures: adjusted_profile_loglik, apl_grid,
  estimate_dispersion_grid, with the same return shapes and dict keys;
- on both timed workloads (20 dispersions each) apl_grid must give the same
  per-gene argmax as the fully converged APL, and at every grid point stay within
  5% of that gene's APL difference to the nearer neighbouring grid point (+ 0.01);
  estimate_dispersion_grid's Newton polish must still beat the coarse grid; the
  jitted apl_grid must lower to StableHLO with no custom_call (so no
  jnp.linalg.* on batched matrices);
- float64 numerics, pure JAX, no new dependencies, no file/OS access, no
  monkeypatching of other modules (statically rejected).

{FITNESS}  The timed call is apl_grid with 20 dispersions.  {TAIL}

{HINTS}"""
BRIEF_IRLS = f"""You optimise one file of a small JAX library for speed on {DEVICE}.

The file is glm_jax/irls.py: batched IRLS (Fisher scoring with step halving)
for one GLM per gene with a shared design, written as a single-gene solver
(_fit_single, lax.while_loop) vmapped over genes and jitted.  Under vmap every
gene waits for the slowest, so _fit_batched runs two passes: _FIRST_PASS_ITER
(12) iterations for all genes, then the remaining iterations for a fixed-size
bucket max(_STRAGGLER_BUCKET, G/8) of the least-converged genes (static shapes,
one program).  On one CPU core fit_glm also chunks genes (_CPU_CHUNK_ELEMS)
with lax.map so per-iteration intermediates stay in cache.

Contract (other modules import these; names and signatures must stay, and
behaviour must stay correct):
- fit_glm(Y, X, *, family=NegativeBinomial(), offset=None, weights=None,
  phi=None, beta0=None, max_iter=50, tol=1e-8, beta_tol=1e-8, ridge=1e-10,
  max_halving=30, batch_size=None) -> GLMResult; 1-D Y is one feature;
- GLMResult(coef (...,p), deviance (...), mu (...,n), cov_unscaled (...,p,p)
  = (X^T W X)^-1 at convergence, n_iter (...), converged (...) bool);
- _fit_single(y, X, offset, w, phi, beta0, use_beta0, *, family, max_iter,
  beta_tol, tol, ridge, max_halving) -> GLMResult for one gene (vmapped and
  differentiated through by glm_jax/dispersion.py);
- _working(family, y, eta, w, phi) -> (W, z, mu);
- _wls(X, W, z, ridge) -> beta;
- _small_chol(A) -> lower-triangular L as nested lists of scalars;
- _broadcast_inputs(Y, X, offset, weights, phi, beta0) -> (Y, X, offset,
  weights, phi, beta0, use_beta0);
- _SMALL_P (int).

Hard constraints (an automatic evaluator enforces them and discards violators):
- tests/test_glm.py must pass (f32 and f64 inputs, families, GPU-free);
- on both timed workloads, every gene the reference fit (max_iter=200)
  converged must come back converged, with max|delta beta| < 1e-6;
- float64 numerics, pure JAX, no new dependencies, no file/OS access, no
  monkeypatching of other modules (statically rejected).

{FITNESS}  The timed call is fit_glm(Y, X, offset=offset, phi=0.1 per gene)
with default tolerances.  {TAIL}

{HINTS}"""
BRIEF_DISPERSION_SC = f"""You optimise one file of a small JAX library for speed on {DEVICE}.

The file is glm_jax/dispersion.py: negative-binomial dispersion estimation by
Cox-Reid adjusted profile likelihood (APL).  estimate_dispersion_grid walks a
30-point grid of phi (the first point fit to convergence, the rest with 2 fixed
warm-started Fisher steps each, quasi-Newton first step reusing the previous
point's X^T W X), then takes 2 safeguarded Newton steps in log(phi) per gene
from the grid argmax (forward-over-forward derivatives through the fixed steps).
At 100k cells the grid is now most of the time; the polish is the rest.

It imports from glm_jax.irls, whose code you cannot change: _SMALL_P,
_broadcast_inputs, _exact_f32 (context manager forcing exact f32 matmuls),
_fit_single, _spd_solve, _wls, _working, _xtwx (X^T diag(W) X for all genes as
one (genes x cells) @ (cells x p(p+1)/2) GEMM), fit_glm.

Hard constraints (an automatic evaluator enforces them and discards violators):
- keep the public functions and signatures: adjusted_profile_loglik, apl_grid,
  estimate_dispersion_grid (including polish_common), PHI_GRID, with the same
  return shapes and dict keys; tests/test_glm.py -k "apl or dispersion" pass;
- accuracy is judged by an independent frozen copy of the current file: at your
  tagwise estimates, every gene's f64 APL may be at most 0.05 (float64 run) /
  0.1 (float32 run) below the APL at a reference estimate (60-point grid,
  4 Newton steps).  Changing the grid, the number of steps or the polish is
  allowed if it passes this;
- the compiled program's temporary memory must stay within 1.1x of the current
  file's (memory per gene x cell sets how many genes fit per device chunk at
  1M+ cells, so a faster but larger program is slower at scale);
- pure JAX, no new dependencies, no file/OS access, no monkeypatching of other
  modules or of jax/jnp/lax/np attributes (statically rejected).

{FITNESS}  The timed call is the jitted
estimate_dispersion_grid(Y, X, offset=offset, polish_common=False)["tagwise"].
{TAIL}

{HINTS}"""
# per target: f() is the timed call; REF saves the reference to sys.argv[1]; GATE sets bad (None = pass)
TARGETS = {
    "dispersion": {
        "brief": BRIEF_DISPERSION,
        "pytest": ["tests/test_glm.py", "-k", "apl or dispersion"],
        "run": "f = lambda: np.asarray(gj.apl_grid(Y, X, grid, offset=offset))  # eager: apl_grid jits internally\n",
        "ref": """
apl = np.asarray(gj.apl_grid(Y, X, grid, offset=offset, steps=None))
np.savez(sys.argv[1], apl=apl, argmax=apl.argmax(1))
""",
        "gate": """
apl, r = f(), ref["apl"]
same = apl.shape == r.shape
# Scale each error by the gene's own APL gap to the nearer grid neighbour: an absolute
# tolerance means nothing across genes whose APL spans 1e1..1e5 (Hao gene 1054: 19.6 off
# with steps=2, yet 0.2% of a ~1e4 gap). 5% of the gap cannot move the argmax or the
# Newton start; + 0.01 covers genes whose APL is flat in phi (all-zero rows).
d = np.abs(np.diff(r, axis=1))
gap = np.fmin(np.pad(d, ((0, 0), (1, 0)), constant_values=np.nan), np.pad(d, ((0, 0), (0, 1)), constant_values=np.nan))
rel = float(np.max(np.abs(apl - r) / (0.05 * gap + 0.01))) if same else np.inf
err = float(np.max(np.abs(apl - r))) if same else np.inf
agree = float(np.mean(apl.argmax(1) == ref["argmax"])) if same else 0.0
ok = np.all(np.isfinite(apl)) and rel < 1.0 and agree == 1.0
bad = None if ok else f"shape={apl.shape} max_rel_to_gap={rel:.3g} max_abs_diff={err:.3g} argmax_agree={agree:.4f}"
""",
    },
    "dispersion_sc": {
        "file": "dispersion",
        "brief": BRIEF_DISPERSION_SC,
        "pytest": ["tests/test_glm.py", "-k", "apl or dispersion"],
        "run": """
Yd, Xd, od = (jnp.asarray(a, DT) for a in (Y, X, offset))
prog = jax.jit(lambda Y: gj.estimate_dispersion_grid(Y, Xd, offset=od, polish_common=False)["tagwise"]).lower(Yd).compile()
mem = prog.memory_analysis().temp_size_in_bytes
f = lambda: jax.block_until_ready(prog(Yd))
""",
        # reference from the repo code at tighter settings; mem is the current file's program
        "ref": """
Y64, X64, o64 = (jnp.asarray(a, jnp.float64) for a in (Y, X, offset))
tag = gj.estimate_dispersion_grid(Y64, X64, offset=o64, phi_grid=np.logspace(-4, 1, 60), newton=4, polish_common=False)["tagwise"]
np.savez(sys.argv[1], tag=np.asarray(tag), apl=np.asarray(gj.adjusted_profile_loglik(Y64, X64, tag, offset=o64)), mem=mem)
""",
        "gate": """
from glm_jax._trusted_dispersion import adjusted_profile_loglik as trusted_apl  # repo copy, not the candidate
tag = np.asarray(f())
same = tag.shape == ref["tag"].shape and bool(np.all(np.isfinite(tag)))
loss = ref["apl"] - np.asarray(trusted_apl(*(jnp.asarray(a, jnp.float64) for a in (Y, X)), jnp.asarray(tag, jnp.float64),
                                           offset=jnp.asarray(offset, jnp.float64))) if same else np.array([np.inf])
ok = same and bool(np.all(np.isfinite(loss))) and loss.max() < TOL and mem <= 1.1 * ref["mem"]
bad = None if ok else f"shape={tag.shape} max_apl_loss={loss.max():.3g} (tol {TOL}) temp_memory={mem / ref['mem']:.2f}x current"
""",
    },
    "irls": {
        "brief": BRIEF_IRLS,
        "pytest": ["tests/test_glm.py"],
        "run": "phi = np.full(Y.shape[0], 0.1)\nf = lambda: jax.block_until_ready(gj.fit_glm(Y, X, offset=offset, phi=phi))\n",
        "ref": """
r = gj.fit_glm(Y, X, offset=offset, phi=np.full(Y.shape[0], 0.1), max_iter=200)
np.savez(sys.argv[1], coef=np.asarray(r.coef), converged=np.asarray(r.converged))
""",
        "gate": """
r = f(); coef, conv = np.asarray(r.coef), np.asarray(r.converged)
both = conv & ref["converged"]
missed = int(np.sum(ref["converged"] & ~conv))
err = float(np.max(np.abs(coef[both] - ref["coef"][both]), initial=0.0)) if coef.shape == ref["coef"].shape else np.inf
ok = missed == 0 and conv.mean() >= ref["converged"].mean() and err < 1e-6
bad = None if ok else (f"converged={conv.mean():.4f} ref={ref['converged'].mean():.4f} "
                       f"missed={missed} max_abs_dbeta={err:.3g}")
""",
    },
}
T = TARGETS[TARGET]
FILE = T.get("file", TARGET)
OTHERS = "|".join(m for m in ("irls", "dispersion", "families", "inference", "glm_jax", "jax", "jnp", "lax", "np")
                  if m != FILE)
BANNED = re.compile(rf"\b({OTHERS})(\.\w+)+\s*=[^=]|\b_trusted|\bsetattr\(|\bimportlib\b|\bos\.(?!sched_getaffinity\(|cpu_count\()|"  # irls.py reads its core count
                    r"\bsubprocess\b|\bopen\(|\b__import__\b|\bsys\.modules\b")
TIMER = """
ref = np.load(sys.argv[1])
{gate}
if bad:
    print("REJECT " + bad); sys.exit(3)
ts = []
for _ in range(5):
    t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
print(sorted(ts)[2])
"""


def ref_path(w):
    # remote references live in ~/REMOTE_DIR on the cluster (different device, different timings)
    return (f"{REMOTE_DIR}/" if REMOTE else os.path.join(HERE, "")) + f"reference_{TARGET}_{w}.npz"


def ref_exists(w):
    if REMOTE:
        return subprocess.run(["ssh", "-o", "BatchMode=yes", REMOTE, f"test -e {ref_path(w)}"]).returncode == 0
    return os.path.exists(ref_path(w))


def script(w, body):
    return HEAD + WORKLOADS[w] + T["run"] + body


def _run_remote(code, ref, cwd, timeout):
    """Run `code` (with sys.argv[1]=ref) on the GPU node as one SLURM job; cwd/glm_jax goes along."""
    job = f"{REMOTE_DIR}/job-{os.getpid()}-{abs(hash(code)) % 10**8}"
    tmp = tempfile.mkdtemp(prefix="glmjax-remote-")  # never write into cwd: for references cwd is the repo
    open(os.path.join(tmp, "script.py"), "w").write(code)
    sh = ["ssh", "-o", "BatchMode=yes", REMOTE]
    try:
        subprocess.run(sh + [f"mkdir -p {job}"], check=True, capture_output=True, timeout=120)
        subprocess.run(["rsync", "-az", "--exclude", "__pycache__", os.path.join(cwd, "glm_jax"),
                        os.path.join(tmp, "script.py"), f"{REMOTE}:{job}/"], check=True, capture_output=True, timeout=300)
        # ponytail: one sbatch --wait per script; queue wait is small while the GPU is ours
        run = f"PYTHONPATH=. JAX_PLATFORMS=cuda {_rp(REMOTE_PY)} script.py {_rp(ref)}"
        if SBATCH:
            run = f"sbatch --wait -J glmjax-evolve --gres=gpu:1 -c 4 --mem=32G -t 00:30:00 -o out.txt --wrap='{run}' >/dev/null"
        else:
            run += " > out.txt 2>&1"
        cmd = f"cd {job} && {run}; rc=$?; cat out.txt; cd ~ && rm -rf {job}; exit $rc"
        r = subprocess.run(sh + [cmd], capture_output=True, text=True, timeout=timeout)
        return subprocess.CompletedProcess(code, r.returncode, r.stdout, r.stderr)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(code, 124, "", f"remote timeout after {timeout} s")
    except (OSError, subprocess.CalledProcessError) as e:
        return subprocess.CompletedProcess(code, 125, "", f"{e}: {getattr(e, 'stderr', b'')!r}"[-2000:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run(args, cwd, pythonpath, timeout=900):
    if REMOTE and args[1] == "-c":  # timing/reference scripts; pytest stays on the local core
        return _run_remote(args[2], args[3], cwd, max(timeout, 2400))
    # EVOLVE_LOCAL_GPU=1 (on the GPU host itself, e.g. evolve/batch.py evaluate): timing scripts use the GPU
    gpu = os.environ.get("EVOLVE_LOCAL_GPU") == "1" and args[1] == "-c"
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""), "PYTHONPATH": pythonpath,
           "JAX_PLATFORMS": "cuda" if gpu else "cpu", "OMP_NUM_THREADS": "1",
           "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false"}
    pin = ["taskset", "-c", os.environ.get("EVOLVE_CORE", "0")]
    try:
        return subprocess.run(pin + args, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 124, "", f"timeout after {timeout} s")
    except OSError as e:
        return subprocess.CompletedProcess(args, 125, "", str(e))


def ensure_reference():
    have = subprocess.run(["ssh", "-o", "BatchMode=yes", REMOTE, f"test -e {HAO}"]).returncode == 0 if REMOTE \
        else os.path.exists(HAO)
    if not have and any(w.startswith("hao") for w in WORKLOADS):
        raise FileNotFoundError(f"real-data workload needs {HAO} on {REMOTE or 'this machine'} (Y, X, offset; examples/real_data.py)")
    for w in WORKLOADS:
        if not ref_exists(w):
            r = _run([sys.executable, "-c", script(w, T["ref"]), ref_path(w)], REPO, REPO, timeout=3600)
            if r.returncode != 0:
                raise RuntimeError(f"reference build failed ({TARGET}/{w}): " + r.stderr[-2000:])


def evaluate(candidate, tests=True):
    m = BANNED.search(open(candidate).read())
    if m:
        return {"ok": False, "time": None, "times": {}, "error": f"banned construct: {m.group(0)!r}"}
    ensure_reference()
    work = tempfile.mkdtemp(prefix="glmjax-eval-")
    try:
        for d in ("glm_jax", "tests"):
            shutil.copytree(os.path.join(REPO, d), os.path.join(work, d),
                            ignore=shutil.ignore_patterns("__pycache__"))
        # frozen repo copy for gates that must not trust the candidate's own code
        shutil.copy(os.path.join(REPO, "glm_jax", "dispersion.py"), os.path.join(work, "glm_jax", "_trusted_dispersion.py"))
        shutil.copy(candidate, os.path.join(work, "glm_jax", f"{FILE}.py"))
        if tests:
            t = _run([sys.executable, "-m", "pytest", "-q", "-x"] + T["pytest"], work, work)
            if t.returncode != 0:
                return {"ok": False, "time": None, "times": {}, "error": (t.stdout + t.stderr)[-3000:]}
        times = {}
        for w in WORKLOADS:
            r = _run([sys.executable, "-c", script(w, TIMER.replace("{gate}", T["gate"])), ref_path(w)], work, work)
            if r.returncode != 0:
                return {"ok": False, "time": None, "times": times, "error": f"[{w}] " + (r.stdout + r.stderr)[-3000:]}
            times[w] = float(r.stdout.strip().splitlines()[-1])
        return {"ok": True, "time": math.exp(sum(map(math.log, times.values())) / len(times)), "times": times,
                "error": ""}
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    print(json.dumps(evaluate(sys.argv[1], tests="--no-tests" not in sys.argv)))
