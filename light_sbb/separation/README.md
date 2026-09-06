# Separation tests: SB vs SBB on low-entropy targets

Implements the two experiments of `docs/sbb_note/sbb_note_corrected.pdf`. Both sweep a
concentration parameter `delta` at **fixed** `epsilon`, so the comparison is not rigged
by rescaling the noise, and both keep `beta * T > 1` as the note requires.

| | Test A | Test B |
|---|---|---|
| `mu_0` | `N(0, I_2)` | `N(0, 1)` |
| `mu_T^delta` | `N(0, diag(1, delta^2))` | `0.5 N(-2, delta^2) + 0.5 N(+2, delta^2)` |
| dimension | 2 | 1 |
| reference | exact Gaussian SBB solution | none in closed form |
| what it adds | reuses the closed form of §5.2 | optimal volatility is provably state-dependent |

As `delta -> 0` the target entropy diverges while `W_2` stays bounded, so the SB lower
bound of Proposition 1 grows like `2 eps log(1/delta)` while the SBB upper bound of
Proposition 3 stays flat. Both bounds are recorded in every result row, so the
prediction can be checked against what the two methods actually achieve.

## Running

From inside `light_sbb/`, one command per test:

```bash
python separation/run_separation.py --test A --device cuda:0
python separation/run_separation.py --test B --device cuda:0
```

Defaults reproduce the training configuration of `run_2d_benchmark.py` and
`run_heavy_tail.py`: 20 000 epochs, 50 potentials, batch 512, `lr=1e-3`, `eps=1`,
`S_init=0.1`, and `t_model=8`/`d_model=32` for the inverse net. Each run trains three
models per `delta` (LightSBB at `beta=10` and `beta=100`, plus the LightSB-M baseline)
over six `delta` values.

`beta=10` uses the moderate-beta algorithm with an inverse network, `beta=100` uses the
inverse-net-free large-beta algorithm; the switch is `--large-beta`. Both regimes run at
`K=5` by default. The published 2D run uses `K=15` below the threshold, which triples the
training time; `--K-moderate 15` reproduces it.

Every knob is overridable, for instance:

```bash
python separation/run_separation.py --test B --device cuda:1 \
    --deltas 0.5 0.1 0.02 --betas 10 100 --seeds 42 43 44 \
    --K 5 --n-epochs 20000 --batch-size 512 --lr 1e-3 --n-potentials 50
```

Useful flags: `--baseline-only` / `--no-baseline` to train one side alone,
`--n-eval` for the number of generated pairs scored, `--no-archive` to skip the tarball.

Splitting the sweep across devices is one column each, and the summary stays complete
whichever process finishes last:

```bash
python separation/run_separation.py --test A --betas 100 --no-baseline --device cuda:0
python separation/run_separation.py --test A --betas 10  --no-baseline --device cuda:1
python separation/run_separation.py --test A --baseline-only            --device cuda:2
```

## Output

Results land under `results/separation/test_<A|B>/<family>/delta_<d>/seed_<s>/`, each
holding `metrics.json`, `pairs.npy` (the generated `(X_0, X_1)`) and `weights.pt`. Those
per-run files are the source of truth. The summary
`results/separation/summary_test_<A|B>.json`, which is what the comparison figure should
read, is rebuilt at the end of every run by re-reading each `metrics.json` on disk, so
runs from separate invocations accumulate rather than overwrite each other.
`--summary-only` rebuilds it without training anything, which is the way to regenerate it
after unpacking results produced elsewhere. The whole folder is tarred to
`separation_test_<A|B>.tar.gz` at the repo root at the end of the run.

The folder name records `beta` but not which algorithm produced it, so the same `beta` run
under two different `--large-beta` thresholds overwrites itself.

## Metrics

Both tests report the distribution metrics of the heavy-tail table (`W_1`, energy
distance, KS) plus moment and quantile errors; `std_rel_err` is the decisive one, since
a Schrodinger bridge with frozen diffusion cannot contract below `sqrt(eps)` without an
unbounded drift cost and should saturate as `delta` shrinks. Test A adds the metrics of
the ground-truth experiment against the exact solution (plan `SW_2`, cross-covariance
error, objective gap, and the control errors `E_a`, `E_b`), computed per coordinate with
the `collapse_` prefix marking the axis being squeezed. Test B adds the mode statistics,
the achieved drift and volatility costs, and `sigma_spread`, which measures how far the
learned volatility varies across the state and should be strictly positive for LightSBB
and exactly zero for the baseline.

## Caveats from the note

`delta = 0` is never run: the target is singular there and the current LightSBB-M
implementation solves an inner SB problem that requires non-singular marginals. The
defensible claim is quantitative, that SB degrades like `log(1/delta)` while SBB stays
bounded, and not that the algorithm handles singular targets. `delta = 0.8` is kept as
a control point where the entropy penalty is mild, so a large discrepancy there would
indicate an implementation problem rather than the effect being tested.
