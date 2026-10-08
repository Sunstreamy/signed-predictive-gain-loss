# Signed Predictive-Gain Loss

Code and results for **Signed Predictive-Gain Loss: Directional Anomaly Evidence in Industrial Time Series**, Jiangyang He.

Dr monitors whether a fixed cross-channel correction loses its predictive benefit on targets where that benefit is repeatable under normal conditions. Matched readouts are S, Jr, Dr and PairError; external references are PCA-SPE and, on HAI, GCAD-P/GCAD-D with seeds 10/11/12.

## Reproducibility

- **Verified:** result-summary checks for all HAI/TE methods; frozen-parameter replay of S/Jr/Dr/PairError/PCA-SPE on all 58 HAI events, normal files and all 1,000 TE fault runs. Qualification, normal costs, event partitions, delays and TE family intervals match the supplied results.
- **GCAD:** sampled prediction/dependency-score and independent autograd checks cover all three seeds. Full-stream replay is not verified.
- **Fresh training:** commands are included, but end-to-end reproduction from newly fitted parameters is not verified.

Write rerun outputs to `outputs/`; `results/` contains the paper's result records.

## Setup

Tested on macOS arm64 with Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3 and OpenBLAS. GCAD additionally uses PyTorch 2.14.0 and scikit-learn 1.6.1. TE decoding uses base R (tested with R 4.6.0).

```bash
conda create --prefix .venv --file environment-osx-arm64.txt
conda activate ./.venv
python -m pip install -r requirements-gcad.txt  # optional: GCAD only
python scripts/check_manifest.py
python -m unittest discover -s tests -v
python scripts/check_results.py
```

All 29 tests pass. The Conda profile targets macOS arm64; `requirements-core.txt` gives numerical dependency pins for other platforms, which have not been tested. `check_results.py` checks the supplied result tables without running detectors.

## Data and parameters

Raw datasets and per-point outputs are not included in Git.

- **HAI 22.04:** [official data](https://github.com/icsdataset/hai), revision `2a814cebc9a66b06c9e5cd545e2d72e65d383737`. URLs and SHA256 for all ten CSVs are in [data_sources/hai.json](data_sources/hai.json).
- **TE 2017 v1.0:** [Harvard Dataverse](https://doi.org/10.7910/DVN/6C3JR1), files 3031241 (FaultFree Training), 3031240 (FaultFree Testing) and 3031243 (Faulty Testing). URLs and hashes are in [data_sources/te.json](data_sources/te.json). Faulty Training is unused.

Frozen parameters are in [Release v1.0.0](https://github.com/Sunstreamy/signed-predictive-gain-loss/releases/tag/v1.0.0). Download, verify and extract from the repository root:

```bash
release_url=https://github.com/Sunstreamy/signed-predictive-gain-loss/releases/download/v1.0.0
curl -fL -o hai-frozen-parameters-v1.zip "$release_url/hai-frozen-parameters-v1.zip"
curl -fL -o te-frozen-parameters-v1.zip "$release_url/te-frozen-parameters-v1.zip"
printf '%s\n' \
  '94262cdb0c3a351f8e2575d31b998774ad3115194349ce25710f4dfba3c1eb42  hai-frozen-parameters-v1.zip' \
  '4d2045ffbdffe5ea81e185b201b5da86d6d645b7fc05d2b16f1e6abd3e438a65  te-frozen-parameters-v1.zip' | shasum -a 256 -c -
unzip -q hai-frozen-parameters-v1.zip -d artifacts
unzip -q te-frozen-parameters-v1.zip -d artifacts
python scripts/check_sources.py parameters --raw-dir artifacts
```

The archives contain `hai/` and `te/` parameter directories, internal checksums and license notices. Individual file hashes are in [data_sources/model_artifacts.json](data_sources/model_artifacts.json).

Allow several GB for inputs and outputs. Decoding the full TE fault file requires substantially more memory than its selected 1,000-run panel (about 5.8 GiB peak R-vector allocation was measured).

## Frozen-parameter replay

### HAI

```bash
python scripts/check_sources.py hai --raw-dir data/hai/raw --download
python scripts/prepare_hai.py --raw-dir data/hai/raw --output data/hai/prepared
python scripts/fit_hai.py --cache data/hai/prepared --parameters artifacts/hai --output outputs/hai-normal
python scripts/score_hai.py --cache data/hai/prepared --parameters artifacts/hai --output outputs/hai
python scripts/evaluate_hai.py --cache data/hai/prepared --scores outputs/hai
python scripts/verify_scores.py hai --scores outputs/hai --cache data/hai/prepared --parameters artifacts/hai --window-checks --report outputs/hai/verification.json
```

With `--parameters`, `fit_hai` recomputes normal qualification using fixed weights. Default scoring covers the five simple readouts. Add `--with-gcad --device mps` to `score_hai` for all six GCAD seed readouts, using a separate output directory. The full reference event union requires all ten reference readouts.

### TE

```bash
python scripts/check_sources.py te --raw-dir data/te/raw --download
Rscript scripts/export_te_normal.R data/te/raw data/te/decoded
Rscript scripts/export_te_fault.R data/te/raw/TEP_Faulty_Testing.RData data_sources/te_selected_ids.txt data/te/external_replication_v1
python scripts/fit_te.py --data-dir data/te --parameters artifacts/te --output outputs/te
python scripts/score_te.py --data-dir data/te --normal outputs/te
python scripts/evaluate_te.py --scores outputs/te
python scripts/verify_scores.py te --scores outputs/te --window-checks --report outputs/te/verification.json
```

`fit_te` recomputes qualification, normal gains and calibration before fault scoring. `score_te` evaluates the fixed 20 × 50 panel. Verification reports retain mismatches and return a nonzero exit status on failure.

## Fresh training

After preparing the same inputs:

```bash
python scripts/fit_hai.py --cache data/hai/prepared --output outputs/hai-fitted
python scripts/train_gcad.py --cache data/hai/prepared --output outputs/gcad-fitted --device mps
python scripts/fit_te.py --data-dir data/te --output outputs/te-fitted
```

Use these fitted directories in the replay commands. For GCAD, place `gcad_s10`, `gcad_s11` and `gcad_s12` alongside the fitted HAI parameters before scoring. Training uses the fixed configurations; empty qualification stops matched detection. Newly fitted outputs may differ numerically from the released parameters.

## Evaluation and expected results

- **HAI:** train1/train3 fit, train4/train5 calibrate, train2/train6 describe previously inspected normal alarm behavior. Test4 has 24 development events; test1–3 have 34 frozen-candidate holdout events. Added reference comparisons are post-hoc. Q3 uses zero-based targets `[62,63,74]`; `q5` rows in `qualification.csv` contain opposite-file checks, and `q3` adds both pooled-fit-file checks.
- **TE:** fit/calibration/audit roles and the 50 family IDs are in [configs/te.json](configs/te.json). Forty of 52 targets qualify. Full horizon `[160,960)` is 40 h; early horizon `[160,240)` is 4 h. Family uncertainty uses 999 resamples with seed 20260930, conditional on fixed fitting and calibration.
- **Alarm rule:** Self sees only own lags `{1,4,16}`; correction uses `{0,1,4,16}`. W64 and three-point confirmation precede separate normal calibration of each score at α = 0.005/0.01/0.02. No point adjustment, test-optimal thresholds or best-seed/operating-point selection. Continuing pre-event alarms do not count as new detections. HAI common support starts at index 81; TE native/common costs are both retained.

Event counts at the primary nominal level α = 1%:

| Readout | HAI development / 24 | HAI holdout / 34 | TE full / 1,000 | TE early / 1,000 |
|---|---:|---:|---:|---:|
| S | 4 | 0 | 876 | 818 |
| Jr | 4 | 0 | 937 | 853 |
| Dr | 10 | 4 | 948 | 854 |
| PairError | 4 | 0 | 941 | 858 |
| PCA-SPE | 10 | 10 | 927 | 849 |

[results/hai/](results/hai/) and [results/te/](results/te/) include all seed results, event partitions, FPR, persistence, delays, normal gains and uncertainty intervals. The HAI reference union contains S, Jr, PairError, PCA-SPE and all six GCAD readouts; it is an event-set summary, not a fused detector.

HAI includes a negative test1 and test2 false-alarm migration. TE has substantial normal pseudo-event costs despite low pointwise FPR. HAI is a physical/HIL testbed; TE is a simulation. These results do not establish overall detection superiority, causal attribution or universal cross-run FPR control.

Alternative summation changed three HAI and eighteen TE calibration-boundary decisions without changing the reported normal-audit, attack or fault decisions. Numerical discrepancy records are in `results/hai/numerical_audit.json` and `results/te/normal_verification.json`.

## Licenses and citation

Own code is MIT. HAI fitted parameters are CC BY-SA 4.0; TE fitted parameters are MIT with dataset terms retained. GCAD and RevIN are MIT. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the notices in each archive.

Please cite the companion article when published, HAI's official dataset reference, Rieth et al. (2017), *Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation*, DOI 10.7910/DVN/6C3JR1, and Liu et al., *Granger Causality-based Anomaly Detection for Multivariate Time Series* when using GCAD.
