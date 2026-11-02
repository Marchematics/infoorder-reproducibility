# InfoOrder

Reproduction code for **InfoOrder: Preserving Evidence for Future Reasoning**.
Python 3.11+; model runs use a CUDA GPU.

```bash
pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/download_model.py
python scripts/run_qa.py train
python scripts/run_qa.py write
python scripts/run_qa.py eval
```

For the mixed MuSiQue + 2Wiki writer, run `train`, `write`, and `eval` with
`--mixed`; select evaluation data with `--dataset 2wiki` or `--dataset hotpot`.
`--target facts|summary|prose|extractive` selects the alternative training targets.
`--seed`, `--budget`, and `--reader` select the remaining QA configurations.

```bash
python scripts/run_qa.py lookup
python scripts/run_theory.py
python scripts/analyze_results.py
python -m unittest discover -s tests
```

`run_phi.py` provides the controlled Phi training, writing, and reading stages.
Fixed inputs, targets, split IDs, and recorded per-question scores are included.
See [DATA.md](DATA.md) for dataset sources.

The training-target comparisons use `--target shuffled|no_relation|triples|answer|prose`.
Prepare the additional targets and run evidence-only or five-block evaluation with:

```bash
python scripts/run_followups.py prepare
python scripts/run_followups.py train-noq
python scripts/run_followups.py future
python scripts/run_followups.py train-update
python scripts/run_followups.py train-append
python scripts/run_followups.py recursive
python scripts/run_followups.py retention
```

`run_followups.py lengths` measures lookup memory lengths.
`analyze_followups.py` aggregates recorded scores; `score_retention.py` verifies coverage from saved memories on CPU.
The experiment settings are in `configs/followups.json`.

Matched summary targets use `run_qa.py --target reasoning` or `--target summary_matched`
for each training, writing, and evaluation stage. `analyze_review.py` aggregates the
additional recorded results on CPU. Review experiment settings are in `configs/followups.json`;
their runners are in `sota/analysis/`.
