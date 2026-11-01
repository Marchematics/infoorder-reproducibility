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
