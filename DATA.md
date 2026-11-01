# Data

The fixed QA inputs are derived from these datasets:

- [MuSiQue v1.0](https://github.com/stonybrooknlp/musique): 1,917 training and 500 evaluation questions.
- [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop): 3,000 training and 300 evaluation questions.
- [HotpotQA](https://hotpotqa.github.io/): 500 evaluation questions.

Upstream dataset licenses and attribution apply to the derived QA records.
`data/splits.json` fixes the IDs and order; `data/targets/` contains the training targets.
`data/phi/` and the lookup inputs contain the generated controlled examples.
`results/scores/` contains per-question measurements for the reported comparisons.

To regenerate annotation-based targets, obtain `musique_ans_v1.0_dev.jsonl`
from MuSiQue and pass it to `prepare_data.py --musique-source FILE`.
The corresponding preparation scripts are in `sota/analysis/`.
