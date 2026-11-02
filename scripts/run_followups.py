"""Run evidence-only and five-block experiments."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = {'prepare': ['build_noq_sft', 'build_ablation_data_v3'],
          'train-noq': ['train_noq_writer'], 'train-update': ['train_updater'],
          'train-append': ['train_append_updater'], 'future': ['run_delayed_nl3'],
          'recursive': ['run_recursive_fixed'], 'retention': ['run_recursive_retention'],
          'lengths': ['m5_lengths_rule']}

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=STAGES)
    p.add_argument('--preview', action='store_true')
    a = p.parse_args()
    env = os.environ.copy()
    defaults = {'DN_N':'200','DN_PREFIX':'12','DN_BUDGET':'128',
                'DN_OUT':str(ROOT/'sota/results/delayed_nl_noq.jsonl'),
                'RF_LIMIT':'40','RF_T':'5','RF_BUDGET':'128','RF_EQUAL_INPUT':'1',
                'RT_LIMIT':'40','RT_T':'5','RT_BUDGET':'128',
                'CF_PER_FAMILY':'15','CF_BUDGET':'128',
                'CF_MANIFEST':str(ROOT/'sota/data/simple_lookup_200.jsonl')}
    for k,v in defaults.items(): env.setdefault(k,v)
    for stage in STAGES[a.stage]:
        command=[sys.executable,str(ROOT/'sota/analysis'/(stage+'.py'))]
        if a.preview: print(' '.join(command))
        else: subprocess.run(command,cwd=ROOT,env=env,check=True)

    if a.stage == 'prepare' and not a.preview:
        for target in (ROOT/'sota/data/ablation_v3').glob('*.jsonl'):
            (ROOT/'sota/data'/('abl3_'+target.name)).write_bytes(target.read_bytes())
