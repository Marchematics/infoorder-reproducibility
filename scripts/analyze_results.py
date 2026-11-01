"""Summarize recorded scores and paired bootstrap intervals."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load(path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len({r['id'] for r in rows}) == len(rows)
    return {r['id']: r for r in rows}

def compare(left, right, metric='em'):
    assert set(left)==set(right)
    ids=sorted(left)
    delta=np.array([float(left[i][metric])-float(right[i][metric]) for i in ids])
    indices=np.random.default_rng(20261102).integers(0,len(ids),size=(10000,len(ids)))
    return {'difference':float(delta.mean()),'ci95':np.quantile(delta[indices].mean(axis=1),[.025,.975]).tolist()}

def analyze(folder):
    runs={p.stem:load(p) for p in sorted(folder.glob('*.jsonl'))}
    summaries={}
    for name, rows in runs.items():
        values=list(rows.values()); fields=['em','f1','answer_in_context','memory_tokens','full_token_count']
        fields += [k for k in values[0] if k.endswith(('_both_correct','_separated'))]
        summaries[name]={'n':len(values),**{k:statistics.mean(float(r[k]) for r in values) for k in fields if all(k in r for r in values)}}
    comparisons={}
    for tag,left,right in [
        ('chain_vs_summary','musique500_chain_qwen_b128','musique500_summary_b128_full'),
        ('chain_vs_prose','musique500_chain_qwen_b128','musique500_summary_aware_v2_b128'),
        ('2wiki_adaptation','2wiki_multidata_chain_b128','2wiki_v2_chain_b128'),
        ('2wiki_vs_summary','2wiki_multidata_chain_b128','2wiki_v2_summary_b128')]:
        if left in runs and right in runs:comparisons[tag]=compare(runs[left],runs[right])
    if 'simple_lookup_table_60' in runs:
        rows=runs['simple_lookup_table_60']
        comparisons['retained_bindings']=compare({i:{'em':r['chain_table_both_correct']} for i,r in rows.items()},
                                               {i:{'em':r['chain_both_correct']} for i,r in rows.items()})
    return {'runs':summaries,'paired_comparisons':comparisons}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores',type=Path,default=ROOT/'results/scores')
    p.add_argument('--out',type=Path,default=ROOT/'results/summary.json')
    a=p.parse_args();result=analyze(a.scores);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['paired_comparisons'],indent=2))
