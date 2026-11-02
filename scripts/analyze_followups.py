"""Aggregate future-query, recursion, retention, and length measurements."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]

def load(name):
    return [json.loads(l) for l in (ROOT/'results/followups'/name).read_text().splitlines()]

def analyze():
    queries=load('delayed_nl_noq.jsonl')
    future={k:statistics.mean(r[k] for r in queries) for k in queries[0] if k not in ('id','n_selectors')}
    rounds=[]
    for name in ('recursive_fixed2.jsonl','recursive_retention.jsonl'):
        values=load(name)
        for t in range(1,6):
            for cond in ('direct_cap','rec_chain','rec_append'):
                rr=[r for r in values if r['t']==t and r['cond']==cond]
                rounds.append({'source':name,'t':t,'cond':cond,'n':len(rr),
                    **{k:statistics.mean(r[k] for r in rr) for k in rr[0] if k not in ('id','t','cond')}})
    lengths=load('m5_lengths_rule.jsonl')
    return {'future_queries':{'histories':len(queries),'queries':sum(r['n_selectors'] for r in queries),'means':future},
            'rounds':rounds,'length_run':{k:statistics.mean(r[k] for r in lengths) for k in lengths[0] if k.endswith('_tokens')}}

if __name__=='__main__': print(json.dumps(analyze(),indent=2))
