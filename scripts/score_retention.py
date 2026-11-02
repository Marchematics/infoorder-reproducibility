"""Recompute reader-free coverage from the saved memory text on CPU."""
import gzip
import json
from pathlib import Path
import re
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sota/harness'))
from scoring import answer_in_context
STOP={'Which','Who','What','When','Where','How','Many','The','In','Of','And','A','An'}

def entity_of(question):
    candidates=re.findall(r'[A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)*',question or '')
    candidates=[c for c in candidates if c.split()[0] not in STOP]
    return max(candidates,key=len) if candidates else None

def score():
    text=gzip.decompress((ROOT/'data/inputs/manifest_500.jsonl.gz').read_bytes())
    manifest={r['id']:r for l in text.splitlines() for r in [json.loads(l)]}
    output=[]
    for line in (ROOT/'results/followups/recursive_retention_memories.jsonl').read_text().splitlines():
        r=json.loads(line);src=manifest[r['id']];memory=r['memory']
        hops=[h for h in src.get('question_decomposition',[]) if str(h.get('answer','')).strip()]
        covered=[];bound=[]
        for h in hops:
            hit=bool(answer_in_context(h.get('answer'),memory,src.get('answer_aliases')))
            covered.append(float(hit));entity=entity_of(h.get('question'))
            if entity: bound.append(float(hit and entity.lower() in memory.lower()))
        output.append({'id':r['id'],'t':r['t'],'cond':r['cond'],
                       'hop_coverage':statistics.mean(covered) if covered else 0.0,
                       'binding_coverage':statistics.mean(bound) if bound else 0.0,
                       'final_aic':float(bool(answer_in_context(src.get('answer'),memory,src.get('answer_aliases'))))})
    return output

if __name__=='__main__':
    expected={(r['id'],r['t'],r['cond']):r for l in (ROOT/'results/followups/recursive_retention.jsonl').read_text().splitlines() for r in [json.loads(l)]}
    scored=score()
    for r in scored:
        e=expected[r['id'],r['t'],r['cond']]
        for k in ('hop_coverage','binding_coverage','final_aic'): assert abs(r[k]-e[k])<1e-12
    print('Verified',len(scored),'memory records against recorded coverage scores')
