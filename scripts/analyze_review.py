"""Aggregate the recorded future-query, memory, and system experiments on CPU."""
import json
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parents[1]

def load(name, folder='followups'):
    rows=[json.loads(s) for s in (ROOT/'results'/folder/(name+'.jsonl')).read_text().splitlines() if s.strip()]
    keys=[(r['id'],r.get('t'),r.get('cond')) for r in rows]
    assert len(keys)==len(set(keys)),name
    return rows

def means(rows,keys):
    return {k:st.mean(float(r[k]) for r in rows) for k in keys}

def analyze():
    report={}
    for name in ('delayed_nl_250','w2_scale'):
        rows=load(name)
        report[name]={'histories':len(rows),'queries':sum(r['n_selectors'] for r in rows),
                      **means(rows,[k for k in rows[0] if k not in ('id','n_selectors')])}
    for name in ('w3_deltalog','m4_cap_64','m4_cap_96','m4_cap_256'):
        rows=load(name); groups=[]
        for t in range(1,6):
            for cond in ('direct_cap','rec_chain','rec_append'):
                group=[r for r in rows if r['t']==t and r['cond']==cond]
                groups.append({'t':t,'cond':cond,**means(group,[k for k in group[0] if k not in ('id','t','cond')])})
        report[name]=groups
    hops=[h for r in load('w1b_faithfulness') for h in r['hops']]
    matched=[h for h in hops if h['gold_match']]
    report['alignment']={'hops':len(hops),'answer_in_selected':st.mean(h['grounded'] for h in hops),
                         'answer_in_any':st.mean(h['grounded_any_passage'] for h in hops),
                         'support_selected':st.mean(h['best_is_gold_support'] for h in matched),
                         'support_near_tie':st.mean(h['gold_support_near_tie'] for h in matched)}
    sm=sum([load('simplemem_100_'+s) for s in 'abcd'],[])
    assert len({r['id'] for r in sm})==len(sm)==100
    report['simplemem']=means(sm,['em','n_memories','elapsed'])
    epc=load('epc_adapted_60c')
    chain={r['id']:r for r in load('musique500_chain_qwen_b128','scores')}
    report['epc']={**means(epc,['em','tokens','calls']),
                   'paired_chain_em':st.mean(chain[r['id']]['em'] for r in epc)}
    for name in ('musique500_reasoning_summ_b128','musique500_summ_matched_b128',
                 'musique500_brief_style_b128','musique500_autodecomp_b128'):
        rows=load(name,'scores')
        report[name]={'n':len(rows),**means(rows,['em','answer_in_context','memory_tokens'])}
    return report

if __name__=='__main__':print(json.dumps(analyze(),indent=2))
