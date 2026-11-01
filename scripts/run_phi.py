"""Train and evaluate the controlled Phi memory writers."""
import argparse
import contextlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))

def load(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['train','write','read'])
    p.add_argument('--target',choices=['relations','facts'],default='relations')
    p.add_argument('--cohort',choices=['controlled','fresh'],default='controlled')
    p.add_argument('--writer',type=Path,default=ROOT/'models/Phi-4-mini-instruct')
    p.add_argument('--reader',type=Path,default=ROOT/'models/Qwen3.5-4B')
    p.add_argument('--out',type=Path,default=ROOT/'results/phi')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    import torch
    from memory_model import MemoryModel,writer_prompt,BRIDGE_SYSTEM
    from sft_fact_state import SYSTEM,sft_examples
    from staged_memory_records import grouped_prefixes
    from staged_reader import read
    adapter=a.out/(a.target+'_adapter')
    if a.stage=='train':
        model=MemoryModel(str(a.writer),train=True,seed=17)
        rows=load(ROOT/'data/phi/fact_examples.jsonl') if a.target=='facts' else [e for r in load(ROOT/'data/phi/train_worlds.jsonl') for e in sft_examples(r,model.tokenizer)]
        assert len(rows)==120
        random.Random(17).shuffle(rows)
        params=[p for p in model.model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=5e-5)
        for i,r in enumerate(rows,1):
            prompt=model.prompt(SYSTEM,writer_prompt(r['question'],r['previous_memory'],r['evidence'],128))
            completion=r['target']+model.tokenizer.eos_token
            n=len(model.tokenizer.encode(completion,add_special_tokens=False))
            optimizer.zero_grad(set_to_none=True);model.model.train()
            loss=-model.completion_logprob(prompt,completion)/n
            loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
            optimizer.step();print(json.dumps({'step':i,'loss':float(loss.detach())}),flush=True)
        model.model.save_pretrained(adapter)
    elif a.stage=='write':
        writer=MemoryModel(str(a.writer),adapter=str(adapter),seed=17)
        rows=load(ROOT/'data/phi'/(a.cohort+'.jsonl'))
        out=[]
        for group in grouped_prefixes(rows).values():
            for policy in [a.target,'untrained','bridge','full']:
                mem=''
                if policy=='full':mem='\n\n'.join(group['evidence_chunks'])
                else:
                    for evidence in group['evidence_chunks']:
                        with writer.reader_context() if policy in ['untrained','bridge'] else contextlib.nullcontext():
                            mem=writer.update(group['question'],mem,evidence,128,system=BRIDGE_SYSTEM if policy=='bridge' else SYSTEM)
                out.append({'group_id':group['group_id'],'method':policy,'memory':mem})
        (a.out/(a.target+'_'+a.cohort+'_memories.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in out))
    else:
        reader=MemoryModel(str(a.reader),seed=17)
        memories=load(a.out/(a.target+'_'+a.cohort+'_memories.jsonl'))
        indexed={(r['group_id'],r['method']):r['memory'] for r in memories}
        out=[]
        for r in load(ROOT/'data/phi'/(a.cohort+'.jsonl')):
            for policy in [a.target,'untrained','bridge','full']:
                result=read(reader,r['question'],indexed[(r['group_id'],policy)],r['later_evidence'],binary=True)
                out.append({'id':r['id'],'group_id':r['group_id'],'method':policy,'correct':result['answer']==str(r['answer'])})
        (a.out/(a.target+'_'+a.cohort+'_scores.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in out))

if __name__=='__main__':main()
