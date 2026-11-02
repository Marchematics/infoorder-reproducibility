"""Train, write, or evaluate a fixed QA configuration."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = {'musique': 'manifest_500', '2wiki': 'manifest_2wiki_fixed_300', 'hotpot': 'manifest_hotpotqa_500'}
TARGETS = {'chain': 'chain_sft_train', 'facts': 'facts_format_sft_train',
           'summary': 'summary_sft_train', 'prose': 'answer_aware_summary_sft_train',
           'extractive': 'extractive_sft_train', 'shuffled': 'abl3_chain_shuffled',
           'no_relation': 'abl3_chain_no_relation', 'triples': 'abl3_chain_triples', 'answer': 'abl3_direct_answer',
           'reasoning': 'reasoning_summary_sft_train', 'summary_matched': 'summary_matched_sft_train'}
ADAPTERS = {'chain': 'chain_writer_qwen_adapter', 'facts': 'facts_format_adapter',
            'summary': 'summary_writer_qwen_adapter', 'prose': 'prose_adapter', 'extractive': 'extractive_adapter', 'shuffled': 'abl3_chain_shuffled_adapter',
            'no_relation': 'abl3_chain_no_relation_adapter', 'triples': 'abl3_chain_triples_adapter', 'answer': 'abl3_direct_answer_adapter',
            'reasoning': 'reasoning_summ_adapter', 'summary_matched': 'summ_matched_adapter'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['train', 'write', 'eval', 'lookup', 'table'])
    p.add_argument('--dataset', choices=MANIFEST, default='musique')
    p.add_argument('--target', choices=TARGETS, default='chain')
    p.add_argument('--mixed', action='store_true')
    p.add_argument('--hops', choices=['all','2hop','23hop'], default='all')
    p.add_argument('--seed', type=int, default=20261027)
    p.add_argument('--budget', type=int, default=128)
    p.add_argument('--steps', type=int)
    p.add_argument('--max-new-tokens', type=int, default=None)
    p.add_argument('--method', choices=['chain_writer','summary','full','bm25','hybrid_chain','llmlingua','trunc_head','trunc_tail'],default='chain_writer')
    p.add_argument('--reader', type=Path, default=ROOT/'models/Qwen3.5-4B')
    p.add_argument('--adapter', type=Path)
    p.add_argument('--preview', action='store_true')
    a=p.parse_args()
    tag=('mixed' if a.mixed else a.target) + ('' if a.hops=='all' else '_'+a.hops)
    if a.seed != 20261027:tag += '_seed'+str(a.seed)
    adapter=a.adapter or ROOT/'sota/infoorder'/('chain_writer_multidata_adapter' if a.mixed else ADAPTERS[a.target])
    if a.adapter is None and (a.hops!='all' or a.seed!=20261027):adapter=ROOT/'sota/infoorder'/(tag+'_adapter')
    manifest=ROOT/'sota/data'/(MANIFEST[a.dataset]+'.jsonl')
    memory=ROOT/'sota/data'/(tag+'_'+a.dataset+'_b'+str(a.budget)+'_memories.jsonl')
    out=ROOT/'sota/results'/(tag+'_'+a.dataset+'_'+a.method+'_b'+str(a.budget)+'.jsonl')
    env=os.environ.copy()
    data='multidata_chain_sft_train' if a.mixed else TARGETS[a.target]
    if a.hops!='all':data='chain_sft_train_'+a.hops
    env.update(SOTA_SFT_DATA=str(ROOT/'sota/data'/(data+'.jsonl')),
               SOTA_ADAPTER_OUT=str(adapter),SOTA_ADAPTER=str(adapter),
               SOTA_SEED=str(a.seed),SOTA_MAX_LEN='2048' if a.mixed or a.target in ('facts','summary','extractive') else '1536',
               SOTA_SFT_STEPS=str(a.steps or (400 if a.mixed else 300)),
               SOTA_GEN_MANIFEST=str(manifest),SOTA_GEN_OUT=str(memory),
               SOTA_GEN_MAXNEW=str(a.max_new_tokens or (160 if a.target in ('prose','summary','extractive','reasoning','summary_matched') else 128)),SOTA_GEN_BUDGET=str(a.budget),
               SOTA_MEMORY_FILE=str(memory),SOTA_MEMORY_MANIFEST=str(manifest))
    prompts = {
        'reasoning': 'You write a compact reasoning-style summary for a multi-hop question. Restate the reasoning that leads to the answer as a few fluent prose sentences. Use only information present in the evidence. Be terse.',
        'summary_matched': 'You write a compact abstractive summary of the evidence for a multi-hop question. Summarize the facts needed to answer the question in a few fluent sentences. Use only information present in the evidence. Be terse.'}
    if a.target in prompts: env['SOTA_SYSTEM'] = prompts[a.target]
    if a.stage=='train':cmd=['sota/infoorder/train_chain_writer.py']
    elif a.stage=='write':cmd=['sota/infoorder/gen_chain_memories.py']
    elif a.stage=='table':cmd=['sota/analysis/gen_chain_table.py']
    elif a.stage=='lookup':
        env.update(CF_PER_FAMILY='15',CF_MANIFEST=str(ROOT/'sota/data/simple_lookup_200.jsonl'),CF_OUT=str(ROOT/'sota/results/simple_lookup_60.jsonl'))
        cmd=['sota/analysis/eval_both_correct_controlled.py']
    else:
        cmd=['sota/harness/run_eval.py','--method',a.method,'--budget',str(a.budget),'--manifest',str(manifest),
             '--model',str(a.reader),'--out',str(out),'--batch-size','8']
    if a.preview:
        print(' '.join([sys.executable]+cmd));return
    subprocess.run([sys.executable]+cmd,cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()
