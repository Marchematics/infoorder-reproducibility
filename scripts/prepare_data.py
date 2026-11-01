"""Unpack the fixed training and evaluation inputs."""
import argparse
import gzip
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

def prepare():
    dest = ROOT / 'sota/data'
    dest.mkdir(parents=True, exist_ok=True)
    (ROOT / 'sota/results').mkdir(exist_ok=True)
    for path in sorted((ROOT / 'data/inputs').glob('*.jsonl.gz')):
        target = dest / path.name[:-3]
        target.write_bytes(gzip.decompress(path.read_bytes()))
    mu = (dest / 'chain_sft_train.jsonl').read_bytes()
    wiki = (dest / 'twowiki_chain_sft_train.jsonl').read_bytes()
    (dest / 'multidata_chain_sft_train.jsonl').write_bytes(mu + wiki)
    for name, limit in [('2hop', {'2hop'}), ('23hop', {'2hop', '3hop'})]:
        rows = [r for line in mu.splitlines() for r in [json.loads(line)] if r['hop'] in limit]
        (dest / ('chain_sft_train_' + name + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    targets = ROOT / 'data/targets'
    if targets.exists():
        base = {json.loads(l)['id']: json.loads(l) for l in mu.splitlines()}
        for path in targets.glob('*.jsonl'):
            rows = [{**base[r['id']], 'target': r['target']} for line in path.read_text().splitlines() for r in [json.loads(line)]]
            (dest / path.name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    return dest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--musique-source', type=Path)
    args = parser.parse_args()
    dest = prepare()
    if args.musique_source:
        (dest/'data').mkdir(exist_ok=True)
        shutil.copyfile(args.musique_source, dest/'data/musique_ans_v1.0_dev.jsonl')
    print('Prepared inputs in sota/data')
