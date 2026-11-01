import argparse
import hashlib
import json
import random
from pathlib import Path

FAMILIES = ('parity', 'entity', 'source', 'multihop')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def make_world(family, rng, width=16, namespace='train'):
    bits = [rng.randrange(2) for _ in range(width)]
    names = [f'{namespace}_object_{rng.getrandbits(32):08x}' for _ in bits]
    permutation = list(range(width))
    rng.shuffle(permutation)
    return {'family': family, 'bits': bits, 'names': names,
            'permutation': permutation, 'width': width}

def question(world):
    return {
        'parity': 'Determine whether the device will raise an alarm. Its alarm is the stored switch at the channel later selected by the controller, exclusive-or the later wiring bit. Exclusive-or is one when its inputs differ.',
        'entity': 'Determine whether the shipment will require inspection. A later dispatch identifies the item by its code. Inspection is the item status exclusive-or the later inspection rule bit.',
        'source': 'Determine whether the monitored event occurred. Each source reports the event exclusive-or its own polarity bit. A later calibration identifies a source and reveals its polarity.',
        'multihop': 'Determine whether the selected route ends at an active terminal. A later instruction selects a start node. Follow its stored edge to a junction. The terminal state is the junction flag exclusive-or the later terminal connection bit.',
    }[world['family']]

def facts(world, variant=0):
    names, bits, perm = world['names'], world['bits'], world['permutation']
    out = []
    for i, bit in enumerate(bits):
        name = names[i]
        if world['family'] == 'parity':
            text = f'Channel {name} has switch bit {bit}.'
        elif world['family'] == 'entity':
            text = f'Code code_{perm[i]} identifies item {name}. Item {name} has status bit {bit}.'
        elif world['family'] == 'source':
            text = f'Source {name} reported bit {bit}.'
        else:
            text = f'Start start_{perm[i]} leads to junction {name}. Junction {name} has flag bit {bit}.'
        if variant:
            text = text.replace('has switch bit', 'recorded switch value').replace('reported bit', 'recorded observation')
        out.append(text)
    return out

def continuation(world, index, key):
    name = world['names'][index]
    code = world['permutation'][index]
    return {
        'parity': f'The controller selects channel {name}. The wiring bit is {key}.',
        'entity': f'The dispatch selects code_{code}. The inspection rule bit is {key}.',
        'source': f'Calibration selects source {name}. This source polarity bit is {key}.',
        'multihop': f'The instruction selects start_{code}. The terminal connection bit is {key}.',
    }[world['family']]

def candidate_continuations(world, count, rng, changed_index=None):
    all_indices = list(range(world['width']))
    rng.shuffle(all_indices)
    if changed_index is not None:
        all_indices.remove(changed_index)
        all_indices.insert(0, changed_index)
    result = []
    for index in all_indices[:count]:
        key = rng.randrange(2)
        result.append({'text': continuation(world, index, key), 'index': index,
                       'key': key, 'answer': world['bits'][index] ^ key})
    return result

def chunks(values, updates):
    return [values[len(values)*i//updates:len(values)*(i+1)//updates] for i in range(updates)]

def record(world, rng, split, updates=2):
    changed = rng.randrange(world['width'])
    paired = dict(world, bits=list(world['bits']))
    paired['bits'][changed] ^= 1
    probes = candidate_continuations(world, 8, rng, changed)
    probes_paired = [dict(c, answer=paired['bits'][c['index']] ^ c['key']) for c in probes]
    index = rng.randrange(world['width'])
    key = rng.randrange(2)
    early = facts(world, variant=int(split == 'test'))
                                                                                 
    evidence = ['\n'.join(c) for c in chunks(early, updates)]
    later = continuation(world, index, key)
    return {
        'id': digest(world), 'latent_fingerprint': digest([world['family'], world['bits'], world['permutation']]),
        'split': split, 'family': world['family'], 'world': world,
        'question': question(world), 'current_target_probability': 0.5,
        'history': '\n'.join(early), 'evidence_chunks': evidence,
        'paired_history': '\n'.join(facts(paired, variant=int(split == 'test'))),
        'changed_index': changed, 'continuations': probes,
        'paired_continuations': probes_paired, 'later_evidence': later,
        'answer': world['bits'][index] ^ key, 'later_index': index, 'later_key': key,
    }

def build(seed=17, train=512, validation=128, test=512):
    rng = random.Random(seed)
    seen = set()
    outputs = {s: [] for s in ('train', 'validation', 'test')}
    for split, size in [('train', train), ('validation', validation), ('test', test)]:
        for family in FAMILIES:
            for _ in range(size):
                while True:
                    world = make_world(family, rng, namespace=split)
                    item = record(world, rng, split, updates=rng.choice([1, 2]))
                    if item['latent_fingerprint'] not in seen:
                        seen.add(item['latent_fingerprint'])
                        break
                outputs[split].append(item)
    return outputs

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='data/continuing')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--train', type=int, default=512)
    p.add_argument('--validation', type=int, default=128)
    p.add_argument('--test', type=int, default=512)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in build(args.seed, args.train, args.validation, args.test).items():
        target = out / f'{split}.jsonl'
        target.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        print(split, len(rows), hashlib.sha256(target.read_bytes()).hexdigest())
