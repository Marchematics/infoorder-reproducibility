import argparse
import json
import random
from pathlib import Path
from continuing_worlds import chunks, digest


def make_world(rng, depth, width=6):
    names=[[f'{rng.choice(("pavo","mira","lyra","vela","nash","orion"))}_{rng.getrandbits(24):06x}'
            for _ in range(depth)] for _ in range(width)]
    permutation=list(range(width));rng.shuffle(permutation)
    return {'family':'multihop','depth':depth,'width':width,'paths':names,
            'permutation':permutation,'bits':[rng.randrange(2) for _ in range(width)]}


def triples(world):
    result=[]
    for i,path in enumerate(world['paths']):
        nodes=[f'start_{world["permutation"][i]}']+path
        result.extend([[a,'leads_to',b] for a,b in zip(nodes,nodes[1:])])
        result.append([path[-1],'flag',str(world['bits'][i])])
    return result


def facts(world):
    return [f'Node {a} leads to node {b}.' if relation=='leads_to' else f'Node {a} has flag bit {b}.'
            for a,relation,b in triples(world)]


def continuation(world,index,key):
    return f'The instruction selects start_{world["permutation"][index]}. The terminal connection bit is {key}.'


def record(world,rng,condition,updates):
    base_id=digest(world)
    changed=rng.randrange(world['width'])
    paired=dict(world,bits=list(world['bits']));paired['bits'][changed]^=1
    before,after=facts(world),facts(paired)
    ordering=list(range(len(before)))
    if condition=='shuffled':rng.shuffle(ordering)
    if condition=='reversed':ordering.reverse()
    before=[before[i] for i in ordering];after=[after[i] for i in ordering]
    if condition=='redundant':before=before*4;after=after*4
    if condition=='irrelevant':
        noise=[f'Workshop room_{i} stores {rng.randrange(20,80)} empty crates.' for i in range(len(before))]
        before=[text for pair in zip(before,noise) for text in pair]
        after=[text for pair in zip(after,noise) for text in pair]
    probes=[]
    indices=[changed]+[i for i in range(world['width']) if i!=changed]
    for i in indices:
        key=rng.randrange(2)
        probes.append({'text':continuation(world,i,key),'index':i,'key':key,'answer':world['bits'][i]^key})
    selected=rng.randrange(world['width']);key=rng.randrange(2)
    return {'id':digest([base_id,condition,updates]),'group_id':base_id,
        'world':world,'family':'multihop','condition':condition,'path_depth':world['depth'],
        'question':'Determine whether the selected route ends at an active terminal. A later instruction selects a start node. Follow the stored edges until a node with a flag is reached. The terminal state is this flag exclusive-or the later terminal connection bit. Exclusive-or is one when its inputs differ.',
        'current_target_probability':0.5,'history':'\n'.join(before),'paired_history':'\n'.join(after),
        'evidence_chunks':['\n'.join(part) for part in chunks(before,updates)],
        'changed_index':changed,'continuations':probes,
        'paired_continuations':[dict(p,answer=paired['bits'][p['index']]^p['key']) for p in probes],
        'later_evidence':continuation(world,selected,key),'answer':world['bits'][selected]^key,
        'later_index':selected,'later_key':key}


def build(count=128,seed=29):
    rng=random.Random(seed)
    train=[];test=[];seen=set()
    def fresh(depth):
        while True:
            world=make_world(rng,depth)
            fingerprint=digest([world['bits'],world['permutation'],depth])
            if fingerprint not in seen:
                seen.add(fingerprint);return world
    for _ in range(count):
        train.append(record(fresh(rng.choice([1,2])),rng,'ordinary',rng.choice([1,2])))
    for depth in (3,4,6):
        for _ in range(count):
            world=fresh(depth)
            for updates in (3,4,6):test.append(record(world,rng,'composition',updates))
    for _ in range(count):
        world=fresh(2)
        for condition in ('shuffled','reversed','redundant','irrelevant'):
            test.append(record(world,rng,condition,6))
    return train,test


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='data/continuing/composition')
    parser.add_argument('--count',type=int,default=128)
    parser.add_argument('--seed',type=int,default=29)
    args=parser.parse_args()
    target=Path(args.output);target.mkdir(parents=True,exist_ok=True)
    for name,rows in zip(('train','test'),build(args.count,args.seed)):
        (target/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        print(name,len(rows),len({r['group_id'] for r in rows}))
