"""Check both branches in the symbolic equal-belief task."""
import json
from pathlib import Path
import re
def parse_history(family, text):
    bits, code2item, start2junc = {}, {}, {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if family == "parity":
            for m in re.finditer(r"Channel (\S+) (?:has switch bit|recorded switch value) (\d)", line):
                bits[m.group(1)] = int(m.group(2))
        elif family == "entity":
            for m in re.finditer(r"Code (code_\d+) identifies item (\S+?)\.", line):
                code2item[m.group(1)] = m.group(2)
            for m in re.finditer(r"Item (\S+) has status bit (\d)", line):
                bits[m.group(1)] = int(m.group(2))
        elif family == "source":
            for m in re.finditer(r"Source (\S+) (?:reported bit|recorded observation) (\d)", line):
                bits[m.group(1)] = int(m.group(2))
        else:
            for m in re.finditer(r"Start (start_\d+) leads to junction (\S+?)\.", line):
                start2junc[m.group(1)] = m.group(2)
            for m in re.finditer(r"Junction (\S+) has flag bit (\d)", line):
                bits[m.group(1)] = int(m.group(2))
    return bits, code2item, start2junc

def answer(family, bits, code2item, start2junc, cont):
    cont = cont.strip()
    if family == "parity":
        m = re.match(r"The controller selects channel (\S+)\. The wiring bit is (\d)", cont)
        if not m: return None
        name, key = m.group(1), int(m.group(2))
        return None if name not in bits else bits[name] ^ key
    if family == "entity":
        m = re.match(r"The dispatch selects (code_\d+)\. The inspection rule bit is (\d)", cont)
        if not m: return None
        item = code2item.get(m.group(1)); key = int(m.group(2))
        return None if item is None or item not in bits else bits[item] ^ key
    if family == "source":
        m = re.match(r"Calibration selects source (\S+)\. This source polarity bit is (\d)", cont)
        if not m: return None
        name, key = m.group(1), int(m.group(2))
        return None if name not in bits else bits[name] ^ key
    m = re.match(r"The instruction selects (start_\d+)\. The terminal connection bit is (\d)", cont)
    if not m: return None
    j = start2junc.get(m.group(1)); key = int(m.group(2))
    return None if j is None or j not in bits else bits[j] ^ key

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    rows=[json.loads(l) for l in (root/'sota/data/controlled_bothcorrect_200.jsonl').read_text().splitlines()]
    both=0
    for r in rows:
        correct=[]
        for h,a in [('history','answer_A'),('paired_history','answer_B')]:
            bits,codes,starts=parse_history(r['family'],r[h])
            correct.append(answer(r['family'],bits,codes,starts,r['continuation'])==r[a])
        both+=all(correct)
    assert both==len(rows)==200
    print(json.dumps({'pairs':len(rows),'full_bindings_both_correct':both,'current_belief_both_correct':0}))
