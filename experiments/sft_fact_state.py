import re

SYSTEM = ('Store observed facts for the fixed task in compact factual memory. '
          'Keep exact selector bindings and observed bits. Distinguish reported '
          'bits from polarity, item status from the inspection rule, and terminal '
          'flags from the terminal connection. You may combine already observed '
          'links into status by code or terminal flag by start. For many names '
          'sharing a prefix, state the prefix once and list suffix=bit pairs. '
          'State which interpretation is unresolved. Do not predict the final '
          'answer or invent the future selection, calibration, or rule. '
          'Output the updated memory only; the fixed task remains available.')

HEADERS = {'switch': 'Observed switch bits', 'report': 'Observed reported bits',
           'status': 'Observed status by code', 'flag': 'Observed terminal flags by start'}
UNRESOLVED = {'parity': 'selection and wiring bit', 'source': 'selection and source polarity',
              'entity': 'selection and inspection rule', 'multihop': 'selection and terminal connection'}
ROLE = {'parity': 'switch', 'source': 'report', 'entity': 'status', 'multihop': 'flag'}


def observed_bindings(text, family):
    """Recognize exact public finite-world statements and our target format."""
    if family not in ROLE or not isinstance(text, str):
        raise ValueError('Unknown task family or non-text evidence')
    result = {}
    role = ROLE[family]
    header = re.escape(HEADERS[role])
    for match in re.finditer(header + r'(?: \[prefix=([^\]\n]+)\])?:\n([^\n]*)', text):
        prefix = match.group(1) or ''
        for name, bit in re.findall(r'([\w-]+)=([01])(?:;|$)', match.group(2)):
            result[prefix + name] = int(bit)
    if family == 'parity':
        result.update((name, int(bit)) for name, bit in re.findall(r'Channel ([\w-]+) (?:has switch bit|recorded switch value) ([01])\.', text))
    elif family == 'source':
        result.update((name, int(bit)) for name, bit in re.findall(r'Source ([\w-]+) (?:reported bit|recorded observation) ([01])\.', text))
    elif family == 'entity':
        links = dict(re.findall(r'Code ([\w-]+) identifies item ([\w-]+)\.', text))
        bits = dict(re.findall(r'Item ([\w-]+) has status bit ([01])\.', text))
        result.update((code, int(bits[item])) for code, item in links.items() if item in bits)
    else:
        links = dict(re.findall(r'(?:Start|Node) ([\w-]+) leads to (?:junction|node) ([\w-]+)\.', text))
        bits = dict(re.findall(r'(?:Junction|Node) ([\w-]+) has flag bit ([01])\.', text))
        for start in links:
            if not start.startswith('start_'):
                continue
            node, seen = start, set()
            while node in links and node not in seen:
                seen.add(node)
                node = links[node]
            if node in bits:
                result[start] = int(bits[node])
    return result


def target_state(observed, family, tokenizer, budget=128):
    """Pack complete selector facts; all counted in the fixed Phi token budget."""
    if type(budget) is not int or budget < 1:
        raise ValueError('Positive token budget required')
    bindings = observed_bindings(observed, family)
    if not bindings:
        raise ValueError('No supported observed selector binding; do not invent one')
    prefix = ''
    if family in ('parity', 'source'):
        prefixes = {name.rsplit('_', 1)[0] + '_' for name in bindings}
        if len(prefixes) == 1:
            prefix = prefixes.pop()
    header = HEADERS[ROLE[family]] + (f' [prefix={prefix}]' if prefix else '') + ':\n'
    suffix = '\nUnresolved: ' + UNRESOLVED[family] + '.'
    accepted = []
    for name in sorted(bindings):
        display = name[len(prefix):] if prefix else name
        fact = f'{display}={bindings[name]};'
        candidate = header + ''.join(accepted + [fact]) + suffix
        if len(tokenizer.encode(candidate, add_special_tokens=False)) <= budget:
            accepted.append(fact)
    if not accepted:
        raise ValueError('Budget cannot retain a complete observed binding')
    memory = header + ''.join(accepted) + suffix
    if len(tokenizer.encode(memory, add_special_tokens=False)) > budget:
        raise ValueError('Packing exceeded token budget')
    return memory


def sft_examples(row, tokenizer, budget=128):
    """Produce 3 examples for 2 updates, without reading any later/gold fields."""
    previous, prefix, examples = '', [], []
    for index, evidence in enumerate(row['evidence_chunks']):
        prefix.append(evidence)
        direct = target_state('\n'.join(prefix), row['family'], tokenizer, budget)
        examples.append({'id': row['id'] + f':direct:{index}', 'group_id': row['id'],
                         'family': row['family'], 'branch': 'direct', 'question': row['question'],
                         'previous_memory': '', 'evidence': '\n'.join(prefix), 'target': direct})
        recursive = target_state(previous + '\n' + evidence, row['family'], tokenizer, budget)
        if index:
            examples.append({'id': row['id'] + f':recursive:{index}', 'group_id': row['id'],
                             'family': row['family'], 'branch': 'recursive', 'question': row['question'],
                             'previous_memory': previous, 'evidence': evidence, 'target': recursive})
        previous = recursive
    return examples
