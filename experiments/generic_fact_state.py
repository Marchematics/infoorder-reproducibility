from collections import OrderedDict
from itertools import product
import re

BUDGET = 128
RULE = {
    'source': 'actual previous_memory followed by actual evidence; no question or future fields',
    'facts': 'observed atomic switch/report/status/flag values and item/next edges; no composition',
    'previous_teacher': 'already supplied code-status/start-flag values remain supplied facts',
    'duplicates': 'keep the first occurrence of an identical triple; retain differing values',
    'selection': 'visit triples in observation order; append a complete triple if the whole rendered state fits; skip otherwise',
    'format': 'group retained triples by predicate in first-occurrence order; preserve within-group observation order',
    'prefixes': 'factor shared underscore-delimited identifier prefixes within a predicate; choose the shortest tokenized rendering, literal wins ties',
    'unresolved': 'fixed Unresolved: future evidence. independent of question and task family',
    'budget_phi_tokens': BUDGET,
}

RAW = [
    (r'Channel ([\w-]+) (?:has switch bit|recorded switch value) ([01])\.', 'switch'),
    (r'Source ([\w-]+) (?:reported bit|recorded observation) ([01])\.', 'reported'),
    (r'Code ([\w-]+) identifies item ([\w-]+)\.', 'item'),
    (r'Item ([\w-]+) has status bit ([01])\.', 'status'),
    (r'(?:Start|Node) ([\w-]+) leads to (?:junction|node) ([\w-]+)\.', 'next'),
    (r'(?:Junction|Node) ([\w-]+) has flag bit ([01])\.', 'flag'),
]
LEGACY = {'Observed switch bits': 'switch', 'Observed reported bits': 'reported',
          'Observed status by code': 'code_status',
          'Observed terminal flags by start': 'start_flag'}


def observed_facts(text):
    """Read literal observed triples in order, without consulting a query."""
    if not isinstance(text, str):
        raise ValueError('Observed input must be text')
    events = []
    for pattern, predicate in RAW:
        for match in re.finditer(pattern, text):
            events.append((match.start(), (match[1], predicate, match[2])))
    for heading, predicate in LEGACY.items():
        pattern = re.escape(heading) + r'(?: \[prefix=([^\]\n]+)\])?:\n([^\n]*)'
        for match in re.finditer(pattern, text):
            prefix = match[1] or ''
            for fact in re.finditer(r'([\w-]+)=([01])(?:;|$)', match[2]):
                events.append((match.start(2) + fact.start(),
                               (prefix + fact[1], predicate, fact[2])))
    result, seen = [], set()
    for _, triple in sorted(events, key=lambda event: event[0]):
        if triple not in seen:
            result.append(triple)
            seen.add(triple)
    return result


def _prefix(names):
    if len(names) < 2 or any('_' not in name for name in names):
        return ''
    values = {name.rsplit('_', 1)[0] + '_' for name in names}
    return values.pop() if len(values) == 1 else ''


def render_facts(facts, tokenizer):
    """Compact typed records, with reversible prefix factoring only."""
    groups = OrderedDict()
    for subject, predicate, value in facts:
        groups.setdefault(predicate, []).append((subject, value))
    lines = []
    for predicate, pairs in groups.items():
        relation = predicate in ('item', 'next')
        left = _prefix([subject for subject, _ in pairs])
        right = _prefix([value for _, value in pairs]) if relation else ''
        candidates = []
        for use_left, use_right in product((False, True) if left else (False,),
                                           (False, True) if right else (False,)):
            lp, rp = left if use_left else '', right if use_right else ''
            meta = (f' [from={lp};to={rp}]' if relation and (lp or rp) else
                    f' [prefix={lp}]' if lp else '')
            content = ''.join(subject[len(lp):] + ('>' if relation else '=') +
                              value[len(rp):] + ';' for subject, value in pairs)
            candidate = predicate + meta + ':\n' + content
            candidates.append(candidate)
        lines.append(min(candidates, key=lambda value: len(tokenizer.encode(value, add_special_tokens=False))))
    return 'Observed facts:\n' + '\n'.join(lines) + '\nUnresolved: future evidence.'


def pack_facts(text, tokenizer, budget=BUDGET):
    """Greedy whole-triple packing under the actual tokenizer budget."""
    if type(budget) is not int or budget < 1:
        raise ValueError('Positive integer token budget required')
    facts = observed_facts(text)
    if not facts:
        raise ValueError('No supported observed atomic fact')
    accepted = []
    for fact in facts:
        candidate = render_facts(accepted + [fact], tokenizer)
        if len(tokenizer.encode(candidate, add_special_tokens=False)) <= budget:
            accepted.append(fact)
    if not accepted:
        raise ValueError('Budget cannot retain one complete atomic fact')
    memory = render_facts(accepted, tokenizer)
    return memory, accepted


def replace_targets(examples, tokenizer, budget=BUDGET):
    """Keep the frozen input records and row order, changing only target."""
    result = []
    for original in examples:
        observed = original['previous_memory'] + '\n' + original['evidence']
        memory, _ = pack_facts(observed, tokenizer, budget)
        updated = dict(original)
        updated['target'] = memory
        result.append(updated)
    return result
