import hashlib
import json
from pathlib import Path


def reader_inputs(row, record):
    """Compact readers never receive the writer's original history or states."""
    return row['question'], record['memory'], row['later_evidence']


def grouped_prefixes(rows):
    groups, histories = {}, {}
    for row in rows:
        value = [row['group_id'], row['question'], row['evidence_chunks']]
        key = hashlib.sha256(json.dumps(value).encode()).hexdigest()
        if row['group_id'] in histories and histories[row['group_id']] != key:
            raise ValueError('A comparison group contains different observed prefixes')
        histories[row['group_id']] = key
        if key not in groups:
            groups[key] = {'key': key, 'group_id': row['group_id'], 'question': row['question'],
                           'evidence_chunks': row['evidence_chunks'], 'ids': []}
        groups[key]['ids'].append(row['id'])
    return groups


def validate_memory(record, group, method, budget, tokenizer=None):
    if (record['key'] != group['key'] or record['group_id'] != group['group_id']
            or record['ids'] != group['ids'] or not isinstance(record['memory'], str)):
        raise ValueError('Emitted memory is not bound to the fixed observed prefix')
    if method in ('full', 'none'):
        expected = '\n\n'.join(group['evidence_chunks']) if method == 'full' else ''
        if record['memory'] != expected or record['states'] or record['state_tokens'] or record['writer_calls']:
            raise ValueError('Reference memory differs from observed evidence')
    else:
        if (len(record['states']) != len(group['evidence_chunks']) or not record['states']
                or record['memory'] != record['states'][-1]
                or len(record['state_tokens']) != len(record['states'])):
            raise ValueError('Incomplete writer update sequence')
        if any(not isinstance(text, str) for text in record['states']):
            raise ValueError('Invalid emitted memory state')
        if any(not isinstance(n, int) or n < 0 or n > budget for n in record['state_tokens']):
            raise ValueError('Writer memory exceeds its fixed token budget')
        if tokenizer is not None:
            counts = [len(tokenizer.encode(text, add_special_tokens=False)) for text in record['states']]
            if counts != record['state_tokens']:
                raise ValueError('Recorded writer token counts differ from actual tokenization')


def load_memories(path, groups, method, budget, tokenizer=None, complete=False):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []
    keys = [record['key'] for record in records]
    if len(keys) != len(set(keys)) or set(keys) - set(groups):
        raise ValueError('Duplicate or unexpected emitted prefixes')
    for record in records:
        validate_memory(record, groups[record['key']], method, budget, tokenizer)
    if complete and set(keys) != set(groups):
        raise ValueError('Emitted memory coverage is incomplete')
    return {record['key']: record for record in records}
