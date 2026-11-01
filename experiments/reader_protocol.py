import re

def reasoned_reader_prompt(question, memory, later='', binary=False):
    answer_rule = 'The final answer must be exactly 0 or 1.' if binary else 'Use the shortest answer phrase.'
    system = ('Use only the supplied evidence. Work through the relevant links briefly. '
              'Put the answer on the last line as Final answer: <answer>. ' + answer_rule)
    user = f'Stored evidence:\n{memory}\n\nNew evidence:\n{later}\n\nTask:\n{question}'
    return system, user

def parse_final_answer(text, binary=False):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    matches = [re.fullmatch(r'Final answer: (.+)', line) for line in lines]
    if sum(match is not None for match in matches) != 1 or matches[-1] is None:
        return None
    answer = matches[-1].group(1).strip()
    if binary and answer not in ('0', '1'):
        return None
    return answer or None
