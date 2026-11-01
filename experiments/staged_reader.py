import re


def trace_prompt(question, memory, later):
    system = ('Reason only from the supplied evidence. Trace the relations needed '
              'for the original question. Do not replace the requested answer '
              'with an intermediate entity or add conditions absent from the question.')
    user = (f'Original question:\n{question}\n\nEarlier evidence:\n{memory}'
            f'\n\nLater evidence:\n{later}\n\n'
            'First state the attribute the question asks for. Then follow the '
            'necessary links in the evidence, including any stated interpretation '
            'rule. Ignore unrelated passages. If a required link is missing, say '
            'it is not established. Give a brief evidence-based reasoning trace.')
    return system, user


def answer_prompt(question, memory, later, trace, binary=False):
    rule = ('Return exactly 0 or 1.' if binary else
            'Return only the shortest answer phrase, not a sentence or explanation. '
            'Use UNANSWERABLE if the supplied evidence does not establish an answer.')
    system = 'Answer the original question from supplied evidence. ' + rule
    user = (f'Original question:\n{question}\n\nEarlier evidence:\n{memory}'
            f'\n\nLater evidence:\n{later}\n\nProvisional reasoning:\n{trace}'
            '\n\nThe reasoning may be wrong; check it against the evidence. '
            'Answer the requested final attribute, not an intermediate entity. '
            + rule + f'\n\nQuestion to answer:\n{question}')
    return system, user


def parse_short_answer(text):
    """Accept predefined one-line forms; never extract using reference answers."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    answer = re.sub(r'^(?:Final answer|Answer):\s*', '', lines[0], flags=re.I).strip()
    if not answer or answer.upper() == 'UNANSWERABLE':
        return None
    return answer


def read(runtime, question, memory, later='', binary=False,
         trace_budget=256, answer_budget=32):
    system, user = trace_prompt(question, memory, later)
    trace = runtime.generate(system, user, trace_budget, reader=True)
    system, user = answer_prompt(question, memory, later, trace, binary)
    if binary:
                                                                              
                                                                                
        scores, probabilities = runtime.staged_binary_scores(system, user)
        answer = str(int(probabilities[1] > probabilities[0]))
        return {'trace': trace, 'raw': answer, 'answer': answer, 'valid': True,
                'probabilities': probabilities, 'candidate_logprobs': scores}
    raw = runtime.generate(system, user, answer_budget, reader=True)
    answer = parse_short_answer(raw)
    return {'trace': trace, 'raw': raw, 'answer': answer, 'valid': answer is not None,
            'probabilities': None}
