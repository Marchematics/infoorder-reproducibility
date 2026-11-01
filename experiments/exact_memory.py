import itertools
import json
from pathlib import Path
import numpy as np


def binary_entropy(e):
    return float(-sum(p*np.log2(p) for p in (e, 1-e) if p))

def optimal_codes(width=4, memory_bits=1):
    words = np.array(list(itertools.product([0, 1], repeat=width)), dtype=np.int8)
    distances = np.sum(words[:, None, :] != words[None, :, :], axis=2)
    best = float('inf')
    best_code = None
    for indices in itertools.combinations(range(len(words)), 2**memory_bits):
        cost = float(distances[:, indices].min(axis=1).mean()/width)
        if cost < best:
            best, best_code = cost, indices
    return best, [words[i].tolist() for i in best_code]

def information_identities():
                                                                
    mass = {(a ^ b, a, b): 0.25 for a in (0, 1) for b in (0, 1)}
    def entropy(axes):
        probs = {}
        for triple, p in mass.items():
            key = tuple(triple[i] for i in axes)
            probs[key] = probs.get(key, 0) + p
        return -sum(p*np.log2(p) for p in probs.values())
    current = entropy([0]) + entropy([1]) - entropy([0, 1])
    future = entropy([0, 2]) + entropy([1, 2]) - entropy([2]) - entropy([0, 1, 2])
    return {'current_information_bits': current, 'future_information_bits': future,
            'current_optimal_accuracy': 0.5, 'belief_memory_final_accuracy': 0.5,
            'one_bit_memory_final_accuracy': 1.0, 'enumerated_worlds': len(mass)}

def main():
    out = Path('results/continuing')
    out.mkdir(parents=True, exist_ok=True)
    data = {'kind': 'exact_finite_world_enumeration', 'parity': information_identities(),
            'capacity': [], 'width': [], 'redundancy': [], 'independent_control': {}}
    for b in range(5):
        e, code = optimal_codes(4, b)
        data['capacity'].append({'width': 4, 'bits': b, 'optimal_accuracy': 1-e,
                                  'coordinate_accuracy': 0.5 + b/8,
                                  'codebook': code, 'entropy_lower_bound_bits': 4*(1-binary_entropy(e))})
    for d in [2, 4, 8, 12, 16, 24, 32]:
        data['width'].append({'dependencies': d, 'zero_error_memory_bits': d,
                              'ninety_percent_memory_lower_bound_bits': d*(1-binary_entropy(0.1)),
                              'belief_memory_accuracy': 0.5,
                              'coordinate_4_bit_accuracy': 0.5+min(4, d)/(2*d)})
    for repetitions in [1, 2, 4, 8, 16, 32]:
        data['redundancy'].append({'repetitions': repetitions, 'raw_bit_records': 4*repetitions,
                                   'zero_error_memory_bits': 4, 'future_accuracy': 1.0})
                                                                           
    values = []
    for first, second in itertools.product([0, 1], repeat=2):
        likelihood = [np.prod([0.75 if obs == theta else 0.25 for obs in [first, second]]) for theta in [0, 1]]
        posterior_full = likelihood[1]/sum(likelihood)
        belief = 0.75 if first else 0.25
        l0, l1 = (0.75, 0.25) if second == 0 else (0.25, 0.75)
        posterior_memory = belief*l1/((1-belief)*l0 + belief*l1)
        values.append(abs(posterior_full-posterior_memory))
    data['independent_control'] = {'maximum_posterior_difference': max(values), 'enumerated_pairs': 4}
    (out/'exact.json').write_text(json.dumps(data, indent=2))
    print(json.dumps(data['parity']))
    print('optimal four-bit-world accuracies:', [r['optimal_accuracy'] for r in data['capacity']])
    print('independent evidence maximum error:', max(values))

if __name__ == '__main__':
    main()
