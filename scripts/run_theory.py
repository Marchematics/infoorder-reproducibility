"""Compute the finite-world capacity and query-rank results."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import future_conditioned_memory as theory
from exact_memory import optimal_codes
def rank_suite():
    rows, crosschecks, repetition_checks = [], 0, 0
    for d in (4, 8, 16, 32, 64):
        for r in (1, 2, 4, 8, 16):
            if r > d:
                continue
            geometries = {
                'coordinate_front': tuple(1 << i for i in range(r)),
                'coordinate_tail': tuple(1 << (d - 1 - i) for i in range(r)),
                'shared_tail_parities': tuple((1 << i) ^ (1 << (d - 1)) if i < d - 1 else 1 << i for i in range(r)),
            }
            for name, queries in geometries.items():
                if len(theory.basis(queries, d)) != r:
                    raise AssertionError('Query geometry lost rank')
                for factor in (1, 4, 16, 64):
                    assert len(theory.basis(queries, d * factor)) == r
                    # Queries act on the first copy. The support is the diagonal
                    # repetition image of original A, NOT uniform F2^(d*factor).
                    for a in ([0, (1 << d) - 1] + [1 << i for i in range(d)]):
                        repeated = theory.repeated_history(a, d, factor)
                        assert repeated & ((1 << d) - 1) == a
                        assert all((repeated & v).bit_count() % 2 == (a & v).bit_count() % 2 for v in queries)
                        repetition_checks += 1
                budgets = set(range(r + 1)) | {b for b in (1, 2, 4, 8, 16, d) if b <= d}
                for b in sorted(budgets):
                    # Candidate copies do not inject new history information.
                    code = theory.continuation_code(queries + queries, d, b)
                    stored = {'current_posterior_only': (), 'raw_first_bits': tuple(1 << i for i in range(b)),
                              'task_relevant_linear': queries[:b], 'continuation_linear': code}
                    for distribution in ('basis_uniform', 'span_uniform'):
                        accuracies = {method: theory.linear_accuracy(queries, memory, d, distribution) for method, memory in stored.items()}
                        if d <= 8:
                            if distribution == 'basis_uniform':
                                explicit_queries = queries
                            else:
                                explicit_queries = tuple(v for v in range(1, 1 << d) if theory.in_span(v, queries, d))
                            for method, memory in stored.items():
                                assert theory.enumerate_accuracy(explicit_queries, memory, d) == accuracies[method]
                                crosschecks += 1
                        if b >= r:
                            assert accuracies['task_relevant_linear'] == accuracies['continuation_linear'] == 1
                        rows.append({'d': d, 'r': r, 'budget_bits': b, 'geometry': name,
                                     'query_distribution': distribution, 'query_basis_masks': queries,
                                     'accuracy_exact': accuracies, 'encoded_rank': len(code),
                                     'distinct_history_classes': 1 << r,
                                     'duplicate_history_factors': [1, 4, 16, 64],
                                     'exact_capacity_bits_each_duplicate_factor': [r] * 4})
    return {'rows': rows, 'small_enumeration_crosschecks': crosschecks,
            'duplicate_embedding_checks': repetition_checks,
            'duplicate_support': 'Injective repeated copies of original history; no extra independent bits. Query evaluation and rank checked on actual repetition embeddings.',
            'large_dimension_method': 'GF(2) intersection rank and exact integer class counting; not extrapolated enumeration',
            'encoder_family': 'linear'}

def entropy_suite():
    rows = []
    for a in range(21):
        for b in range(a, 21):
            for w in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)):
                ps = [Fraction(a, 20), Fraction(b, 20)]
                result = theory.posterior_gap(ps, [w, 1 - w])
                discrepancy, information = result['discrepancy'], result['information_nats']
                assert information + 1e-13 >= float(discrepancy)
                eps = min(ps + [1 - p for p in ps])
                upper = discrepancy / (4 * eps * (1 - eps)) if eps else None
                global_upper = theory.entropy(math.sqrt(float(discrepancy / 2)))
                assert information <= global_upper + 1e-13
                if upper is not None:
                    assert information <= float(upper) + 1e-13
                rows.append({'posteriors': ps, 'weights': [w, 1 - w], **result,
                             'epsilon': eps, 'upper_bound': upper,
                             'global_continuity_upper_nats': global_upper})
    return {'cells': rows, 'verified_cells': len(rows),
            'arithmetic': 'rational discrepancy/interior bounds; KL uses 80/160-digit agreement and curvature guards then float output, not certified interval arithmetic. Global modulus uses float entropy. These checks are not formal proof.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('results/theory.json'))
    a=p.parse_args()
    result={'capacity':[1-optimal_codes(4,b)[0] for b in range(5)],
            'rank':rank_suite(),'entropy':entropy_suite()}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(json.dumps({'capacity':result['capacity'],'rank_configurations':len(result['rank']['rows']),
          'enumeration_checks':result['rank']['small_enumeration_crosschecks'],
          'repetition_checks':result['rank']['duplicate_embedding_checks']}))
