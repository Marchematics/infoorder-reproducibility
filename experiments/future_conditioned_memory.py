from collections import Counter, defaultdict
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
import math


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def probability(value):
    if type(value) not in (int, float, Fraction) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Need a finite probability, not bool')
    return value


def entropy(p):
    p = probability(p)
    if p in (0, 1):
        return 0.0
    def log_probability(q):
                                                                             
                                                                       
        if isinstance(q, Fraction):
            return math.log(q.numerator) - math.log(q.denominator)
        return math.log(q)
    return -float(p) * log_probability(p) - float(1 - p) * log_probability(1 - p)


def posterior_gap(posteriors, weights):
    """One positive (M,Z) cell; conditional iid pair discrepancy is exact."""
    if not posteriors or len(posteriors) != len(weights):
        raise ValueError('Need matching nonempty conditional laws')
    values = tuple(posteriors) + tuple(weights)
    if any(type(p) not in (int, Fraction) for p in values):
        raise ValueError('Use exact rational probabilities and masses')
    for p in values:
        probability(p)
    posteriors = tuple(Fraction(p) for p in posteriors)
    weights = tuple(Fraction(w) for w in weights)
    if sum(weights) != 1:
        raise ValueError('Conditional masses must sum exactly to one')
    mean = sum(w * p for w, p in zip(weights, posteriors))
    variance = sum(w * (p - mean) ** 2 for w, p in zip(weights, posteriors))
                                                                             
                                                                             
                                                                              
                                                                            
    def numerical_information(precision):
        with localcontext() as context:
            context.prec = precision
            return calculate_information()

    def calculate_information():
        def decimal(q):
            return Decimal(q.numerator) / Decimal(q.denominator)
        information_decimal = Decimal(0)
        if mean not in (0, 1):
            for p, w in zip(posteriors, weights):
                if not w:
                    continue
                value = Decimal(0)
                if p:
                    value += decimal(p) * (decimal(p) / decimal(mean)).ln()
                if p != 1:
                    value += decimal(1 - p) * (decimal(1 - p) / decimal(1 - mean)).ln()
                information_decimal += decimal(w) * value
        return information_decimal

    low_precision = numerical_information(80)
    information_decimal = numerical_information(160)
    with localcontext() as context:
        context.prec = 160
        def decimal(q):
            return Decimal(q.numerator) / Decimal(q.denominator)
        discrepancy_decimal = decimal(2 * variance)
        tolerance = Decimal('1e-25')
        if variance and (information_decimal <= 0 or abs(low_precision - information_decimal) > abs(information_decimal) * tolerance):
            raise ValueError('80/160-digit KL precision agreement unresolved')
        epsilon = min(min(p, 1 - p) for p, w in zip(posteriors, weights) if w)
        if variance and information_decimal < discrepancy_decimal * (1 - tolerance):
            raise ValueError('Numerical KL violates exact lower curvature bound')
        if epsilon:
            upper = decimal(variance / (2 * epsilon * (1 - epsilon)))
            if information_decimal > upper * (1 + tolerance):
                raise ValueError('Numerical KL violates exact upper curvature bound')
        information = float(information_decimal)
    if variance and (not math.isfinite(information) or information <= 0 or float(variance) == 0):
        raise ValueError('Positive discrepancy/information cannot be represented as a float result')
    return {'mean': mean, 'discrepancy': 2 * variance, 'information_nats': information}


def repeated_history(history, dimension, factor):
    """Injective repetition, not new independent facts or expanded full support."""
    integer(dimension, 'dimension', 1)
    integer(factor, 'factor', 1)
    integer(history, 'history')
    if history >= 1 << dimension:
        raise ValueError('History outside original support')
    return sum(history << (dimension * copy) for copy in range(factor))


def basis(vectors, dimension):
    """Canonical reduced GF(2) row basis, using integer bit masks."""
    integer(dimension, 'dimension')
    pivots = {}
    for value in vectors:
        integer(value, 'vector')
        if value >= 1 << dimension:
            raise ValueError('Vector outside the declared history dimension')
        for pivot in sorted(pivots, reverse=True):
            if value & (1 << pivot):
                value ^= pivots[pivot]
        if value:
            pivot = value.bit_length() - 1
            for old in pivots:
                if pivots[old] & (1 << pivot):
                    pivots[old] ^= value
            pivots[pivot] = value
    return tuple(pivots[pivot] for pivot in sorted(pivots, reverse=True))


def in_span(vector, rows, dimension):
    return len(basis(tuple(rows) + (vector,), dimension)) == len(basis(rows, dimension))


def linear_accuracy(query_basis, stored, dimension, distribution='basis_uniform'):
    """Exact Bayes accuracy after revealing (v,K), averaged over queries.

    Uniform A makes unanswerable linear queries fair within every memory cell.
    'span_uniform' ranges over every nonzero vector, counted algebraically.
    """
    rows = tuple(query_basis)
    rank = len(basis(rows, dimension))
    if not rows or rank != len(rows):
        raise ValueError('Need a nonempty independent query basis')
    stored = basis(stored, dimension)
    if distribution == 'basis_uniform':
        known = sum(in_span(v, stored, dimension) for v in rows)
        total = len(rows)
    elif distribution == 'span_uniform':
        intersection = rank + len(stored) - len(basis(rows + stored, dimension))
        known, total = (1 << intersection) - 1, (1 << rank) - 1
    else:
        raise ValueError('Unknown query distribution')
    return Fraction(total + known, 2 * total)


def enumerate_accuracy(queries, stored, dimension):
    """Independent small-world check, aggregating truth inside memory cells."""
    integer(dimension, 'dimension', 1)
    if dimension > 10 or not queries:
        raise ValueError('Enumeration is limited to dimension 10, nonempty queries')
    basis(queries, dimension)
    basis(stored, dimension)
    correct = 0
    for query in queries:
        for key in (0, 1):
            cells = defaultdict(Counter)
            for history in range(1 << dimension):
                memory = tuple((history & v).bit_count() % 2 for v in stored)
                target = (history & query).bit_count() % 2 ^ key
                cells[memory][target] += 1
            correct += sum(max(counts.values()) for counts in cells.values())
    return Fraction(correct, (1 << dimension) * 2 * len(queries))


def continuation_code(candidates, dimension, budget):
    """Keep the first independent candidate parities; no evaluation answers."""
    integer(budget, 'budget')
    basis(candidates, dimension)
    selected = []
    for v in candidates:
        if len(selected) < budget and not in_span(v, selected, dimension):
            selected.append(v)
    return tuple(selected)


def coverage_inputs(scores, weights):
    if not scores or len(scores) != len(weights) or not scores[0]:
        raise ValueError('Need nonempty pair-by-candidate scores and pair weights')
    n = len(scores[0])
    if any(len(row) != n for row in scores):
        raise ValueError('Ragged candidate score matrix')
    for value in list(weights) + [x for row in scores for x in row]:
        if type(value) not in (int, float, Fraction) or not math.isfinite(value) or value < 0:
            raise ValueError('Scores/weights must be finite and nonnegative, not bool')
    return n


def coverage_value(scores, weights, selected):
    n = coverage_inputs(scores, weights)
    if len(set(selected)) != len(selected) or any(type(i) is not int or not 0 <= i < n for i in selected):
        raise ValueError('Need distinct valid candidate indices')
    return sum(w * max((row[i] for i in selected), default=0) for row, w in zip(scores, weights))


def greedy_coverage(scores, weights, budget):
    """Fixed-score max coverage; deterministic earliest-index tie breaking."""
    n = coverage_inputs(scores, weights)
    integer(budget, 'budget')
    selected, values, evaluations = [], [0], 0
    while len(selected) < min(budget, n):
        options = []
        for i in range(n):
            if i not in selected:
                options.append((coverage_value(scores, weights, selected + [i]), -i))
                evaluations += 1
        value, negative_index = max(options)
        selected.append(-negative_index)
        values.append(value)
    return {'indices': selected, 'values': values, 'marginal_evaluations': evaluations}


def binary_js(p, q):
    return entropy((probability(p) + probability(q)) / 2) - (entropy(p) + entropy(q)) / 2


def legal_tail(selector, key, dimension, length):
    integer(dimension, 'dimension', 1)
    integer(length, 'length', 1)
    integer(selector, 'selector')
    if selector >= dimension or type(key) is not int or key not in (0, 1):
        raise ValueError('Invalid selector or key')
    return tuple([('advance',)] * (length - 1) + [('reveal', selector, key)])


def prefix_posterior(bits, tail, length):
    """Observation-only grammar: no alternative shortcut reveals are legal."""
    integer(length, 'length', 1)
    if not bits or any(type(b) is not int or b not in (0, 1) for b in bits):
        raise ValueError('Need a nonempty binary history')
    if len(tail) > length or any(token != ('advance',) for token in tail[:length - 1]):
        raise ValueError('Illegal prefix')
    if len(tail) < length:
        return Fraction(1, 2)
    token = tail[-1]
    if len(token) != 3 or token[0] != 'reveal':
        raise ValueError('Final evidence must reveal selector and fair key')
    _, selector, key = token
    legal_tail(selector, key, len(bits), length)
    return Fraction(bits[selector] ^ key)


def active_witness(left, right, length, horizon):
    """Both observed histories are available; no supplied changed-index oracle.

    A full scan is explicitly charged. This is a symbolic witness constructor,
    not a learned beam-search implementation or a black-box-reader comparison.
    """
    integer(horizon, 'horizon')
    prefix_posterior(left, (), length)
    prefix_posterior(right, (), length)
    if len(left) != len(right):
        raise ValueError('Paired histories must have matching dimensions')
    differences = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
    if horizon < length or not differences:
        return {'tail': None, 'history_bit_comparisons': len(left), 'evidence_steps': 0, 'posterior_evaluations': 0}
    tail = legal_tail(differences[0], 0, len(left), length)
    if prefix_posterior(left, tail, length) == prefix_posterior(right, tail, length):
        raise AssertionError('Constructed tail is not a witness')
    return {'tail': tail, 'history_bit_comparisons': len(left), 'evidence_steps': length, 'posterior_evaluations': 2}


def enumerated_lookahead(dimension, length, horizon):
    """Enumerate the actual joint law A,J,K and group observed prefixes."""
    integer(dimension, 'dimension', 1)
    integer(length, 'length', 1)
    integer(horizon, 'horizon')
    if dimension > 10 or horizon > length:
        raise ValueError('Invalid enumeration size or horizon')
    full, compressed = defaultdict(Counter), defaultdict(Counter)
    for bits in product((0, 1), repeat=dimension):
        for selector in range(dimension):
            for key in (0, 1):
                prefix = legal_tail(selector, key, dimension, length)[:horizon]
                target = bits[selector] ^ key
                full[(bits, prefix)][target] += 1
                compressed[prefix][target] += 1
    denominator = (1 << dimension) * dimension * 2
    def conditional_entropy(cells):
        return math.fsum(sum(c.values()) / denominator * entropy(Fraction(c[1], sum(c.values()))) for c in cells.values())
    full_loss = conditional_entropy(full)
    constant_loss = conditional_entropy(compressed)
    return {'joint_atoms': denominator, 'full_loss_nats': full_loss,
            'constant_loss_nats': constant_loss, 'gap_nats': constant_loss - full_loss}
