import gzip
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'sota/harness'))
from scoring import score_answer, answer_in_context
sys.path.insert(0, str(ROOT/'scripts'))
from analyze_results import analyze

class ReproductionTests(unittest.TestCase):
    def test_answer_scoring(self):
        self.assertEqual(score_answer('The Eiffel Tower.', ['Eiffel Tower'])['em'], 1)
        self.assertEqual(score_answer('Paris', ['City of Paris', 'Paris'])['em'], 1)
        self.assertEqual(score_answer('UNANSWERABLE', ['Paris'])['em'], 0)
        self.assertFalse(answer_in_context('York', 'Yorkshire'))
        self.assertTrue(answer_in_context('New York', 'Born in New York.'))

    def test_fixed_inputs(self):
        splits = json.loads((ROOT/'data/splits.json').read_text())
        expected={'chain_sft_train.jsonl':1917,'twowiki_chain_sft_train.jsonl':3000,
                  'manifest_500.jsonl':500,'manifest_2wiki_fixed_300.jsonl':300,
                  'manifest_hotpotqa_500.jsonl':500}
        for name,n in expected.items():
            rows=[json.loads(l) for l in gzip.decompress((ROOT/'data/inputs'/(name+'.gz')).read_bytes()).splitlines()]
            self.assertEqual([r['id'] for r in rows], splits[name])
            self.assertEqual(len(set(splits[name])),n)
        self.assertFalse(set(splits['chain_sft_train.jsonl']) & set(splits['manifest_500.jsonl']))
        self.assertFalse(set(splits['twowiki_chain_sft_train.jsonl']) & set(splits['manifest_2wiki_fixed_300.jsonl']))

    def test_reported_comparisons(self):
        result=analyze(ROOT/'results/scores')
        comparisons=result['paired_comparisons']
        self.assertAlmostEqual(result['runs']['musique500_chain_qwen_b128']['em'],.726)
        self.assertAlmostEqual(result['runs']['2wiki_multidata_chain_b128']['em'],224/300)
        self.assertAlmostEqual(comparisons['chain_vs_summary']['difference'],.316)
        self.assertAlmostEqual(comparisons['chain_vs_prose']['difference'],.192)
        self.assertAlmostEqual(comparisons['2wiki_adaptation']['difference'],.21)
        self.assertAlmostEqual(comparisons['retained_bindings']['difference'],16/60)
        self.assertEqual(comparisons['chain_vs_summary']['ci95'], [.266,.366])

if __name__=='__main__':unittest.main()
