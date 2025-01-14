"""
Tests for NIST translation evaluation metric
"""

import io
import unittest

from nltkma.data import find
from nltkma.translate.nist_score import corpus_nist


class TestNIST(unittest.TestCase):
    def test_sentence_nist(self):
        ref_file = find('models/wmt15_eval/ref.ru')
        hyp_file = find('models/wmt15_eval/google.ru')
        mteval_output_file = find('models/wmt15_eval/mteval-13a.output')

        # Reads the NIST scores from the `mteval-13a.output` file.
        # The order of the list corresponds to the order of the ngrams.
        with open(mteval_output_file, 'r') as mteval_fin:
            # The numbers are located in the last 4th line of the file.
            # The first and 2nd item in the list are the score and system names.
            mteval_nist_scores = map(float, mteval_fin.readlines()[-4].split()[1:-1])

        with io.open(ref_file, 'r', encoding='utf8') as ref_fin:
            with io.open(hyp_file, 'r', encoding='utf8') as hyp_fin:
                # Whitespace tokenize the file.
                # Note: split() automatically strip().
                hypotheses = list(map(lambda x: x.split(), hyp_fin))
                # Note that the corpus_bleu input is list of list of references.
                references = list(map(lambda x: [x.split()], ref_fin))
                # Without smoothing.
                for i, mteval_nist in zip(range(1, 10), mteval_nist_scores):
                    nltk_nist = corpus_nist(references, hypotheses, i)
                    # Check that the NIST scores difference is less than 0.5
                    assert abs(mteval_nist - nltk_nist) < 0.05
