# Natural Language Toolkit: Tokenizers
#
# Copyright (C) 2001-2021 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com> (minor additions)
# Contributors: matthewmc, clouds56
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

r"""
NLTK Tokenizer Package

Tokenizers divide strings into lists of substrings.  For example,
tokenizers can be used to find the words and punctuation in a string:

    >>> from nltkma.tokenize import word_tokenize
    >>> s = '''Good muffins cost $3.88\nin New York.  Please buy me
    ... two of them.\n\nThanks.'''
    >>> word_tokenize(s)
    ['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York', '.',
    'Please', 'buy', 'me', 'two', 'of', 'them', '.', 'Thanks', '.']

This particular tokenizer requires the Punkt sentence tokenization
models to be installed. NLTK also provides a simpler,
regular-expression based tokenizer, which splits text on whitespace
and punctuation:

    >>> from nltkma.tokenize import wordpunct_tokenize
    >>> wordpunct_tokenize(s)
    ['Good', 'muffins', 'cost', '$', '3', '.', '88', 'in', 'New', 'York', '.',
    'Please', 'buy', 'me', 'two', 'of', 'them', '.', 'Thanks', '.']

We can also operate at the level of sentences, using the sentence
tokenizer directly as follows:

    >>> from nltkma.tokenize import sent_tokenize, word_tokenize
    >>> sent_tokenize(s)
    ['Good muffins cost $3.88\nin New York.', 'Please buy me\ntwo of them.', 'Thanks.']
    >>> [word_tokenize(t) for t in sent_tokenize(s)]
    [['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York', '.'],
    ['Please', 'buy', 'me', 'two', 'of', 'them', '.'], ['Thanks', '.']]

Caution: when tokenizing a Unicode string, make sure you are not
using an encoded version of the string (it may be necessary to
decode it first, e.g. with ``s.decode("utf8")``.

NLTK tokenizers can produce token-spans, represented as tuples of integers
having the same semantics as string slices, to support efficient comparison
of tokenizers.  (These methods are implemented as generators.)

    >>> from nltkma.tokenize import WhitespaceTokenizer
    >>> list(WhitespaceTokenizer().span_tokenize(s))
    [(0, 4), (5, 12), (13, 17), (18, 23), (24, 26), (27, 30), (31, 36), (38, 44),
    (45, 48), (49, 51), (52, 55), (56, 58), (59, 64), (66, 73)]

There are numerous ways to tokenize text.  If you need more control over
tokenization, see the other methods provided in this package.

For further information, please see Chapter 3 of the NLTK book.
"""

import re

from nltkma.data import load
from nltkma.tokenize.casual import TweetTokenizer, casual_tokenize
from nltkma.tokenize.mwe import MWETokenizer
from nltkma.tokenize.destructive import NLTKWordTokenizer
from nltkma.tokenize.punkt import PunktSentenceTokenizer
from nltkma.tokenize.regexp import (
    RegexpTokenizer,
    WhitespaceTokenizer,
    BlanklineTokenizer,
    WordPunctTokenizer,
    wordpunct_tokenize,
    regexp_tokenize,
    blankline_tokenize,
)
from nltkma.tokenize.repp import ReppTokenizer
from nltkma.tokenize.sexpr import SExprTokenizer, sexpr_tokenize
from nltkma.tokenize.simple import (
    SpaceTokenizer,
    TabTokenizer,
    LineTokenizer,
    line_tokenize,
)
from nltkma.tokenize.texttiling import TextTilingTokenizer
from nltkma.tokenize.toktok import ToktokTokenizer
from nltkma.tokenize.treebank import TreebankWordTokenizer
from nltkma.tokenize.util import string_span_tokenize, regexp_span_tokenize
from nltkma.tokenize.stanford_segmenter import StanfordSegmenter
from nltkma.tokenize.sonority_sequencing import SyllableTokenizer
from nltkma.tokenize.legality_principle import LegalitySyllableTokenizer


# Standard sentence tokenizer.
def sent_tokenize(text, language="english"):
    """
    Return a sentence-tokenized copy of *text*,
    using NLTK's recommended sentence tokenizer
    (currently :class:`.PunktSentenceTokenizer`
    for the specified language).

    :param text: text to split into sentences
    :param language: the model name in the Punkt corpus
    """
    tokenizer = load("tokenizers/punkt/{0}.pickle".format(language))
    return tokenizer.tokenize(text)


# Standard word tokenizer.
_treebank_word_tokenizer = NLTKWordTokenizer()


def word_tokenize(text, language="english", preserve_line=False):
    """
    Return a tokenized copy of *text*,
    using NLTK's recommended word tokenizer
    (currently an improved :class:`.TreebankWordTokenizer`
    along with :class:`.PunktSentenceTokenizer`
    for the specified language).

    :param text: text to split into words
    :type text: str
    :param language: the model name in the Punkt corpus
    :type language: str
    :param preserve_line: An option to keep the preserve the sentence and not sentence tokenize it.
    :type preserve_line: bool
    """
    sentences = [text] if preserve_line else sent_tokenize(text, language)
    return [
        token for sent in sentences for token in _treebank_word_tokenizer.tokenize(sent)
    ]
