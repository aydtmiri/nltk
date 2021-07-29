from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

## Test bigram counters with discontinuous bigrams and repeated words

_EPSILON = 1e-8
SENT = 'this this is is a a test test'.split()


def close_enough(x, y):
    """Verify that two sequences of n-gram association values are within
       _EPSILON of each other.
    """

    return all(abs(x1[1] - y1[1]) <= _EPSILON for x1, y1 in zip(x, y))


def test_bigram2():
    b = BigramCollocationFinder.from_words(SENT)

    assert sorted(b.ngram_fd.items()) == [
        (('a', 'a'), 1),
        (('a', 'test'), 1),
        (('is', 'a'), 1),
        (('is', 'is'), 1),
        (('test', 'test'), 1),
        (('this', 'is'), 1),
        (('this', 'this'), 1),
    ]
    assert sorted(b.word_fd.items()) == [
        ('a', 2), ('is', 2), ('test', 2), ('this', 2)
    ]

    assert len(SENT) == sum(b.word_fd.values()) == sum(b.ngram_fd.values()) + 1
    assert close_enough(
        sorted(b.score_ngrams(BigramAssocMeasures.pmi)),
        [
            (('a', 'a'), 1.0),
            (('a', 'test'), 1.0),
            (('is', 'a'), 1.0),
            (('is', 'is'), 1.0),
            (('test', 'test'), 1.0),
            (('this', 'is'), 1.0),
            (('this', 'this'), 1.0),
        ]

    )


def test_bigram3():
    b = BigramCollocationFinder.from_words(SENT, window_size=3)
    assert sorted(b.ngram_fd.items()) == sorted([
        (('a', 'test'), 3),
        (('is', 'a'), 3),
        (('this', 'is'), 3),
        (('a', 'a'), 1),
        (('is', 'is'), 1),
        (('test', 'test'), 1),
        (('this', 'this'), 1),
    ])

    assert sorted(b.word_fd.items()) == sorted([('a', 2), ('is', 2), ('test', 2), ('this', 2)])

    assert len(SENT) == sum(b.word_fd.values()) == (sum(b.ngram_fd.values()) + 2 + 1) / 2.0
    assert close_enough(
        sorted(b.score_ngrams(BigramAssocMeasures.pmi)),
        sorted([
            (('a', 'test'), 1.584962500721156),
            (('is', 'a'), 1.584962500721156),
            (('this', 'is'), 1.584962500721156),
            (('a', 'a'), 0.0),
            (('is', 'is'), 0.0),
            (('test', 'test'), 0.0),
            (('this', 'this'), 0.0),
        ])
    )


def test_bigram_custom_span_1():
    pivot_tokens = ['numbers']
    target_tokens = ['calls', 'personal']
    corpus = ['calls', 'to', '0800', 'numbers', 'are', 'free', 'from', 'personal', 'mobiles', 'and', 'landlines']

    b = BigramCollocationFinder.from_words(pivot_tokens, target_tokens, corpus, (3, 4))

    assert sorted(b.ngram_fd.items()) == sorted([
        (('calls', 'numbers'), 1),
        (('numbers', 'personal'), 1),

    ])

def test_bigram_custom_span_2():
    pivot_tokens = ['numbers']
    target_tokens = ['calls', 'personal']
    corpus = ['calls', 'to', '0800', 'numbers', 'are', 'free', 'from', 'from','personal', 'mobiles', 'and', 'landlines']

    b = BigramCollocationFinder.from_words(pivot_tokens, target_tokens, corpus, (3, 4))

    assert sorted(b.ngram_fd.items()) == sorted([
        (('calls', 'numbers'), 1)

    ])

def test_bigram_custom_span_3():
        pivot_tokens = ['test']
        target_tokens = ['yes', 'not']
        corpus = ['calls', 'to', '0800', 'numbers', 'are', 'free', 'from', 'from', 'personal', 'mobiles', 'and',
                  'landlines']

        b = BigramCollocationFinder.from_words(pivot_tokens, target_tokens, corpus, (3, 4))

        assert sorted(b.ngram_fd.items()) == []

def test_bigram_custom_span_4():
    pivot_tokens = ['numbers']
    target_tokens = ['calls', 'personal']
    corpus = ['calls', 'to', '0800', 'numbers', 'are', 'free', 'from', 'from','personal', 'mobiles', 'and', 'landlines']

    b = BigramCollocationFinder.from_words(pivot_tokens, target_tokens, corpus, (3, 3))

    assert sorted(b.ngram_fd.items()) == sorted([
        (('calls', 'numbers'), 1)

    ])

def test_bigram_custom_span_5():
        pivot_tokens = ['numbers']
        target_tokens = ['calls', 'personal']
        corpus = ['calls', 'to', '0800', 'numbers', 'are', 'free', 'from', 'from', 'personal', 'mobiles', 'and',
                  'landlines']

        b = BigramCollocationFinder.from_words(pivot_tokens, target_tokens, corpus, (1, 1))

        assert sorted(b.ngram_fd.items()) == []