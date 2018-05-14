# -*- coding:utf8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
from tensorflow.contrib import learn  # pylint: disable=g-bad-import-order
import jieba

jieba.load_userdict("../dict/useWords.txt")
# TOKENIZER_RE = re.compile(u'([\u4e00-\u9fa5]+)', re.UNICODE)
TOKENIZER_RE = re.compile(u'^[\u4e00-\u9fa5_a-zA-Z0-9]+$', re.UNICODE)  # 保留汉字英文和数字，没有保留问号


def tokenizer_char(iterator):
    for value in iterator:
        yield list(value)


def tokenizer_word(iterator):
    for value in iterator:
        yield TOKENIZER_RE.findall(value)


def tockenizer_chword(iterator):
    for value in iterator:
        words = [w for w in jieba.cut(value) if w.strip()]
        # words1 = [TOKENIZER_RE.findall(w)[0] for w in words if TOKENIZER_RE.findall(w)]
        yield [TOKENIZER_RE.findall(w)[0] for w in words if TOKENIZER_RE.findall(w)]


class MyVocabularyProcessor(learn.preprocessing.VocabularyProcessor):
    def __init__(self,
                 max_document_length,
                 min_frequency=0,
                 vocabulary=None):
        tokenizer_fn = tockenizer_chword
        self.sup = super(MyVocabularyProcessor, self)
        self.sup.__init__(max_document_length, min_frequency, vocabulary, tokenizer_fn)

    def transform(self, raw_documents):
        """Transform documents to word-id matrix.
        Convert words to ids with vocabulary fitted with fit or the one
        provided in the constructor.
        Args:
          raw_documents: An iterable which yield either str or unicode.
        Yields:
          x: iterable, [n_samples, max_document_length]. Word-id matrix.
        """
        for tokens in self._tokenizer(raw_documents):
            word_ids = np.zeros(self.max_document_length, np.int64)
            for idx, token in enumerate(tokens):
                if idx >= self.max_document_length:
                    break
                word_ids[idx] = self.vocabulary_.get(token)
            yield word_ids
