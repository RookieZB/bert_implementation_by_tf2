# -*- coding: utf-8 -*-

"""Tokenizer for BERT and GPT"""

import re
import regex
import unicodedata
import json
import numpy as np


class Tokenizer:
    def __init__(self, mlm=True, lower=True, num=True):
        self.mlm, self.lower, self.num, self.vocab, self.w, self.e, self.d = mlm, lower, num, None, None, None, None
        self.bos, self.sep, self.pad, self.unk = ['[CLS]', '[SEP]', '[PAD]', '[UNK]'] if mlm else ['<|endoftext|>']*4
        self.pat = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.ch = [[33, 47], [58, 64], [91, 96], [123, 126], [0x4E00, 0x9FFF], [0x3400, 0x4DBF], [0x20000, 0x2A6DF],
                   [0x2A700, 0x2B73F], [0x2B740, 0x2B81F], [0x2B820, 0x2CEAF], [0xF900, 0xFAFF], [0x2F800, 0x2FA1F]]

    def loading(self, path, enc='utf-8'):
        b1 = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
        o1 = [i1 for i1 in range(2**8) if i1 not in b1]
        c1 = [chr(i1) for i1 in b1[:]+[2**8+j1 for j1 in range(len(o1))]]
        self.vocab = dict([(j1.strip(), i1) for i1, j1 in enumerate(open(path, encoding=enc))]) if self.mlm else {}
        self.vocab = self.vocab if self.mlm else json.load(open(path, encoding=enc))
        self.w, self.e, self.d = list(self.vocab.keys()), dict(zip(b1+o1, c1)), dict(zip(c1, b1+o1))

    def splitting(self, token):
        toke1, toke2 = [], (re.findall(r'Ġ\d|\d|\D+', token) if self.num else [token])

        for i1, char1 in enumerate(toke2):
            star1, endi1 = 0, 0

            while star1 < len(char1):
                for endi1 in range(len(char1), star1, -1):
                    subt1 = ('##' if self.mlm and (i1 > 0 or star1 > 0) else '')+char1[star1:endi1]

                    if subt1 in self.vocab:
                        toke1, star1 = toke1+[subt1], endi1
                        break

                if star1 != endi1:
                    return [self.unk]

        return toke1

    def separating(self, text, pre):
        if self.mlm:
            func1 = (lambda x1: any(i1[0] <= ord(x1) <= i1[1] for i1 in self.ch))
            orig1 = [' '+c1+' ' if unicodedata.category(c1).startswith('P') or func1(c1) else c1 for c1 in text]
            text1 = re.sub(r'\[ mask ]|\[ MASK ]', '[MASK]', ''.join(orig1)).strip().split()
        else:
            orig1 = regex.findall(self.pat, (' ' if pre and text[0] != ' ' else '')+text)
            text1 = [''.join([self.e[j1] for j1 in i1.encode('utf-8')]) for i1 in orig1]

        text2 = sum([self.splitting(t1) for t1 in text1], [])
        return text2, len(text2)

    def processing(self, a, b, maxlen, pre):
        a1, l1 = self.separating(a.lower() if self.lower else a, pre)
        b1, l2 = self.separating(b.lower() if self.lower else b, pre) if b else ([], 0)
        a2 = a1[:min(l1, int(np.ceil(maxlen/2)))] if l1 < l2 else a1[:max(maxlen-l2, int(np.ceil(maxlen/2)))]
        return a2, b1[:min(l2, maxlen//2)] if l2 < l1 else b1[:max(maxlen-l1, maxlen//2)]

    def encoding(self, a, b=None, maxlen=64, bos=True, sep=True, pad=True, pre=True, length=False):
        a1, b1 = self.processing(a, b, maxlen-bos-sep-(b is not None and sep), pre)
        a1, b1 = [self.bos]*bos+a1+[self.sep]*sep, b1+[self.sep]*(b is not None and sep)
        l1, l2 = len(a1), len(b1)
        padd1 = maxlen-l1-l2 if pad else 0
        sent1 = [self.vocab.get(i1, self.vocab[self.unk]) for i1 in (a1+b1+[self.pad]*padd1)]
        segm1 = [0]*l1+[1]*l2+[0 if not b else 1]*padd1
        mask1 = [0]*(l1+l2)+[1]*padd1
        return (sent1, segm1, mask1, l1 if b is None else (l1, l2)) if length else (sent1, segm1, mask1)

    def decoding(self, token):
        toke1 = [j1 for j1 in [self.w[i1] for i1 in token] if j1 not in [self.bos, self.sep, self.pad]]
        func1 = (lambda x1: re.sub(r' ##', '', x1))
        func2 = (lambda x1: bytearray([self.d[i1] for i1 in x1]).decode('utf-8', errors='replace'))
        return func1(' '.join(toke1)) if self.mlm else func2(''.join(toke1))
