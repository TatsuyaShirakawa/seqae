# -*- coding:utf-8 -*-

from __future__ import print_function, division

import collections
import sys

import nltk
from nltk.tokenize import word_tokenize

import msgpack

def sepline(line):
    return word_tokenize(line.strip().lower())

class Vocab(object):

    padding_word = '<sos>'
    sos_word = '<sos>'
    eos_word = '<sos>'
        

    def __init__(self, padding='<sos>', sos='<sos>', eos='<eos>', unk='<unk>'):

        self.__padding_id = -1
        self.__padding_word = padding

        self.__sos_word = sos
        self.__eos_word = eos
        self.__unk_word = unk
        
        self._init()

        
    def _init(self):
        self.__word2id = {}
        self.__id2word = {}
        self.__id2count = {}
        self.__set(id=0, word='<sos>', count=0)
        self.__set(id=1, word='<eos>', count=0)
        self.__set(id=2, word='<unk>', count=0)                
        

    def __len__(self):
        return len(self.__word2id)

    def entry(self, word, count=1):
        id = self.__word2id.get(word, len(self.__word2id))
        self.__set(id=id, word=word, count=self.__id2count.get(id, 0) + count)

    def __set(self, id, word, count):
        self.__word2id[word] = id
        self.__id2word[id] = word
        self.__id2count[id] = count

    @property
    def padding_word(self):
        return self.__padding_word

    @property
    def padding_id(self):
        return self.__padding_id

    @property
    def sos_word(self):
        return self.__sos_word

    @property
    def sos_id(self):
        return self.get_id(self.sos_word)
    
    @property
    def eos_word(self):
        return self.__eos_word

    @property
    def eos_id(self):
        return self.get_id(self.eos_word)
    
    @property
    def unk_word(self):
        return self.__unk_word

    @property
    def unk_id(self):
        return self.get_id(self.unk_word)
    
    def get_id(self, word):
        if word == self.__padding_word:
            return self.__padding_id
        elif word not in self.__word2id:
            return self.__word2id['<UNK>']
        return self.__word2id[word] if word != self.__padding_word else self.__padding_id

    def get_word(self, id):
        return self.__id2word[id] if id != self.__padding_id else self.__padding_word

    def get_count(self, id):
        assert( id >= 0 and id <= len(self.__id2word) )
        return self.__id2count[id]

    def items(self):
        for id, word in self.__id2word.items():
            yield (id, word, self.__id2count[id])
        raise StopIteration

    def load_pack(self, fin, encoding='utf-8'):
        self._init()
        unpacker = msgpack.Unpacker()
        BUFSIZE = 1024 * 1024
        while True:
            buf = fin.read(BUFSIZE)
            if not buf:
                break
            unpacker.feed(buf)
            for id, word, count in unpacker:
                word = word.decode(encoding)
                self.__set(id=id, word=word, count=count)

    def save_pack(self, fout, encoding='utf-8'):
        packer = msgpack.Packer()
        for id, word, count in sorted(self.items()):
            out.write(packer.pack((id, word.encode(encoding), count)))


def create_vocab(fins, min_count=0, max_vocab=None, sepline=sepline):

    result = Vocab()
    SOS = '<sos>'
    EOS = '<eos>'

    if max_vocab == None:
        max_vocab = float('+inf')

    # entry each words
    num_lines = 0
    for fin in fins:
        for line in fin:
            words = sepline(line)
            # entry <sos> words <eos>
            result.entry(SOS)
            for word in words:
                result.entry(word)
            result.entry(EOS)
            num_lines += 1
            if num_lines % 1000 == 0:
                print( "\rreaded {} lines.".format(num_lines), end="" )
                sys.stdout.flush()
    print( "\rreaded {} lines.".format(num_lines))
    sys.stdout.flush()

    vocab_size_before = len(result)

    print( "reconstructing ...")
    UNK = '<unk>'
    # replace disfrequent words with UNK
    tmp_counts = collections.Counter()
    for id, word, count in result.items():
        if count < min_count:
            word = result.unknown
        tmp_counts[word] += count

    if max_vocab == None:
        counts = tmp_counts
    else:
        counts = collections.Counter()
        # merge disfrequent words (frequency order less than max_vocab) to UNK
        for i, (word, count) in enumerate(sorted(tmp_counts.items(), key=lambda x : x[1], reverse=True)):  
            if i >= max_vocab:
                counts[UNK] += count
            else:
                counts[word] = count

    result._init()
    for i, (word, count) in enumerate(sorted(counts.items(), key=lambda x : x[1], reverse=True)):
        if i >= max_vocab:
            word = result.unknown
        result.entry(word, count=count)

    vocab_size_after = len(result)
    print("reducing vocab {} -> {} (min_count={})".format(vocab_size_before, vocab_size_after, min_count))
    return result

def encode_file_pack(vocab, fin, fout, input_sepline=sepline):

    packer = msgpack.Packer()
    for line in fin:
        words = input_sepline(line)
        encoded = [vocab.sos_id]
        encoded.extend([vocab.get_id(word) for word in words])
        encoded = [vocab.eos_id]
        fout.write(packer.pack(encoded))

        
class MinibatchFeeder(object):

    def __init__(self, fin, batch_size, sepline = sepline, 
                 max_num_lines=None, max_line_length=100,
                 feed_callback=lambda x:x):
        
        self.fin = fin
        self.batch_size = batch_size
        self.sepline = sepline
        self.max_num_lines = max_num_lines
        self.max_line_length = max_line_length
        self.callback = callback

        self.batch = []

        self.num_epochs = 0
        self.num_batches = 0
        self.num_lines = 0

        
        self.num_epoch_lines = 0
        self.__set_num_epoch_lines()

    def __set_num_epoch_lines(self):
        self.num_epoch_lines = 0
        unpacker = msgpack.Unpacker()
        self.fin.seek(0)

        BUFSIZE = 1024 * 1024
            
        while True:
            buf = self.fin.read(BUFSIZE)
            if not buf:
                break
            unpacker.feed(buf)
            for words in unpacker:
                if self.max_num_lines != None and self.num_epoch_lines == self.max_num_lines:
                    self.fin.seek(0)
                    return
                if len(words) > self.max_line_length:
                    continue
                self.num_epoch_lines += 1
        self.fin.seek(0)
                
    @property
    def num_epoch_batches(self):
        return self.num_epoch_lines // self.batch_size

    def __iter__(self):
        return self.__next__()

    def __next__(self):
        return self.__next_pack()

    def __next_pack(self):
        self.batch = []
        unpacker = msgpack.Unpacker()
        BUFSIZE = 1024 * 1024
        num_lines = 0
        while True:
            buf = self.fin.read(BUFSIZE)
            if not buf:
                break
            unpacker.feed(buf)
            for words in unpacker:
                num_lines += 1
                if self.max_num_lines != None and num_lines == self.max_num_lines:
                    break
                if len(words) > self.max_line_length:
                    continue
                self.batch.append(words)
                self.num_lines += 1
                if len(self.batch) == self.batch_size:
                    self.num_batches += 1
                    yield self.feed_callback(self.batch)
                    self.batch = []
            if self.max_num_lines != None and num_lines == self.max_num_lines:
                break
        self.num_epochs += 1
        self.fin.seek(0)
        raise StopIteration
