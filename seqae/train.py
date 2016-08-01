# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division

import os
import sys
import argparse

import numpy as np

import chainer
from chainer import Variable, functions as F, cuda, optimizers, serializers

from dataset import Vocab, MinibatchFeeder
from seq2seq import Seq2Seq
from util import maybe_create_dir

def parse_args(args):
    parser = argparse.ArgumentParser(args)
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--train_file', '-t', type=str, required=True,
                        help='Train File (.pack)')
    parser.add_argument('--valid_file', '-v', type=str, required=True,
                        help='Validation File (.pack)')
    parser.add_argument('--test_file', '-T', type=str, required=False, default=None,
                        help='Validation File (.pack)')
    parser.add_argument('--vocab_file', '-V', type=str, required=True,
                        help='Vacab File (.pack)')
    parser.add_argument('--save_dir', '-s', type=str, action='store', default="./save",
                        help='save directory')
    parser.add_argument('--encoding', '-e', type=str, action='store', default='utf-8',
                        help='encoding')
    result = parser.parse_args()

    return result

args = parse_args(sys.argv)

gpu = args.gpu

hidden_size = 256
num_layers = 2
num_transfer_layers = 2

batch_size = 50

save_every_batches = 250000//batch_size # save model, optimizers every this batches
eval_valid_every_batches = 50000//batch_size # evaluate model on valid data every this batches
eval_train_every_batches = 10000//batch_size # evaluate model on train data every this batches
max_epoch = 10000
max_line_length = 100

train_file = args.train_file
valid_file = args.valid_file
test_file = args.test_file
vocab_file = args.vocab_file
save_dir = args.save_dir

encoding = args.encoding

print( "settings:" )
print( "    gpu                     : {}".format(gpu) )
print( "    hidden_size             : {}".format(hidden_size) )
print( "    num_layers              : {}".format(num_layers) )
print( "    num_transfer_layers     : {}".format(num_transfer_layers) )
print( "    batch_size              : {}".format(batch_size) )
print( "    save_every_batches      : {}".format(save_every_batches) )
print( "    eval_valid_every_batches: {}".format(eval_valid_every_batches) )
print( "    eval_train_every_batches: {}".format(eval_train_every_batches) )
print( "    max_epoch               : {}".format(max_epoch) )
print( "    max_line_length         : {}".format(max_line_length) )
print( "    train_file              : {}".format(train_file) )
print( "    valid_file              : {}".format(valid_file) )
print( "    test_file               : {}".format(test_file) )
print( "    vocab_file              : {}".format(vocab_file) )
print( "    save_dir                : {}".format(save_dir) )
print( "    encoding                : {}".format(encoding) )
    

if gpu >= 0:
    cuda.get_device(gpu).use()

xp = np if gpu < 0 else cuda.cupy

maybe_create_dir(save_dir)

print(' load vocab from {} ...'.format(vocab_file) )
vocab = Vocab().load_pack(open(vocab_file, 'rb'), encoding=encoding)

vocab_size = len(vocab)
print(' vocab size: {}', format(vocab_size) )

train_batches = MinibatchFeeder(open(train_file, 'rb'), batch_size=batch_size)
train_head_batches = MinibatchFeeder(open(train_file, 'rb'), batch_size=batch_size,
                                     max_line_length=max_line_length)
valid_batches = MinibatchFeeder(open(valid_file, 'rb'), batch_size=batch_size)

print( "train      : {} lines".format(train_batches.num_epoch_lines) )
print( "train(head): {} lines".format(train_head_batches.num_epoch_lines) )
print( "valid      : {} lines".format(valid_batches.num_epoch_lines) )


ignore_label = vocab.padding_id

model = Seq2Seq(vocab_size, hidden_size,
                num_layers=num_layers, 
                num_transfer_layers=num_transfer_layers,
                ignore_label=ignore_label)
if gpu >= 0:
    model.to_gpu()

optimizer = optimizers.Adam() # beta1 = 0.5 may do better 
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(5.))

def forward(model, batch, train=True):

    xp = model.xp
    use_gpu = (xp == cuda.cupy)
    if use_gpu:
        batch = cuda.to_gpu(batch)

    model.reset_state()
    model.zerograds()

    def xcode(f, train):
        loss = 0
        if not train:
            ys, ts = [], []
        last_w = None
        for i in range(len(batch[0])-1):
            w, next_w = Variable(batch[:, i]), Variable(batch[:, i+1])
            y = f(w, train=train)            
            loss += F.softmax_cross_entropy(y, next_w)
            if not train:
                ys.append(xp.argmax(y.data, axis=1))
                ts.append(next_w.data)
            last_w = next_w
        f(next_w, train=train) # process last words

        if train:
            return loss
        else:
            ys = xp.vstack(ys).T
            ts = xp.vstack(ts).T
            if use_gpu:
                ys = cuda.to_cpu(ys)
                ts = cuda.to_cpu(ts)
            return loss, ys, ts
        

    if train:
        encode_loss = xcode(model.encode, train=train)
        model.transfer(train=train)
        decode_loss = xcode(model.decode, train=train)
        return (encode_loss, decode_loss)
    else:
        encode_loss, encode_ys, encode_ts = xcode(model.encode, train=train)
        model.transfer(train=train)
        decode_loss, decode_ys, decode_ts = xcode(model.decode, train=train)
        return ( (encode_loss, encode_ys, encode_ts),
                 (decode_loss, decode_ys, decode_ts) )

def evaluate(model, batches, vocab):

    xp = model.xp
    use_gpu = (xp == cuda.cupy)

    ignore_label = vocab.padding_id

    eloss, eys, ets = 0, [], []
    dloss, dys, dts = 0, [], []    

    sum_max_sentence_length = 0
    
    for batch in batches:
        cur_max_sentence_length = (batch != ignore_label).sum(axis=1)
        sum_max_sentence_length += cur_max_sentence_length
        (cur_eloss, cur_eys, cur_ets), (cur_dloss, cur_dys, cur_dts) = forward(model, batch, train=False)

        cur_eloss.unchain_backward()
        cur_dloss.unchain_backward()        

        eloss += cur_eloss
        dloss += cur_dloss
        eys.extend(cur_eys)
        ets.extend(cur_ets)
        dys.extend(cur_dys)
        dts.extend(cur_dts)               

    eloss /= sum_max_sentence_length
    dloss /= sum_max_sentence_length    

    n = len(encode_ys) // 10 
    if n > 0:
        encode_ys = [eys[i*n] for i in range(10)]
        encode_ts = [ets[i*n] for i in range(10)]
        decode_ys = [dys[i*n] for i in range(10)]
        decode_ts = [dts[i*n] for i in range(10)]

    assert( len(eys) == len(ets) )
    assert( len(eys) == len(dys) )
    assert( len(eys) == len(dts) )

    for i in range(10):
        assert( eys[i].shape == ets[i].shape )
        assert( eys[i].shape == dys[i].shape )
        assert( eys[i].shape == dts[i].shape )
        length = len(ets[i])
        print( "actual:" )
        print( " ".join([vocab.get_word(ets[i][j]).encode('utf-8') for j in range(length)]) )
        print( "encode:" )
        print( " ".join([vocab.get_word(eys[i][j]).encode('utf-8') for j in range(length)]) )
        print( " ".join([[".", "x"][ ets[i][j] != -1 and ets[i][j] != eys[i][j] ] for j in range(length)]) )
        print( "decode:" )
        print( " ".join([vocab.get_word(dys[i][j]).encode('utf-8') for j in range(length)]) )
        print( " ".join([[".", "x"][ dts[i][j] != -1 and dts[i][j] != dys[i][j] ] for j in range(length)]) )
        print()

    print( "encode loss: {}".format( eloss.data ) )
    print( "encode perp: {}".format( math.exp(eloss.data) ) ) 

    print( "decode loss: {}".format( dloss.data ) )
    print( "decode perp: {}".format( math.exp(dloss.data) ) ) 
    
def train(model, batches):

    xp = model.xp
    use_gpu = (xp == cuda.cupy)

    for batch in batches:
        eloss, dloss = forward(model, batch, train=True)
        loss = eloss + dloss
        loss.backward()
        loss.unchain_backward()
        optimizer.update()

next_save_batch = save_every_batches
next_eval_valid_batch = 0 # eval initial model
next_eval_train_batch = 0 # eval initial model
num_saved = 0
num_trained_sentences = 0
num_trained_batches = 0
for epoch in range(max_epoch):

    print( "epoch {}/{}".format( epoch + 1, max_epoch ) )

    for batch in train_batches:

        if num_trained_batches == next_save_batch:
            print( "saving model and optimizer ({}/{}) ...".format(num_trained_batches, train_batches.num_epoch_batches ) )
            prefix = '{}_{}_{}'.format(epoch+1, num_saved+1, num_trained_sents)

            model_file = os.path.join(save_dir, prefix + '.model.hdf5')
            print( "save model to {} ...".format(model_file) )
            save_hdf5(model_file, model)

            optimizer_file = os.path.join(save_dir, prefix + '.optimizer.hdf5')
            print( "save optimizer to {} ...".format(optimizer_file) )
            save_hdf5(optimizer_file, optimizer)

            next_save_batch += save_every_batches
            num_saved += 1
            
        if num_trained_batches == next_eval_valid_batch:

            print( "eval on validation dataset ({}/{}) ...".format(num_trained_batches, train_batches.num_epoch_batches ) )
            evaluate(model, valid_batches, vocab)
            print()

            next_eval_valid_batch += eval_valid_every_batches

        if num_trained_batches == next_eval_train_batch:

            print( "eval on training dataset ({}/{}) ...".format(num_trained_batches, train_batches.num_epoch_batches ) )
            evaluate(model, train_head_batches, vocab)
            print()

            next_eval_train_batch += eval_train_every_batches

        train(model)

        num_trained_batches += 1
        num_trained_sents += len(batch.data)

        
print( "saving model and optimizer (last) ...".format(num_trained_batches, train_batches.num_epoch_batches ) )

model_file = os.path.join(save_dir, 'model.hdf5')
print( "save model to {} ...".format(model_file) )
save_hdf5(model_file, model)

optimizer_file = os.path.join(save_dir, 'optimizer.hdf5')
print( "save optimizer to {} ...".format(optimizer_file) )
save_hdf5(optimizer_file, optimizer)

print( "eval on validation dataset ({}/{}) ...".format(num_trained_batches, train_batches.num_epoch_batches ) )
evaluate(model, valid_batches, vocab)
print()

print( "eval on training dataset ({}/{}) ...".format(num_trained_batches, train_batches.num_epoch_batches ) )
evaluate(model, train_head_batches, vocab)
print()        
