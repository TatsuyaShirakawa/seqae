# -*- coding:utf-8 -*-

import numpy as np

import chainer
from chainer import Variable, Chain, functions as F, links as L
from chainer.utils import array

class RNNLM(Chain):

    def __init__(self, vocab_size, hidden_size, num_layers, ignore_label=-1):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ignore_label = ignore_label

        args = {'embed': L.EmbedID(vocab_size, hidden_size, ignore_label=ignore_label),
                'hy': L.Linear(hidden_size, vocab_size),
        }
        args.update({'_xh{}'.format(i): L.Linear(hidden_size, 4*hidden_size) for i in range(self.num_layers)})
        args.update({'_hh{}'.format(i): L.Linear(hidden_size, 4*hidden_size) for i in range(self.num_layers)})

        super(RNNLM, self).__init__(**args)
        
        self.hs = [None for i in range(self.num_layers)]
        self.cs = [None for i in range(self.num_layers)]

        self.xhs = [getattr(self, '_xh{}'.format(i)) for i in range(self.num_layers)]
        self.hhs = [getattr(self, '_hh{}'.format(i)) for i in range(self.num_layers)]

        self.reset_state()

    def to_cpu(self):
        super(RNNLM, self).to_cpu()
        for i in range(self.num_layers):
            if self.hs[i] is not None: self.hs[i].to_cpu()
            if self.cs[i] is not None: self.cs[i].to_cpu()

    def to_gpu(self):
        super(RNNLM, self).to_gpu()
        for i in range(self.num_layers):
            if self.hs[i] is not None: self.hs[i].to_gpu()
            if self.cs[i] is not None: self.cs[i].to_gpu()

    def reset_state(self):
        for i in range(self.num_layers):
            self.hs[i] = None
            self.cs[i] = None

    def maybe_init_state(self, batch_size, dtype):
        
        for i in range(self.num_layers):
            if self.hs[i] is None:
                xp = self.xp
                self.hs[i] = Variable(xp.zeros((batch_size, self.hidden_size), dtype=dtype),
                                      volatile='auto')
            if self.cs[i] is None:
                xp = self.xp
                self.cs[i] = Variable(xp.zeros((batch_size, self.hidden_size), dtype=dtype),
                                      volatile='auto')

    def __call__(self, w, train=True, dpratio=0.2):

        x = self.embed(w)

        self.maybe_init_state(len(x.data), x.data.dtype)

        for i in range(self.num_layers):

            c = F.dropout(self.cs[i], train=train, ratio=dpratio)
            h = self.xhs[i](F.dropout(x, train=train, ratio=dpratio))
            + self.hhs[i](F.dropout(self.hs[i], train=train, ratio=dpratio))

            assert( c.data.shape == (len(x.data), self.hidden_size) )
            assert( h.data.shape == (len(x.data), 4*self.hidden_size) )

            c, h = F.lstm(c, h)

            assert( c.data.shape == (len(x.data), self.hidden_size) )
            assert( h.data.shape == (len(x.data), self.hidden_size) )

            if self.ignore_label != None:
                xp = self.xp
                enable = (x.data != 0)
                self.cs[i] = F.where(enable, c , self.cs[i])
                self.hs[i] = F.where(enable, h , self.hs[i])
            else:
                self.cs[i] = c
                self.hs[i] = h
            x = h
            
        return self.hy(self.hs[-1])

            
class Seq2Seq(Chain):

    def __init__(self, vocab_size, hidden_size, num_layers=1, num_transfer_layers=1, 
                 ignore_label=-1):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_transfer_layers = num_transfer_layers
        self.ignore_label = ignore_label


        args = {'encoder': RNNLM(vocab_size, hidden_size, num_layers=num_layers, ignore_label=ignore_label),
                'decoder': RNNLM(vocab_size, hidden_size, num_layers=num_layers, ignore_label=ignore_label),
            }

        for i in range(num_layers):
            args.update({'_hin{}'.format(i) : L.Linear(hidden_size, 2*hidden_size),
                         '_cin{}'.format(i) : L.Linear(hidden_size, 2*hidden_size),
                         '_hout{}'.format(i) : L.Linear(2*hidden_size, hidden_size),
                         '_cout{}'.format(i) : L.Linear(2*hidden_size, hidden_size),
                     })
            for j in range(num_transfer_layers):
                args.update({'l{}_{}'.format(i, j) : L.Linear(2*hidden_size, 2*hidden_size)})
            
        super(Seq2Seq, self).__init__(**args)
        self.reset_state()

        self.hins = [getattr(self, '_hin{}'.format(i)) for i in range(num_layers)]
        self.cins = [getattr(self, '_cin{}'.format(i)) for i in range(num_layers)]
        self.houts = [getattr(self, '_hout{}'.format(i)) for i in range(num_layers)]
        self.couts = [getattr(self, '_cout{}'.format(i)) for i in range(num_layers)]
        self.ls = []
        for i in range(num_layers):
            cur_ls = []
            for j in range(num_transfer_layers):
                cur_ls.append(getattr(self, 'l{}_{}'.format(i, j)))
            self.ls.append(cur_ls)
                    
        
    def to_cpu(self):
        super(Seq2Seq, self).to_cpu()
        self.encoder.to_cpu()
        self.decoder.to_cpu()

    def to_gpu(self):
        super(Seq2Seq, self).to_gpu()
        self.encoder.to_gpu()
        self.decoder.to_gpu()
        
    def reset_state(self):
        self.encoder.reset_state()
        self.decoder.reset_state()

    def transfer(self, train=True):

        for i in range(self.num_layers):
            
            h0 = self.hins[i](self.encoder.hs[i])
            c0 = self.cins[i](self.encoder.cs[i])
            
            h = F.relu(h0 + c0)
            for j in range(self.num_transfer_layers):
                h = F.relu(self.ls[i][j](h))

            self.decoder.hs[i] = self.houts[i](h)
            self.decoder.cs[i] = self.couts[i](h)

    def encode(self, w, train=True):
        return self.encoder(w, train=train)

    def decode(self, w, train=True):
        return self.decoder(w, train=train)

