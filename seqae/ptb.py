# -*- coding:utf-8 -*-

from util import maybe_download

def prepare_ptb( files = {'train': './ptb/ptb.train.txt',
                          'test': './ptb/ptb.test.txt',
                          'valid': './ptb/ptb.valid_txt'} ):
    train_url = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt'
    valid_url = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt'
    test_url = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt'
    train_txt = maybe_download(files['train'], train_url)
    test_txt = maybe_download(files['test'], test_url)
    valid_txt = maybe_download(files['valid'], valid_url)

            
    
