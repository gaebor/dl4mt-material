import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import ipdb
import numpy
import copy

import os
import warnings
import sys
import time

from collections import OrderedDict

from data_iterator import TextIterator

from lm import *

profile = False

def babble(saveto='model.npz'):
    with open('%s.pkl' % saveto, 'rb') as f:
        model_options = pkl.load(f)

    # load dictionary
    with open(model_options["dictionary"], 'rb') as f:
        worddicts = pkl.load(f)

    # invert dictionary
    worddicts_r = dict()
    worddicts_i = dict()
    for kk, vv in worddicts.iteritems():
        worddicts_r[vv] = kk
        worddicts_i[kk] = vv

    print >>sys.stderr, 'Building model'
    params = init_params(model_options)
    params = load_params("%s.npz" % saveto, params)

    # create shared variables for parameters
    tparams = init_tparams(params)

    # build the symbolic computational graph
    trng, use_noise, \
        x, x_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, model_options)
    inps = [x, x_mask]

    print >>sys.stderr, 'Buliding sampler'
    f_next = build_sampler(tparams, model_options, trng)
    
    for line in sys.stdin:
        sample = [worddicts_i[kk] for kk in line.strip().split() if kk in worddicts_i]
        sample = gen_sample(f_next, model_options, sample)
        ss = sample
        for vv in ss:
            if vv == 0:
                break
            if vv in worddicts_r:
                print worddicts_r[vv],
            else:
                print 'UNK',
        print

if __name__ == '__main__':
    babble(sys.argv[1])
