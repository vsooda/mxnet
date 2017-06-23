#!/usr/bin/env python2.7
#coding=utf-8

import mxnet as mx

data_scp = "/home/sooda/speech/merlin/egs/world/s1/experiments/nana/duration_model/gen_data/lab.scp";
label_scp =  "/home/sooda/speech/merlin/egs/world/s1/experiments/nana/duration_model/gen_data/cmp.scp";
data_iter = mx.io.TTSIter(data_scp=data_scp,
			  label_scp=label_scp,
			  data_shape=(546,),
			  label_shape=(5,),
			  batch_size=32,
			  round_batch=False)

batchidx = 0
for dbatch in data_iter:
    data = dbatch.data[0]
    label = dbatch.label[0]
    print("Batch", batchidx, label.shape)
    print(label.asnumpy())
    batchidx += 1

