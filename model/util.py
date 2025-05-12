# Import libraries
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import math
import keras
import keras_nlp
from keras import ops
import tensorflow as tf

import json
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

import plotly.graph_objs as go
import plotly.express as px

# Configuration
class CFG:
    seed = 42
    preset = "deberta_v3_large_en" # name of pretrained backbone
    train_seq_len = 1024 # max size of input sequence for training
    train_batch_size = 2 * 8 # size of the input batch in training, x 2 as two GPUs
    infer_seq_len = 2000 # max size of input sequence for inference
    infer_batch_size = 2 * 2 # size of the input batch in inference, x 2 as two GPUs
    epochs = 6 # number of epochs to train
    lr_mode = "exp" # lr scheduler mode from one of "cos", "step", "exp"
    
    labels = ["B-EMAIL", "B-ID_NUM", "B-NAME_STUDENT", "B-PHONE_NUM",
              "B-STREET_ADDRESS", "B-URL_PERSONAL", "B-USERNAME",
              "I-ID_NUM", "I-NAME_STUDENT", "I-PHONE_NUM",
              "I-STREET_ADDRESS","I-URL_PERSONAL","O"]
    id2label = dict(enumerate(labels)) # integer label to BIO format label mapping
    label2id = {v:k for k,v in id2label.items()} # BIO format label to integer label mapping
    num_labels = len(labels) # number of PII (NER) tags
    
    train = True # whether to train or use already trained model

# Loss: CrossEntropy
class CrossEntropy(keras.losses.SparseCategoricalCrossentropy):
    def __init__(self, ignore_class=-100, reduction=None, **args):
        super().__init__(reduction=reduction, **args)
        self.ignore_class = ignore_class

    def call(self, y_true, y_pred):
        y_true = ops.reshape(y_true, [-1])
        y_pred = ops.reshape(y_pred, [-1, CFG.num_labels])
        loss = super().call(y_true, y_pred)
        if self.ignore_class is not None:
            valid_mask = ops.not_equal(
                y_true, ops.cast(self.ignore_class, y_pred.dtype)
            )
            loss = ops.where(valid_mask, loss, 0.0)
            loss = ops.sum(loss)
            loss /= ops.maximum(ops.sum(ops.cast(valid_mask, loss.dtype)), 1)
        else:
            loss = ops.mean(loss)
        return loss

# Metric: FBetaScore
class FBetaScore(keras.metrics.FBetaScore):
    def __init__(self, ignore_classes=[-100, 12], average="micro", beta=5.0,
                 name="f5_score", **args):
        super().__init__(beta=beta, average=average, name=name, **args)
        self.ignore_classes = ignore_classes or []

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.convert_to_tensor(y_true, dtype=self.dtype)
        y_pred = ops.convert_to_tensor(y_pred, dtype=self.dtype)
        
        y_true = ops.reshape(y_true, [-1])
        y_pred = ops.reshape(y_pred, [-1, CFG.num_labels])
            
        valid_mask = ops.ones_like(y_true, dtype=self.dtype)
        if self.ignore_classes:
            for ignore_class in self.ignore_classes:
                valid_mask &= ops.not_equal(y_true, ops.cast(ignore_class, y_pred.dtype))
        valid_mask = ops.expand_dims(valid_mask, axis=-1)
        
        y_true = ops.one_hot(y_true, CFG.num_labels)
        
        if not self._built:
            self._build(y_true.shape, y_pred.shape)

        threshold = ops.max(y_pred, axis=-1, keepdims=True)
        y_pred = ops.logical_and(
            y_pred >= threshold, ops.abs(y_pred) > 1e-9
        )

        y_pred = ops.cast(y_pred, dtype=self.dtype)
        y_true = ops.cast(y_true, dtype=self.dtype)
        
        tp = ops.sum(y_pred * y_true * valid_mask, self.axis)
        fp = ops.sum(y_pred * (1 - y_true) * valid_mask, self.axis)
        fn = ops.sum((1 - y_pred) * y_true * valid_mask, self.axis)
            
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

# Data Processing
def get_tokens(words, seq_len, packer):
    # Tokenize input
    token_words = tf.expand_dims(
        tokenizer(words), axis=-1
    )  # ex: (words) ["It's", "a", "cat"] ->  (token_words) [[1, 2], [3], [4]]
    tokens = tf.reshape(
        token_words, [-1]
    )  # ex: (token_words) [[1, 2], [3], [4]] -> (tokens) [1, 2, 3, 4]
    # Pad tokens
    tokens = packer(tokens)[0][:seq_len]
    inputs = {"token_ids": tokens, "padding_mask": tokens != 0}
    return inputs, tokens, token_words


def get_token_ids(token_words):
    # Get word indices
    word_ids = tf.range(tf.shape(token_words)[0])
    # Get size of each word
    word_size = tf.reshape(tf.map_fn(lambda word: tf.shape(word)[0:1], token_words), [-1])
    # Repeat word_id with size of word to get token_id
    token_ids = tf.repeat(word_ids, word_size)
    return token_ids


def get_token_labels(word_labels, token_ids, seq_len):
    # Create token_labels from word_labels ->  alignment
    token_labels = tf.gather(word_labels, token_ids)
    # Only label the first token of a given word and assign -100 to others
    mask = tf.concat([[True], token_ids[1:] != token_ids[:-1]], axis=0)
    token_labels = tf.where(mask, token_labels, -100)
    # Truncate to max sequence length
    token_labels = token_labels[: seq_len - 2]  # -2 for special tokens ([CLS], [SEP])
    # Pad token_labels to align with tokens (use -100 to pad for loss/metric ignore)
    pad_start = 1  # for [CLS] token
    pad_end = seq_len - tf.shape(token_labels)[0] - 1  # for [SEP] and [PAD] tokens
    token_labels = tf.pad(token_labels, [[pad_start, pad_end]], constant_values=-100)
    return token_labels


def process_token_ids(token_ids, seq_len):
    # Truncate to max sequence length
    token_ids = token_ids[: seq_len - 2]  # -2 for special tokens ([CLS], [SEP])
    # Pad token_ids to align with tokens (use -1 to pad for later identification)
    pad_start = 1  # [CLS] token
    pad_end = seq_len - tf.shape(token_ids)[0] - 1  # [SEP] and [PAD] tokens
    token_ids = tf.pad(token_ids, [[pad_start, pad_end]], constant_values=-1)
    return token_ids


def process_data(seq_len=720, has_label=True, return_ids=False):
    # To add spetical tokens: [CLS], [SEP], [PAD]
    packer = keras_nlp.layers.MultiSegmentPacker(
        start_value=tokenizer.cls_token_id,
        end_value=tokenizer.sep_token_id,
        sequence_length=seq_len,
    )

    def process(x):
        # Generate inputs from tokens
        inputs, tokens, words_int = get_tokens(x["words"], seq_len, packer)
        # Generate token_ids for maping tokens to words
        token_ids = get_token_ids(words_int)
        if has_label:
            # Generate token_labels from word_labels
            token_labels = get_token_labels(x["labels"], token_ids, seq_len)
            return inputs, token_labels
        elif return_ids:
            # Pad token_ids to align with tokens
            token_ids = process_token_ids(token_ids, seq_len)
            return token_ids
        else:
            return inputs

    return process

def build_dataset(words, labels=None, return_ids=False, batch_size=4,
                  seq_len=512, shuffle=False, cache=True, drop_remainder=True):
    AUTO = tf.data.AUTOTUNE 

    slices = {"words": tf.ragged.constant(words)}
    if labels is not None:
        slices.update({"labels": tf.ragged.constant(labels)})

    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.map(process_data(seq_len=seq_len,
                             has_label=labels is not None, 
                             return_ids=return_ids), num_parallel_calls=AUTO) # apply processing
    ds = ds.cache() if cache else ds  # cache dataset
    if shuffle: # shuffle dataset
        ds = ds.shuffle(1024, seed=CFG.seed)  
        opt = tf.data.Options() 
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)  # batch dataset
    ds = ds.prefetch(AUTO)  # prefetch next batch
    return ds

# Learning rate Schedule
def get_lr_callback(batch_size=8, mode='cos', epochs=10, plot=False):
    lr_start, lr_max, lr_min = 6e-6, 2.5e-6 * batch_size, 1e-6
    lr_ramp_ep, lr_sus_ep, lr_decay = 3, 0, 0.75

    def lrfn(epoch):  # Learning rate update function
        if epoch < lr_ramp_ep: lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep: lr = lr_max
        elif mode == 'exp': lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        elif mode == 'step': lr = lr_max * lr_decay**((epoch - lr_ramp_ep - lr_sus_ep) // 2)
        elif mode == 'cos':
            decay_total_epochs, decay_epoch_index = epochs - lr_ramp_ep - lr_sus_ep + 3, epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        return lr

    if plot:  # Plot lr curve if plot is True
        fig = px.line(x=np.arange(epochs),
                      y=[lrfn(epoch) for epoch in np.arange(epochs)], 
                      title='LR Scheduler',
                      markers=True,
                      labels={'x': 'epoch', 'y': 'lr'})
        fig.update_layout(
            yaxis = dict(
                showexponent = 'all',
                exponentformat = 'e'
            )
        )
        fig.show()

    return keras.callbacks.LearningRateScheduler(lrfn, verbose=False)  # Create lr callback
