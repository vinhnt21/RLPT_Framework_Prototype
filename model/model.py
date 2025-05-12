# Import libraries
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

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
import util 


keras.utils.set_random_seed(util.CFG.seed)

# Get devices default "gpu" or "tpu"
devices = keras.distribution.list_devices()
print("Device:", devices)

if len(devices) > 1:
    # Data parallelism
    data_parallel = keras.distribution.DataParallel(devices=devices)

    # Set the global distribution.
    keras.distribution.set_distribution(data_parallel)

keras.mixed_precision.set_global_policy("mixed_float16")

BASE_PATH = "/input/data"

# Train-Valid data
data = json.load(open(f"{BASE_PATH}/train.json"))

# Initialize empty arrays
words = np.empty(len(data), dtype=object)
labels = np.empty(len(data), dtype=object)

# Fill the arrays
for i, x in tqdm(enumerate(data), total=len(data)):
    words[i] = np.array(x["tokens"])
    labels[i] = np.array([util.CFG.label2id[label] for label in x["labels"]])

# Splitting the data into training and testing sets
train_words, valid_words, train_labels, valid_labels = train_test_split(
    words, labels, test_size=0.2, random_state=util.CFG.seed
)

# To convert string input or list of strings input to numerical tokens
tokenizer = keras_nlp.models.DebertaV3Tokenizer.from_preset(
    util.CFG.preset,
)

# Preprocessing layer to add spetical tokens: [CLS], [SEP], [PAD]
packer = keras_nlp.layers.MultiSegmentPacker(
    start_value=tokenizer.cls_token_id,
    end_value=tokenizer.sep_token_id,
    sequence_length=10,
)

# Build Train & Valid Dataloader
train_ds = util.build_dataset(train_words, train_labels,  batch_size=util.CFG.train_batch_size,
                         seq_len=util.CFG.train_seq_len, shuffle=True)

valid_ds = util.build_dataset(valid_words, valid_labels, batch_size=util.CFG.train_batch_size, 
                         seq_len=util.CFG.train_seq_len, shuffle=False)


# Modeling
# Build Token Classification model
backbone = keras_nlp.models.DebertaV3Backbone.from_preset(
    util.CFG.preset,
)
out = backbone.output
out = keras.layers.Dense(util.CFG.num_labels, name="logits")(out)
out = keras.layers.Activation("softmax", dtype="float32", name="prediction")(out)
model = keras.models.Model(backbone.input, out)

# Compile model for optimizer, loss and metric
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=2e-5),
    loss=util.CrossEntropy(),
    metrics=[util.FBetaScore()],
)

# Summary of the model architecture
model.summary()

lr_cb = util.get_lr_callback(util.CFG.train_batch_size, mode=util.CFG.lr_mode, plot=True)

# Train the model
if util.CFG.train:
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=CFG.epochs,
        callbacks=[lr_cb],
        verbose=1,
    )
else:
    model.load_weights("/input/save_model/model.weights.h5")

model.save_weights("/input/save_model/model.weights.h5")

# Evaluation

# Build Test Dataloader
# Build Validation dataloader with "infer_seq_len"
valid_ds = util.build_dataset(valid_words, valid_labels, return_ids=False, batch_size=util.CFG.infer_batch_size,
                        seq_len=util.CFG.infer_seq_len, shuffle=False, cache=False)
# Evaluate
model.evaluate(valid_ds, return_dict=True, verbose=0)

# Inference
# Test data
test_data = json.load(open(f"{BASE_PATH}/test.json"))

# Ensure number of samples is divisble by number of devices
need_samples  = len(devices) - len(test_data) % len(devices)
for _ in range(need_samples):
    test_data.append(test_data[-1]) # repeat the last sample
    
# Initialize empty arrays
test_words = np.empty(len(test_data), dtype=object)
test_docs = np.empty(len(test_data), dtype=np.int32)

# Fill the arrays
for i, x in tqdm(enumerate(test_data), total=len(test_data)):
    test_words[i] = np.array(x["tokens"])
    test_docs[i] = x["document"]

# Get token ids
id_ds = util.build_dataset(test_words, return_ids=True, batch_size=len(test_words), 
                        seq_len=CFG.infer_seq_len, shuffle=False, cache=False, drop_remainder=False)
test_token_ids = ops.convert_to_numpy([ids for ids in iter(id_ds)][0])

# Build test dataloader
test_ds = build_dataset(test_words, return_ids=False, batch_size=CFG.infer_batch_size,
                        seq_len=CFG.infer_seq_len, shuffle=False, cache=False, drop_remainder=False)

# Do inference
test_preds = model.predict(test_ds, verbose=1)

# Convert probabilities to class labels via max confidence
test_preds = np.argmax(test_preds, axis=-1)
