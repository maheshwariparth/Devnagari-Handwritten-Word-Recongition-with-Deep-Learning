from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Conv2D
tf.disable_v2_behavior()

class DecoderType:
  BestPath = 0
  BeamSearch = 1
  WordBeamSearch = 2


class Model: 
  "minimalistic TF model for HTR - Support Devnagari Language"

  # model constants
  batchSize = 25
  imgSize = (128, 32)
  maxTextLen = 32

  def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False):
    "init model: add CNN, RNN and CTC and initialize TF"
    self.charList = charList
    self.decoderType = decoderType
    self.mustRestore = mustRestore
    self.snapID = 0

    # input image batch
    self.inputImgs = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))

    # setup CNN, RNN and CTC
    self.setupCNN()
    self.setupRNN()
    self.setupCTC()

    # setup optimizer to train NN
    self.batchesTrained = 0
    self.learningRate = tf.placeholder(tf.float32, shape=[])
    self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

    # initialize TF
    (self.sess, self.saver) = self.setupTF()

  def setupCNN(self):
    "create CNN layers and return output of these layers"
    cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)

    # list of parameters for the layers
    kernelVals = [5, 5, 3, 3, 3]
    featureVals = [1, 32, 64, 128, 128, 256]
    strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
    numLayers = len(strideVals)

    # create layers
    pool = cnnIn4d # input to first CNN layer
    for i in range(numLayers):
      kernel = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
      conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1))
      relu = tf.nn.relu(conv)
      pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')

    self.cnnOut4d = pool
    print("CNN output shape:",pool.get_shape())

  def setupRNN(self):
    "create RNN layers and return output of these layers"
    rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])

    # Define the number of hidden units and output classes
    numHidden = 256
    charList_len = len(self.charList) + 1  # for characters + blank class

    # Define a Bidirectional LSTM layer with stacking (2 layers)
    lstm_fw = LSTM(numHidden, return_sequences=True)
    lstm_fw2 = LSTM(numHidden, return_sequences=True, go_backwards=False)
    lstm_bw = LSTM(numHidden, return_sequences=True, go_backwards=True)
    lstm_bw2 = LSTM(numHidden, return_sequences=True)
    bi_lstm = Bidirectional(lstm_fw2, backward_layer=lstm_bw)

    # Apply the bidirectional LSTM to the input
    rnn_output = lstm_bw2(bi_lstm(lstm_fw(rnnIn3d)))  # Output shape: BxTx2H (2H for concatenated forward and backward states)

    # Expand the dimensions to match the original output shape
    concat = tf.expand_dims(rnn_output, axis=2)  # Shape: BxTx1x2H

    # Define a Conv2D kernel to project the output to character classes
    conv_kernel = Conv2D(
        filters=charList_len,
        kernel_size=(1, 1),
        dilation_rate=1,
        padding='same',
        use_bias=False,
        kernel_initializer=tf.initializers.truncated_normal(stddev=0.1)
    )

    # Apply Conv2D to project output to character classes
    conv_output = conv_kernel(concat)  # Shape: BxTx1xC

    # Remove the dimension with size 1 to get the final shape BxTxC
    self.rnnOut3d = tf.squeeze(conv_output, axis=[2])

    # Print the shape of the final RNN output
    print("RNN_OUT Shape:", self.rnnOut3d.get_shape())

  def setupCTC(self):
    "create CTC loss and decoder and return them"
    # BxTxC -> TxBxC
    self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])
    # ground truth text as sparse tensor
    self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))

    # calc loss for batch
    self.seqLen = tf.placeholder(tf.int32, [None])
    self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True))

    # calc loss for each element to compute label probability
    self.savedCtcInput = tf.placeholder(tf.float32, shape=[Model.maxTextLen, None, len(self.charList) + 1])
    self.lossPerElement = tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput, sequence_length=self.seqLen, ctc_merge_repeated=True)

    # decoder: either best path decoding or beam search decoding
    if self.decoderType == DecoderType.BestPath:
      self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
    elif self.decoderType == DecoderType.BeamSearch:
      self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50, merge_repeated=False)
    elif self.decoderType == DecoderType.WordBeamSearch:
      # import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
      word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')

      # prepare information about language (dictionary, characters in dataset, characters forming words) 
      chars = str().join(self.charList)
      wordChars = open('../model/wordCharList.txt').read().splitlines()[0]
      corpus = open('../data/corpus.txt').read()

      # decode using the "Words" mode of word beam search
      self.decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(self.ctcIn3dTBC, dim=2), 50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))


  def setupTF(self):
    "initialize TF"
    print('Python: '+sys.version)
    print('Tensorflow: '+tf.__version__)

    sess=tf.Session() # TF session

    # saver = tf.train.Saver(max_to_keep=1) # saver saves model to file
    # modelDir = '../model/'
    # latestSnapshot = tf.train.latest_checkpoint(modelDir) # is there a saved model?

    # # if model must be restored (for inference), there must be a snapshot
    # if self.mustRestore and not latestSnapshot:
    #   raise Exception('No saved model found in: ' + modelDir)

    # # load saved model if available
    # if latestSnapshot:
    #   print('Init with stored values from ' + latestSnapshot)
    #   saver.restore(sess, latestSnapshot)
    # else:
    #   print('Init with new values')
    sess.run(tf.global_variables_initializer())

    return (sess,saver)


  def toSparse(self, texts):
    "put ground truth texts into sparse tensor for ctc_loss"
    indices = []
    values = []
    shape = [len(texts), 0] # last entry must be max(labelList[i])

    # go over all texts
    for (batchElement, text) in enumerate(texts):
      # convert to string of label (i.e. class-ids)
      labelStr = [self.charList.index(c) for c in text]
      # sparse tensor must have size of max. label-string
      if len(labelStr) > shape[1]:
        shape[1] = len(labelStr)
      # put each label into sparse tensor
      for (i, label) in enumerate(labelStr):
        indices.append([batchElement, i])
        values.append(label)

    return (indices, values, shape)  # for label in ctc loss

  def decoderOutputToText(self, ctcOutput, batchSize):
    "extract texts from output of CTC decoder"

    # contains string of labels for each batch element
    encodedLabelStrs = [[] for i in range(batchSize)]

    # word beam search: label strings terminated by blank
    if self.decoderType == DecoderType.WordBeamSearch:
      blank=len(self.charList)
      for b in range(batchSize):
        for label in ctcOutput[b]:
          if label== blank:
            break
          encodedLabelStrs[b].append(label)

    # TF decoders: label strings are contained in sparse tensor
    else:
      # ctc returns tuple, first element is SparseTensor 
      decoded=ctcOutput[0][0] 

      # go over all indices and save mapping: batch -> values
      idxDict = { b : [] for b in range(batchSize) }
      for (idx, idx2d) in enumerate(decoded.indices):
        label = decoded.values[idx]
        batchElement = idx2d[0] # index according to [b,t]
        encodedLabelStrs[batchElement].append(label)

    # map labels to chars for all batch elements
    return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]


  def trainBatch(self, batch):
    "feed a batch into the NN to train it"
    numBatchElements = len(batch.imgs)
    sparse = self.toSparse(batch.gtTexts)
    rate = 0.0001 #0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001) # decay learning rate
    evalList = [self.optimizer, self.loss]
    feedDict = {self.inputImgs : batch.imgs, self.gtTexts : sparse , self.seqLen : [Model.maxTextLen] * numBatchElements, self.learningRate : rate}
    (_, lossVal) = self.sess.run(evalList, feedDict)
    self.batchesTrained += 1
    return lossVal


  def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
    "feed a batch into the NN to recngnize the texts"

    # decode, optionally save RNN output
    numBatchElements = len(batch.imgs)
    evalList = [self.decoder] + ([self.ctcIn3dTBC] if calcProbability else [])
    feedDict = {self.inputImgs : batch.imgs, self.seqLen : [Model.maxTextLen] * numBatchElements}
    evalRes = self.sess.run([self.decoder, self.ctcIn3dTBC], feedDict)
    decoded = evalRes[0]
    texts = self.decoderOutputToText(decoded, numBatchElements)

    # feed RNN output and recognized text into CTC loss to compute labeling probability
    probs = None
    if calcProbability:
      sparse = self.toSparse(batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
      ctcInput = evalRes[1]
      evalList = self.lossPerElement
      feedDict = {self.savedCtcInput : ctcInput, self.gtTexts : sparse, self.seqLen : [Model.maxTextLen] * numBatchElements}
      lossVals = self.sess.run(evalList, feedDict)
      probs = np.exp(-lossVals)
    return (texts, probs)


  def save(self):
    "save model to file"
    self.snapID += 1
    self.saver.save(self.sess, '../model/snapshot', global_step=self.snapID)

