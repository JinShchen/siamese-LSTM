# -*- coding:utf-8 -*-

import os
import random
import time
import re
import tensorflow as tf
import numpy as np
from siamese_lstm import SiameseLSTM
from utils import InputHelper

# Parameters
# =================================================

tf.flags.DEFINE_integer('rnn_size',300,
                        'hidden units of RNN , as well as dimensionality of character embedding (default: 64)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability (default : 0.5)')
tf.flags.DEFINE_integer('layer_size', 4, 'number of layers of RNN (default: 2)')
tf.flags.DEFINE_integer('batch_size', 128, 'Batch Size (default : 32)')
tf.flags.DEFINE_integer('sequence_length', 30, 'Sequence length (default : 32)')
tf.flags.DEFINE_float('grad_clip', 5.0, 'clip gradients at this value')
tf.flags.DEFINE_integer("num_epochs", 100, 'Number of training epochs (default: 200)')
tf.flags.DEFINE_float('learning_rate', 0.002, 'learning rate')
tf.flags.DEFINE_float('decay_rate', 0.97, 'decay rate for rmsprop')
tf.flags.DEFINE_string('train_file', 'atec_nlp_sim_train.csv', 'train raw file')
tf.flags.DEFINE_string('test_file', 'validation.txt0', 'train raw file')
tf.flags.DEFINE_string('data_dir', 'data', 'data directory')
tf.flags.DEFINE_string('save_dir', 'save', 'model save directory')
tf.flags.DEFINE_string('log_dir', 'log', 'log directory')
tf.flags.DEFINE_string('init_from', None, 'continue training from saved model at this path')
tf.flags.DEFINE_string('num_batches', None, 'num_batches')
tf.flags.DEFINE_string('vocab_size', None, 'vocab_size')

FLAGS = tf.flags.FLAGS
FLAGS._flags()

def train():

    train_data_loader = InputHelper(FLAGS.data_dir, FLAGS.train_file, FLAGS.batch_size,
                                    FLAGS.sequence_length)
    FLAGS.num_batches = train_data_loader.num_batches
    FLAGS.vocab_size = len(train_data_loader.vocab_processor.vocabulary_)
    print len(train_data_loader.vocab_processor.vocabulary_)


    if FLAGS.init_from is not None:
        assert os.path.isdir(FLAGS.init_from), '{} must be a directory'.format(FLAGS.init_from)
        ckpt = tf.train.get_checkpoint_state(FLAGS.init_from)
        assert ckpt, 'No checkpoint found'
        assert ckpt.model_checkpoint_path, 'No model path found in checkpoint'

    model = SiameseLSTM(FLAGS.rnn_size, FLAGS.layer_size, FLAGS.vocab_size,
                        FLAGS.sequence_length, FLAGS.dropout_keep_prob, FLAGS.grad_clip)

    tf.summary.scalar('train_loss', model.cost)
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())




        # restore model
        if FLAGS.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in xrange(FLAGS.num_epochs):
            train_data_loader.reset_batch()
            b = 0
            while not train_data_loader.eos:
                b += 1
                start = time.time()
                x1_batch, x2_batch, y_batch = train_data_loader.next_batch()
                # random exchange x1_batch and x2_batch
                if random.random() > 0.5:
                    feed = {model.input_x1: x1_batch, model.input_x2: x2_batch, model.y_data: y_batch}
                else:
                    feed = {model.input_x1: x2_batch, model.input_x2: x1_batch, model.y_data: y_batch}
                train_loss, summary, _ = sess.run([model.cost, merged, model.train_op], feed_dict=feed)
                end = time.time()
                print '{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(e * FLAGS.num_batches + b,
                                                                                          FLAGS.num_epochs * FLAGS.num_batches,
                                                                                          e, train_loss, end - start)

                if (e * FLAGS.num_batches + b) % 500 == 0: #500
                    checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * FLAGS.num_batches + b)
                    print 'model saved to {}'.format(checkpoint_path)

                if b % 20 == 0:
                    train_writer.add_summary(summary, e * FLAGS.num_batches + b)


if __name__ == '__main__':
    train()


'''
 if FLAGS.word2vec_model:
            # initial matrix with random uniform
            initW = np.random.uniform(-0.25, 0.25, (len(train_data_loader.vocab_processor.vocabulary_), FLAGS.rnn_size))
            # initW = np.zeros(shape=(len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
            # load any vectors from the word2vec
            print("initializing initW with pre-trained word2vec embeddings")
            TOKENIZER_RE = re.compile(u'[\u4e00-\u9fa5]+', re.UNICODE)
            for w in train_data_loader.vocab_processor.vocabulary_._mapping:
                arr = []
                s = re.sub(u'[\u4e00-\u9fa5]+', '', w)
                ss = TOKENIZER_RE.findall(w)
                if w in train_data_loader.token_dictionary:
                    arr = train_data_loader.token_dictionary[w]
                else:
                    arr = train_data_loader.token_dictionary[u'的']
                # 此处没有数字
                if len(arr) > 0:
                    idx = train_data_loader.vocab_processor.vocabulary_.get(w)
                    initW[idx] = np.asarray(arr).astype(np.float32)
            print("Done assigning intiW. len=" + str(len(initW)))
            sess.run(model.W.assign(initW))
'''