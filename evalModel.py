#! /usr/bin/env python
# -*- coding:utf8 -*-

import tensorflow as tf

from utils import InputHelper

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("eval_filepath", "validation.txt0", "Evaluate on this data (Default: None)")
tf.flags.DEFINE_string("vocab_filepath", "data/vocab",
                       "Load training time vocabulary (Default: None)")
tf.flags.DEFINE_string("model", "save/model.ckpt-100",
                       "Load trained model checkpoint (Default: None)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.eval_filepath == None or FLAGS.vocab_filepath == None or FLAGS.model == None:
    print("Eval or Vocab filepaths are empty.")
    exit()

# load data and map id-transform based on training time vocabulary
inpH = InputHelper("", "", 0, 0, False)
x1_test, x2_test = inpH.getTestDataSet(FLAGS.eval_filepath, FLAGS.vocab_filepath, 40)
# y_test可删除

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = FLAGS.model
print checkpoint_file
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default(), open('predict.txt', 'w') as f:
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x1 = graph.get_operation_by_name("input_x").outputs[0]
        input_x2 = graph.get_operation_by_name("input_y").outputs[0]
        distance = graph.get_operation_by_name("distance").outputs[0]
        output_y = graph.get_operation_by_name("y_data").outputs[0]
        temp_sim = graph.get_operation_by_name("temp_sim").outputs[0]

        # dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # Tensors we want to evaluate
        # predictions = graph.get_operation_by_name("output/distance").outputs[0]

        # accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

        # sim = graph.get_operation_by_name("accuracy/temp_sim").outputs[0]

        # emb = graph.get_operation_by_name("embedding/W").outputs[0]
        # embedded_chars = tf.nn.embedding_lookup(emb,input_x)
        # Generate batches for one epoch
        batches = inpH.batch_iter(list(zip(x1_test, x2_test)), 2 * FLAGS.batch_size, 1, shuffle=False)
        # Collect the predictions here
        all_predictions = []
        all_d = []
        num = 1
        for db in batches:
            x1_dev_b, x2_dev_b = zip(*db)
            distance_out,similarity = sess.run([distance,temp_sim], {input_x1: x1_dev_b, input_x2: x2_dev_b})
            # for sizenum in range(batch_predictions.size):
            #    f.write(str(num) + "\t" + str(batch_predictions[sizenum]) + "\n")
            #    num = num + 1

            for sizenum in range(distance_out.size):
                f.write(str(num) + "\t" + str(distance_out[sizenum]) + "\n")

                num = num + 1
            # if distance[sizenum] > 0.7:
            #        f.write(str(num) + "\t" + str(0) + "\n")
            #   else:
            #        f.write(str(num) + "\t" + str(1) + "\n")
            #    num = num + 1

            print("the end")
