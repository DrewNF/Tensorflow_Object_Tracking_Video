# """
# Preparing model:
#  - Install bazel ( check tensorflow's github for more info )
#     Ubuntu 14.04:
#         - Requirements:
#             sudo add-apt-repository ppa:webupd8team/java
#             sudo apt-get update
#             sudo apt-get install oracle-java8-installer
#         - Download bazel, ( https://github.com/bazelbuild/bazel/releases )
#           tested on: https://github.com/bazelbuild/bazel/releases/download/0.2.0/bazel-0.2.0-jdk7-installer-linux-x86_64.sh
#         - chmod +x PATH_TO_INSTALL.SH
#         - ./PATH_TO_INSTALL.SH --user
#         - Place bazel onto path ( exact path to store shown in the output)
# - For retraining, prepare folder structure as
#     - root_folder_name
#         - class 1
#             - file1
#             - file2
#         - class 2
#             - file1
#             - file2
# - Clone tensorflow
# - Go to root of tensorflow
# - bazel build tensorflow/examples/image_retraining:retrain
# - bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir /path/to/root_folder_name  --output_graph /path/output_graph.pb -- output_labels /path/output_labels.txt -- bottleneck_dir /path/bottleneck
# ** Training done. **
# For testing through bazel,
#     bazel build tensorflow/examples/label_image:label_image && \
#     bazel-bin/tensorflow/examples/label_image/label_image \
#     --graph=/path/output_graph.pb --labels=/path/output_labels.txt \
#     --output_layer=final_result \
#     --image=/path/to/test/image
# For testing through python, change and run this code.
# """

import numpy as np
import tensorflow as tf
import sys
import vid_classes


modelFullPath = 'output_model/retrained_graph.pb' ##### Put the 
checkpoint_dir= "output_model/model.ckpt-250000"
label_file='output_model/retrained_labels.txt'

#### Label Informations Class
# class Label(object):

#     def __init__(self):
#         """Return a rect object whose coords are *0* and infos none ."""
#         self.conf = -1
#         self.label=None
#         self.label_code=None
#         self.label_chall=None

#     def Label(self, conf, label, label_chall, code):
#         self.conf = conf
#         self.label=label
#         self.label_code=code
#         self.label_chall=label_chall
#         return self 

# def run_inference_on_image():
#     answer = None

#     if not tf.gfile.Exists(imagePath):
#         tf.logging.fatal('File does not exist %s', imagePath)
#         return answer

#     image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

#     # Creates graph from saved GraphDef.
#     create_graph()

#     with tf.Session() as sess:

#         softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
#         predictions = sess.run(softmax_tensor,
#                                {'DecodeJpeg/contents:0': image_data})
#         predictions = np.squeeze(predictions)

#         top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
#         f = open(labelsFullPath, 'rb')
#         lines = f.readlines()
#         labels = [str(w).replace("\n", "") for w in lines]
#         for node_id in top_k:
#             human_string = labels[node_id]
#             score = predictions[node_id]
#             print('%s (score = %.5f)' % (human_string, score))

#         answer = labels[top_k[0]]
#         return answer


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

# def load_checkpoint(sess, saver):

#     # ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
#     # if ckpt and ckpt.model_checkpoint_path:
#     #     saver.restore(sess, ckpt.model_checkpoint_path)
#     #     print "Model loaded: %s"%checkpoint_dir
#     # else:
#     #     print "ERROR: ...no checkpoint found..."
#     #     sys.exit()

#     ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
#     if ckpt and ckpt.model_checkpoint_path:
#       if os.path.isabs(ckpt.model_checkpoint_path):
#         # Restores from checkpoint with absolute path.
#         saver.restore(sess, ckpt.model_checkpoint_path)
#       else:
#         # Restores from checkpoint with relative path.
#         saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
#                                          ckpt.model_checkpoint_path))

#       # Assuming model_checkpoint_path looks something like:
#       #   /my-favorite-path/imagenet_train/model.ckpt-0,
#       # extract global_step from it.
#       global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#       print('Succesfully loaded model from %s at step=%s.' %
#             (ckpt.model_checkpoint_path, global_step))
#     else:
#       print('No checkpoint file found')

def run_inception_once(picture_path):

    if not tf.gfile.Exists(picture_path):
        tf.logging.fatal('File does not exist %s', picture_path)
        sys.exit()

    image_data = tf.gfile.FastGFile(picture_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line in tf.gfile.GFile("/tf_files/retrained_labels.txt")]
    # Creates graph from saved GraphDef.
    create_graph()
    saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        # load_checkpoint(sess, saver)

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-3:][::-1]  # Getting top 5 predictions

        #CHECK OUTPUT
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))

        # CHECK BEST LABEL
        print "Best Label: %s with conf: %.5f"%(label_lines[top_k[0]],predictions[top_k[0]])

        return label_lines[top_k[0]],predictions[top_k[0]]

def run_inception(pictures_path_array):

    labels=[]
    confidences=[]
    # Creates graph from saved GraphDef.
    # Creates graph from saved GraphDef.
    create_graph()
    saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        # load_checkpoint(sess, saver)

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        for picture_path in pictures_path_array:

            if not tf.gfile.Exists(picture_path):
                tf.logging.fatal('File does not exist %s', picture_path)
                sys.exit()

            image_data = tf.gfile.FastGFile(picture_path, 'rb').read()
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions

            #CHECK OUTPUT
            # for node_id in top_k:
            #     human_string = vid_classes.code_comp_to_class(node_id)
            #     score = predictions[node_id]
            #     print('%s (score = %.5f)' % (human_string, score))

            #CHECK BEST LABEL
            #print "Best Label: %s with conf: %.5f"%(vid_classes.code_comp_to_class(top_k[0]),predictions[top_k[0]])

            labels.append(vid_classes.code_comp_to_class(top_k[0]), len(labels))
            confidences.append(predictions[top_k[0]], len(confidences))

        return labels, confidences

# def inputs(dataset, batch_size=1, num_preprocess_threads=1):
#   """Generate batches of ImageNet images for evaluation.
#   Use this function as the inputs for evaluating a network.
#   Note that some (minimal) image preprocessing occurs during evaluation
#   including central cropping and resizing of the image to fit the network.
#   Args:
#     dataset: instance of Dataset class specifying the dataset.
#     batch_size: integer, number of examples in batch
#     num_preprocess_threads: integer, total number of preprocessing threads but
#       None defaults to FLAGS.num_preprocess_threads.
#   Returns:
#     images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
#                                        image_size, 3].
#     labels: 1-D integer Tensor of [FLAGS.batch_size].
#   """

#   # Force all input processing onto CPU in order to reserve the GPU for
#   # the forward inference and back-propagation.
#   with tf.device('/cpu:0'):
#     images, labels = batch_inputs(
#         dataset, batch_size, train=False,
#         num_preprocess_threads=num_preprocess_threads,
#         num_readers=1)

# return images, labels

# def batch_inputs(dataset, batch_size, train, num_preprocess_threads=None,
#                  num_readers=1):
#   """Contruct batches of training or evaluation examples from the image dataset.
#   Args:
#     dataset: instance of Dataset class specifying the dataset.
#       See dataset.py for details.
#     batch_size: integer
#     train: boolean
#     num_preprocess_threads: integer, total number of preprocessing threads
#     num_readers: integer, number of parallel readers
#   Returns:
#     images: 4-D float Tensor of a batch of images
#     labels: 1-D integer Tensor of [batch_size].
#   Raises:
#     ValueError: if data is not found
#   """
#   with tf.name_scope('batch_processing'):
#     data_files = dataset.data_files()
#     if data_files is None:
#       raise ValueError('No data files found for this dataset')

#     # Create filename_queue
#     if train:
#       filename_queue = tf.train.string_input_producer(data_files,
#                                                       shuffle=True,
#                                                       capacity=16)
#     else:
#       filename_queue = tf.train.string_input_producer(data_files,
#                                                       shuffle=False,
#                                                       capacity=1)
#     if num_preprocess_threads is None:
#       num_preprocess_threads = FLAGS.num_preprocess_threads

#     if num_preprocess_threads % 4:
#       raise ValueError('Please make num_preprocess_threads a multiple '
#                        'of 4 (%d % 4 != 0).', num_preprocess_threads)

#     if num_readers is None:
#       num_readers = FLAGS.num_readers

#     if num_readers < 1:
#       raise ValueError('Please make num_readers at least 1')

#     # Approximate number of examples per shard.
#     examples_per_shard = 1024
#     # Size the random shuffle queue to balance between good global
#     # mixing (more examples) and memory use (fewer examples).
#     # 1 image uses 299*299*3*4 bytes = 1MB
#     # The default input_queue_memory_factor is 16 implying a shuffling queue
#     # size: examples_per_shard * 16 * 1MB = 17.6GB
#     min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
#     if train:
#       examples_queue = tf.RandomShuffleQueue(
#           capacity=min_queue_examples + 3 * batch_size,
#           min_after_dequeue=min_queue_examples,
#           dtypes=[tf.string])
#     else:
#       examples_queue = tf.FIFOQueue(
#           capacity=examples_per_shard + 3 * batch_size,
#           dtypes=[tf.string])

#     # Create multiple readers to populate the queue of examples.
#     if num_readers > 1:
#       enqueue_ops = []
#       for _ in range(num_readers):
#         reader = dataset.reader()
#         _, value = reader.read(filename_queue)
#         enqueue_ops.append(examples_queue.enqueue([value]))

#       tf.train.queue_runner.add_queue_runner(
#           tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
#       example_serialized = examples_queue.dequeue()
#     else:
#       reader = dataset.reader()
#       _, example_serialized = reader.read(filename_queue)

#     images_and_labels = []
#     for thread_id in range(num_preprocess_threads):
#       # Parse a serialized Example proto to extract the image and metadata.
#       image_buffer, label_index, bbox, _ = parse_example_proto(
#           example_serialized)
#       image = image_preprocessing(image_buffer, bbox, train, thread_id)
#       images_and_labels.append([image, label_index])

#     images, label_index_batch = tf.train.batch_join(
#         images_and_labels,
#         batch_size=batch_size,
#         capacity=2 * num_preprocess_threads * batch_size)

#     # Reshape images into these desired dimensions.
#     height = FLAGS.image_size
#     width = FLAGS.image_size
#     depth = 3

#     images = tf.cast(images, tf.float32)
#     images = tf.reshape(images, shape=[batch_size, height, width, depth])

#     # Display the training images in the visualizer.
#     tf.image_summary('images', images)

# return images, tf.reshape(label_index_batch, [batch_size])

# def _eval_once(saver, top_1_op, top_5_op):
#   """Runs Eval once.
#   Args:
#     saver: Saver.
#     summary_writer: Summary writer.
#     top_1_op: Top 1 op.
#     top_5_op: Top 5 op.
#     summary_op: Summary op.
#   """
#   with tf.Session() as sess:
#     ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
#     if ckpt and ckpt.model_checkpoint_path:
#       if os.path.isabs(ckpt.model_checkpoint_path):
#         # Restores from checkpoint with absolute path.
#         saver.restore(sess, ckpt.model_checkpoint_path)
#       else:
#         # Restores from checkpoint with relative path.
#         saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
#                                          ckpt.model_checkpoint_path))

#       # Assuming model_checkpoint_path looks something like:
#       #   /my-favorite-path/imagenet_train/model.ckpt-0,
#       # extract global_step from it.
#       global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#       print('Succesfully loaded model from %s at step=%s.' %
#             (ckpt.model_checkpoint_path, global_step))
#     else:
#       print('No checkpoint file found')
#       return

#     # Start the queue runners.
#     coord = tf.train.Coordinator()
#     try:
#       threads = []
#       for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
#         threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
#                                          start=True))

#       num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
#       # Counts the number of correct predictions.
#       count_top_1 = 0.0
#       count_top_5 = 0.0
#       total_sample_count = num_iter * FLAGS.batch_size
#       step = 0

#       print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
#       start_time = time.time()
#       while step < num_iter and not coord.should_stop():
#         top_1, top_5 = sess.run([top_1_op, top_5_op])
#         count_top_1 += np.sum(top_1)
#         count_top_5 += np.sum(top_5)
#         step += 1
#         if step % 20 == 0:
#           duration = time.time() - start_time
#           sec_per_batch = duration / 20.0
#           examples_per_sec = FLAGS.batch_size / sec_per_batch
#           print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
#                 'sec/batch)' % (datetime.now(), step, num_iter,
#                                 examples_per_sec, sec_per_batch))
#           start_time = time.time()

#       # Compute precision @ 1.
#       precision_at_1 = count_top_1 / total_sample_count
#       recall_at_5 = count_top_5 / total_sample_count
#       print('%s: precision @ 1 = %.4f recall @ 5 = %.4f [%d examples]' %
#             (datetime.now(), precision_at_1, recall_at_5, total_sample_count))

#     except Exception as e:  # pylint: disable=broad-except
#       coord.request_stop(e)

#     coord.request_stop()
#     coord.join(threads, stop_grace_period_secs=10)


# def evaluate(dataset):
#   """Evaluate model on Dataset for a number of steps."""
#   with tf.Graph().as_default():
#     # Get images and labels from the dataset.
#     images, labels = image_processing.inputs(dataset)

#     # Number of classes in the Dataset label set plus 1.
#     # Label 0 is reserved for an (unused) background class.
#     num_classes = dataset.num_classes() + 1

#     # Build a Graph that computes the logits predictions from the
#     # inference model.
#     logits, _ = inception.inference(images, num_classes)

#     # Calculate predictions.
#     top_1_op = tf.nn.in_top_k(logits, labels, 1)
#     top_5_op = tf.nn.in_top_k(logits, labels, 5)

#     # Restore the moving average version of the learned variables for eval.
#     variable_averages = tf.train.ExponentialMovingAverage(
#         inception.MOVING_AVERAGE_DECAY)
#     variables_to_restore = variable_averages.variables_to_restore()
#     saver = tf.train.Saver(variables_to_restore)

#     while True:
#       _eval_once(saver, top_1_op, top_5_op)
#       if FLAGS.run_once:
#         break
# 	time.sleep(FLAGS.eval_interval_secs)
