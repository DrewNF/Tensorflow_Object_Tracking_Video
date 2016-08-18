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

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

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
