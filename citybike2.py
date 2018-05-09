import numpy as np
import tensorflow as tf
import os
import sys
from sklearn.cluster import KMeans
from solver import ModelSolver
from model.ConvLSTM import ConvLSTM
from model.autoencoder import AutoEncoder
from model.AttConvLSTM import AttConvLSTM
from model.MultiAttConvLSTM import MultiAttConvLSTM
from model.ResNet import ResNet
from model.AttResNet import AttResNet
from util.preprocessing import *
from util.utils import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('gpu', '0', """which gpu to use: 0 or 1""")

tf.app.flags.DEFINE_integer('input_steps', 10,
                            """num of input_steps""")
tf.app.flags.DEFINE_integer('output_steps', 10,
                            """num of output_steps""")
tf.app.flags.DEFINE_integer('batch_size', 16, """batch size for training""")
tf.app.flags.DEFINE_integer('n_epochs', 20, """num of epochs""")
tf.app.flags.DEFINE_float('keep_prob', .9, """for dropout""")
tf.app.flags.DEFINE_float('lr', .0002, """for dropout""")
tf.app.flags.DEFINE_string('update_rule', 'adam', """update rule""")
tf.app.flags.DEFINE_integer('save_every', 1,
                            """steps to save""")
# model: ConvLSTM, AttConvLSTM, ResNet
tf.app.flags.DEFINE_string('model', 'MultiAttConvLSTM', """which model to train and test""")
tf.app.flags.DEFINE_integer('weighted_loss', 0, """is use weighted loss as loss function""")
# ResNet
tf.app.flags.DEFINE_integer('closeness', 3,
                            """num of closeness""")
tf.app.flags.DEFINE_integer('period', 4,
                            """num of period""")
tf.app.flags.DEFINE_integer('trend', 4,
                            """num of trend""")
# AttConvLSTM
tf.app.flags.DEFINE_integer('cluster_num', 128,
                            """num of cluster in attention mechanism""")
tf.app.flags.DEFINE_integer('kmeans_run_num', 5,
                            """number of times for running kmeans to cluster""")
tf.app.flags.DEFINE_integer('att_nodes', 1024,
                            """num of nodes in attention layer""")
# cluster
tf.app.flags.DEFINE_integer('pre_saved_cluster', 0,
                            """if use saved cluster as annotation tensors""")
tf.app.flags.DEFINE_integer('kmeans_cluster', 0,
                            """whether to use kmeans for clustering""")
tf.app.flags.DEFINE_integer('average_cluster', 24,
                            """whether to use average map for clustering""")
# train/test
tf.app.flags.DEFINE_integer('train', 1,
                            """whether to train""")
tf.app.flags.DEFINE_integer('test', 0,
                            """whether to test""")
tf.app.flags.DEFINE_string('pretrained_model', None,
                           """pretrained_model_name""")


def average_cluster_24(vector_data):
    v_d = np.asarray(vector_data, dtype=np.float32)
    v_shape = v_d.shape
    index = 24*np.arange(v_shape[0]/24)
    ave_cluster = np.zeros((24, v_shape[1]))
    for i in xrange(24):
        ave_cluster[i] = np.mean(v_d[index+i], axis=0)
    return ave_cluster

def average_cluster_48(vector_data):
    v_d = np.asarray(vector_data, dtype=np.float32)
    v_shape = v_d.shape
    # weekday and weekend
    w_num = v_shape[0]/168
    wday_index = [np.arange(120)*k for k in xrange(w_num)]
    wend_index = [np.arange(120, 168)*k for k in xrange(w_num)]
    wday = v_d[wday_index]
    wend = v_d[wend_index]
    ave_cluster = np.zeros((48, v_shape[1]))
    ind1 = 24*np.arange(len(wday)/24)
    ind2 = 24*np.arange(len(wend)/24)
    for i in xrange(24):
        ave_cluster[i] = np.mean(wday[ind1+i], axis=0)
    for j in xrange(24):
        ave_cluster[j+24] = np.mean(wend[ind2+j], axis=0)
    return ave_cluster


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    # preprocessing class
    pre_process = MinMaxNormalization01()
    print('load train, validate, test data...')
    split = [17520, 4416, 4368]
    data, train_data, val_data, test_data = load_npy_data(
        filename=['data/citybike/p_map.npy', 'data/citybike/d_map.npy'], split=split)
    # data: [num, row, col, channel]
    print('preprocess train data...')
    pre_process.fit(train_data)

    train_data = pre_process.transform(train_data)
    train_x, train_y = batch_data(data=train_data, batch_size=FLAGS.batch_size,
                                  input_steps=FLAGS.input_steps, output_steps=FLAGS.output_steps)
    val_data = pre_process.transform(val_data)
    val_x, val_y = batch_data(data=val_data, batch_size=FLAGS.batch_size,
                              input_steps=FLAGS.input_steps, output_steps=FLAGS.output_steps)
    test_data = pre_process.transform(test_data)
    test_x, test_y = batch_data(data=test_data, batch_size=FLAGS.batch_size,
                                input_steps=FLAGS.input_steps, output_steps=FLAGS.output_steps)
    train = {'x': train_x, 'y': train_y}
    val = {'x': val_x, 'y': val_y}
    test = {'x': test_x, 'y': test_y}
    input_dim = [train_data.shape[1], train_data.shape[2], train_data.shape[3]]
    if 'AttConvLSTM' in FLAGS.model:
        # train_data: [num, row, col, channel]
        model_path = 'citybike-results/model_save/' + FLAGS.model + '/'
        log_path = 'citybike-results/log/' + FLAGS.model + '/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if FLAGS.pre_saved_cluster:
            cluster_centroid = np.load(model_path + 'cluster_centroid.npy')
        else:
            vector_data = np.reshape(train_data, (train_data.shape[0], -1))
            cluster_centroid_1 = None
            cluster_centroid_2 = None
            cluster_centroid = None
            if FLAGS.kmeans_cluster:
                print('k-means to cluster...')
                kmeans = KMeans(n_clusters=FLAGS.cluster_num, init='random', n_init=FLAGS.kmeans_run_num,
                                tol=0.00000001).fit(vector_data)
                cluster_centroid_1 = kmeans.cluster_centers_
            if FLAGS.average_cluster:
                print('average cluster...')
                if FLAGS.average_cluster == 24:
                    cluster_centroid_2 = average_cluster_24(vector_data)
                elif FLAGS.average_cluster == 48:
                    cluster_centroid_2 = average_cluster_48(vector_data)
            if cluster_centroid_1 is not None:
                cluster_centroid = cluster_centroid_1
            if cluster_centroid_2 is not None:
                if cluster_centroid is not None:
                    cluster_centroid = np.concatenate((cluster_centroid_1, cluster_centroid_2), axis=0)
                else:
                    cluster_centroid = cluster_centroid_2
            # reshape to [cluster_num, row, col, channel]
            cluster_centroid = np.reshape(cluster_centroid,
                                          (-1, train_data.shape[1], train_data.shape[2], train_data.shape[3]))
            np.save(model_path + 'cluster_centroid.npy', cluster_centroid)
        # build model
        print 'build ' + FLAGS.model + ' model...'
        if FLAGS.model == 'AttConvLSTM':
            model = AttConvLSTM(input_dim=input_dim,
                                att_inputs=cluster_centroid, att_nodes=FLAGS.att_nodes,
                                batch_size=FLAGS.batch_size,
                                layer={'encoder': ['conv', 'conv', 'conv_lstm', 'conv_lstm'],
                                       'decoder': ['conv_lstm', 'conv_lstm', 'conv', 'conv'],
                                       'attention': ['conv', 'conv']},
                                layer_param={'encoder': [[[3, 3], [1, 1, 1, 1], 8],
                                                         [[3, 3], [1, 1, 1, 1], 16],
                                                         [[16, 16], [3, 3], 64],
                                                         [[16, 16], [3, 3], 64]],
                                             'decoder': [[[16, 16], [3, 3], 64],
                                                         [[16, 16], [3, 3], 64],
                                                         [[3, 3], [1, 1, 1, 1], 8],
                                                         [[3, 3], [1, 1, 1, 1], 2]],
                                             'attention': [[[3, 3], [1, 1, 1, 1], 8],
                                                           [[3, 3], [1, 1, 1, 1], 16]]},
                                input_steps=10, output_steps=10)
        elif FLAGS.model == 'MultiAttConvLSTM':
            model = MultiAttConvLSTM(input_dim=input_dim,
                                     att_inputs=cluster_centroid, att_nodes=FLAGS.att_nodes,
                                     batch_size=FLAGS.batch_size,
                                     layer={'encoder': ['conv', 'conv', 'conv_lstm', 'conv_lstm'],
                                            'decoder': ['conv_lstm', 'conv_lstm', 'conv', 'conv'],
                                            'attention': ['conv', 'conv']},
                                     layer_param={'encoder': [[[3, 3], [1, 1, 1, 1], 8],
                                                              [[3, 3], [1, 1, 1, 1], 16],
                                                              [[16, 16], [3, 3], 64],
                                                              [[16, 16], [3, 3], 64]],
                                                  'decoder': [[[16, 16], [3, 3], 64],
                                                              [[16, 16], [3, 3], 64],
                                                              [[3, 3], [1, 1, 1, 1], 8],
                                                              [[3, 3], [1, 1, 1, 1], 2]],
                                                  'attention': [[[3, 3], [1, 1, 1, 1], 8],
                                                                [[3, 3], [1, 1, 1, 1], 16]]},
                                     input_steps=10, output_steps=10)
        print('model solver...')
        solver = ModelSolver(model, train, val, preprocessing=pre_process,
                             n_epochs=FLAGS.n_epochs,
                             batch_size=FLAGS.batch_size,
                             update_rule=FLAGS.update_rule,
                             learning_rate=FLAGS.lr, save_every=FLAGS.save_every,
                             pretrained_model=FLAGS.pretrained_model, model_path=model_path,
                             test_model=model_path + 'model-' + str(FLAGS.n_epochs), log_path=log_path)
    if FLAGS.train:
        print('begin training...')
        test_prediction, _ = solver.train(test)
        test_target = np.asarray(test_y)
    if FLAGS.test:
        print('test trained model...')
        solver.test_model = solver.model_path + FLAGS.pretrained_model
        test_prediction = solver.test(test)
        test_target = np.asarray(test_y)
    np.save('citybike-results/results/' + FLAGS.model + '/test_target.npy', test_target)
    np.save('citybike-results/results/' + FLAGS.model + '/test_prediction.npy', test_prediction)


if __name__ == "__main__":
    main()
