import sys
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

#from solver import ModelSolver
#from ConvLSTM import ConvLSTM
#from AttConvLSTM import AttConvLSTM
sys.path.append('/Users/frances/Documents/DeepLearning/Code/TaxiPrediction/util/')
sys.path.append('/home/zx/TaxiPrediction/util/')
from utils import *
from preprocessing import *

#FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string('train_dir', './checkpoints/train_store_convLSTM_taxi',
#                            """dir to store trained net""")

# tf.app.flags.DEFINE_integer('input_steps', 10,
#                             """num of input_steps""")
# tf.app.flags.DEFINE_integer('output_steps', 10,
#                             """num of output_steps""")
# tf.app.flags.DEFINE_integer('batch_size', 16,
#                             """batch size for training""")
# tf.app.flags.DEFINE_integer('n_epochs', 20,
#                             """num of epochs""")
# tf.app.flags.DEFINE_float('keep_prob', .9,
#                             """for dropout""")
# tf.app.flags.DEFINE_float('lr', .000001,
#                             """for dropout""")
# tf.app.flags.DEFINE_string('update_rule', 'adam',
#                             """update rule""")
# tf.app.flags.DEFINE_integer('save_every', 1,
#                             """steps to save""")

# tf.app.flags.DEFINE_boolean('use_att', True,
#                             """whether to use attention mechanism""")
# tf.app.flags.DEFINE_integer('cluster_num', 128,
#                             """num of cluster in attention mechanism""")
# tf.app.flags.DEFINE_float('weight_init', .1,
#                             """weight init for fully connected layers""")

# tf.app.flags.DEFINE_integer('input_height', 64,
#                             """height of input frame""")
# tf.app.flags.DEFINE_integer('input_width', 64,
#                             """width of input frame""")
# tf.app.flags.DEFINE_integer('input_channel', 1,
#                             """channels of input frame""")

def main():
    # load train dataset
    pre_process = MinMaxNormalization01()
    print('load train, validate, test data...')
    #data, train_data, _, _ = load_data(filename=['../data/taxi/p_map.mat', '../data/taxi/d_map.mat'], split=[43824, 8760, 8760])
    data, train_data, val_data, test_data = load_npy_data(filename=['../data/citybike/p_map.npy', '../data/citybike/d_map.npy'], split=[17520, 4416, 4368])
    print('preprocess train data...')
    train_data = pre_process.fit_transform(train_data)
    # data: [num, row, col, channel]
    #print('get batch data...')
    #train_x, train_y = batch_data(data=train_data, batch_size=FLAGS.batch_size,
    #    input_steps=FLAGS.input_steps, output_steps=FLAGS.output_steps)
    # load validate dataset 
    #val_data = pre_process.transform(val_data)
    #val_x, val_y = batch_data(data=val_data, batch_size=FLAGS.batch_size,
    #    input_steps=FLAGS.input_steps, output_steps=FLAGS.output_steps)

    #train = {'x': train_x, 'y': train_y}
    #val = {'x': val_x, 'y': val_y}

    #input_dim = [train_data.shape[1], train_data.shape[2], train_data.shape[3]]
    #print('build model...')

    #if FLAGS.use_att:
    K = range(2,101,2)
    meandistortions = []
    sc = []
    for k in K:
        # k-means to cluster train_data
        # train_data: [num, row, col, channel]
	print '=============== k : '+str(k)
        print('k-means to cluster...')
        vector_data = np.reshape(train_data, (train_data.shape[0], -1))
        print('shape of vector_data: ', vector_data.shape)
        #init_vectors = vector_data[:FLAGS.cluster_num, :]
        kmeans = KMeans(n_clusters=k, n_init=5, tol=0.00000001).fit(vector_data)
        labels = kmeans.labels_
        cluster_centroid = kmeans.cluster_centers_
        print('cluster centroid shape: ', cluster_centroid.shape)
        meandistortions.append(sum(np.min(cdist(vector_data, cluster_centroid, 'euclidean'), axis=1))/vector_data.shape[0])
        sc.append(metrics.silhouette_score(vector_data, labels,metric='euclidean'))
        # for i in range(FLAGS.cluster_num):
        #     print('label %d : %d', i, labels[i])
        #print('inertia is: ', kmeans.inertia_)
        # build model
        # print('build AttConvLSTM model...')
    meandistortions = np.array(meandistortions)
    sc = np.array(sc)
    np.save('bike-meandistortions.npy', meandistortions)
    np.save('bike-sc.npy', sc)
    # plt.plot(K, meandistortions, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('mean distortions')
    # plt.plot(K, sc, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('silhouette coefficient')

if __name__ == "__main__":
    main()
