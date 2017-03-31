import numpy as np
import tensorflow as tf
import sys
from sklearn.cluster import KMeans
from solver import ModelSolver

sys.path.append('/home/zx/TaxiPrediction/model/')
sys.path.append('./util/')
sys.path.append('./data/')
#from model.ConvLSTM import ConvLSTM
from ConvLSTM import *
#from model.AttConvLSTM import AttConvLSTM
from AttConvLSTM import *
from preprocessing import *
from utils import *

FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string('train_dir', './checkpoints/train_store_convLSTM_taxi',
#                            """dir to store trained net""")

tf.app.flags.DEFINE_integer('input_steps', 10,
                            """num of input_steps""")
tf.app.flags.DEFINE_integer('output_steps', 10,
                            """num of output_steps""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """batch size for training""")
tf.app.flags.DEFINE_integer('n_epochs', 30,
                            """num of epochs""")
tf.app.flags.DEFINE_float('keep_prob', .9,
                            """for dropout""")
tf.app.flags.DEFINE_float('lr', .000001,
                            """for dropout""")
tf.app.flags.DEFINE_string('update_rule', 'adam',
                            """update rule""")
tf.app.flags.DEFINE_integer('save_every', 1,
                            """steps to save""")

tf.app.flags.DEFINE_boolean('use_att', True,
                            """whether to use attention mechanism""")
tf.app.flags.DEFINE_integer('cluster_num', 128,
                            """num of cluster in attention mechanism""")
tf.app.flags.DEFINE_integer('kmeans_run_num', 5,
                            """num of cluster in attention mechanism""")
tf.app.flags.DEFINE_integer('att_nodes', 1024,
                            """num of nodes in attention layer""")
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
    train_data, val_data, test_data = load_data(filename=['data/p_map.mat', 'data/d_map.mat'], split=[43824, 8760, 8760])
    print('preprocess train data...')
    train_data = pre_process.fit_transform(train_data)
    # data: [num, row, col, channel]
    print('get batch data...')
    train_x, train_y = batch_data(data=train_data, batch_size=FLAGS.batch_size,
        input_steps=FLAGS.input_steps, output_steps=FLAGS.output_steps)
    # load validate dataset 
    val_data = pre_process.transform(val_data)
    val_x, val_y = batch_data(data=val_data, batch_size=FLAGS.batch_size,
        input_steps=FLAGS.input_steps, output_steps=FLAGS.output_steps)

    train = {'x': train_x, 'y': train_y}
    val = {'x': val_x, 'y': val_y}

    input_dim = [train_data.shape[1], train_data.shape[2], train_data.shape[3]]
    #print('build model...')

    if FLAGS.use_att:
        # k-means to cluster train_data
        # train_data: [num, row, col, channel]
        print('k-means to cluster...')
        vector_data = np.reshape(train_data, (train_data.shape[0], -1))
        #init_vectors = vector_data[:FLAGS.cluster_num, :]
	    #cluster_centroid = init_vectors
        kmeans = KMeans(n_clusters=FLAGS.cluster_num, init='random', n_init=FLAGS.kmeans_run_num, tol=0.00000001).fit(vector_data)
        cluster_centroid = kmeans.cluster_centers_
        # reshape to [cluster_num, row, col, channel]
        cluster_centroid = np.reshape(cluster_centroid, (-1, train_data.shape[1], train_data.shape[2], train_data.shape[3]))
        # build model
        print('build AttConvLSTM model...')
        model = AttConvLSTM(input_dim=input_dim, 
            att_inputs=cluster_centroid, att_nodes=FLAGS.att_nodes, 
            batch_size=FLAGS.batch_size, 
            layer={'encoder': ['conv', 'conv', 'conv_lstm', 'conv_lstm'], 
            'decoder': ['conv_lstm', 'conv_lstm', 'conv', 'conv'],
            'attention': ['conv', 'conv']}, 
            layer_param={'encoder': [ [[3,3], [1,2,2,1], 8], 
            [[3,3], [1,2,2,1], 16], 
            [[16,16], [3,3], 64], 
            [[16,16], [3,3], 64] ],
            'decoder': [ [[16,16], [3,3], 64], 
            [[16,16], [3,3], 64], 
            [[3,3], [1,2,2,1], 8], 
            [[3,3], [1,2,2,1], 2] ],
            'attention': [ [[3,3], [1,2,2,1], 8], 
            [[3,3], [1,2,2,1], 16] ]}, 
            input_steps=10, output_steps=10)
        print('model solver...')
        solver = ModelSolver(model, train, val, preprocessing=pre_process,
            n_epochs=FLAGS.n_epochs, 
            batch_size=FLAGS.batch_size, 
            update_rule=FLAGS.update_rule,
            learning_rate=FLAGS.lr, save_every=FLAGS.save_every, 
            pretrained_model=None, model_path='model_save/AttConvLSTM/', 
            test_model='model_save/AttConvLSTM/model-'+str(FLAGS.n_epochs), log_path='log/AttConvLSTM/')
    else:
        print('build ConvLSTM model...')
        model = ConvLSTM(input_dim=input_dim, batch_size=FLAGS.batch_size, 
            layer={'encoder': ['conv', 'conv', 'conv_lstm', 'conv_lstm'], 
            'decoder': ['conv_lstm', 'conv_lstm', 'conv', 'conv']}, 
            layer_param={'encoder': [ [[3,3], [1,2,2,1], 8], 
            [[3,3], [1,2,2,1], 16], 
            [[16,16], [3,3], 64], 
            [[16,16], [3,3], 64] ],
            'decoder': [ [[16,16], [3,3], 64], 
            [[16,16], [3,3], 64], 
            [[3,3], [1,2,2,1], 8], 
            [[3,3], [1,2,2,1], 2] ]}, 
            input_steps=10, output_steps=10)
        print('model solver...')
        solver = ModelSolver(model, train, val, preprocessing=pre_process,
            n_epochs=FLAGS.n_epochs, 
            batch_size=FLAGS.batch_size, 
            update_rule=FLAGS.update_rule,
            learning_rate=FLAGS.lr, save_every=FLAGS.save_every, 
            pretrained_model=None, model_path='model_save/ConvLSTM/', 
            test_model='model_save/ConvLSTM/model-'+str(FLAGS.n_epochs), log_path='log/ConvLSTM/')
    

    print('begin training...')
    solver.train()
    # preprocess test data and get batch test data
    print('test trained model...')
    test_data = pre_process.transform(test_data)
    test_x, test_y = batch_data(data=test_data, batch_size=FLAGS.batch_size,
        input_steps=FLAGS.input_steps, output_steps=FLAGS.output_steps)
    test = {'x': test_x, 'y': test_y}
    solver.test(test)

if __name__ == "__main__":
    main()
