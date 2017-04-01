import numpy as np
import tensorflow as tf
import sys
#from sklearn.cluster import KMeans
from solver import ModelSolver
# for mac debug
sys.path.append('/Users/frances/Documents/DeepLearning/Code/TaxiPrediction/model/')
sys.path.append('/Users/frances/Documents/DeepLearning/Code/TaxiPrediction/util/')
# for server running
sys.path.append('/home/zx/TaxiPrediction/model/')
sys.path.append('./util/')
sys.path.append('./data/')

from ResNet import *
from preprocessing import *
from utils import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('closeness', 3,
                            """num of closeness""")
tf.app.flags.DEFINE_integer('period', 4,
                            """num of period""")
tf.app.flags.DEFINE_integer('trend', 4,
                            """num of trend""")
tf.app.flags.DEFINE_integer('test_num', 10*24,
                            """num of test data""")

tf.app.flags.DEFINE_integer('input_steps', 10,
                            """num of input_steps""")
tf.app.flags.DEFINE_integer('output_steps', 10,
                            """num of output_steps""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """batch size for training""")
tf.app.flags.DEFINE_integer('n_epochs', 50,
                            """num of epochs""")
tf.app.flags.DEFINE_float('keep_prob', .9,
                            """for dropout""")
tf.app.flags.DEFINE_float('lr', .0002,
                            """for dropout""")
tf.app.flags.DEFINE_string('update_rule', 'adam',
                            """update rule""")
tf.app.flags.DEFINE_integer('save_every', 1,
                            """steps to save""")
tf.app.flags.DEFINE_boolean('use_att', False,
                            """whether to use attention mechanism""")
tf.app.flags.DEFINE_integer('cluster_num', 128,
                            """num of cluster in attention mechanism""")
tf.app.flags.DEFINE_integer('att_nodes', 1024,
                            """num of nodes in attention layer""")

def main():
    # load train dataset
    pre_process = MinMaxNormalization01()
    print('load data...')
    data, timestamps = load_h5data('./data/NYC14_M16x8_T60_NewEnd.h5')
    print('preprocess data...')
    # data: [num, channel, row, col]
    data = pre_process.fit_transform(data)
    pre_index = max(FLAGS.closeness*1, FLAGS.period*7, FLAGS.trend*7*24)

    train_data = data[:-FLAGS.test_num]
    train_timestamps = timestamps[:-FLAGS.test_num]
    test_data = data[-pre_index-FLAGS.test_num:]
    test_timestamps = timestamps[-pre_index-FLAGS.test_num:]
    print('get batch data...')
    train_x, train_y = batch_data_cpt_ext(train_data, train_timestamps, batch_size=FLAGS.batch_size, close=FLAGS.closeness, period=FLAGS.period, trend=FLAGS.trend)
    test_x, test_y = batch_data_cpt_ext(test_data, test_timestamps, batch_size=FLAGS.batch_size, close=FLAGS.closeness, period=FLAGS.period, trend=FLAGS.trend)
    # train_x = x[:-FLAGS.test_num]
    # test_x = x[-FLAGS.test_num:]
    # train_y = y[:-FLAGS.test_num]
    # test_y = y[-FLAGS.test_num:]  
    #print(len(train_y))
    #print(len(test_y))
    train = {'x': train_x, 'y': train_y}
    test = {'x': test_x, 'y': test_y}
    nb_flow = data.shape[1]
    row = data.shape[2]
    col = data.shape[3]
    print('build ResNet model...')
    model = ResNet(input_conf=[[FLAGS.closeness,nb_flow,row,col],[FLAGS.period,nb_flow,row,col],
        [FLAGS.trend,nb_flow,row,col],[8]], batch_size=FLAGS.batch_size, 
        layer=['conv', 'res_net', 'conv'],
        layer_param = [ [[3,3], [1,1,1,1], 64],
        [ 3, [ [[3,3], [1,1,1,1], 64], [[3,3], [1,1,1,1], 64] ] ],
        [[3,3], [1,1,1,1], 2] ])
    print('model solver...')
    solver = ModelSolver(model, train, train, preprocessing=pre_process,
            n_epochs=FLAGS.n_epochs, 
            batch_size=FLAGS.batch_size, 
            update_rule=FLAGS.update_rule,
            learning_rate=FLAGS.lr, save_every=FLAGS.save_every, 
            pretrained_model=None, model_path='model_save/ResNet/', 
            test_model='model_save/ResNet/model-'+str(FLAGS.n_epochs), log_path='log/ResNet/', 
            cross_val=True)

    print('begin training...')
    solver.train()
    print('begin testing for predicting next 1 step')
    solver.test(test)
    # test 1 to n
    print('begin testing for predicting next %d steps', FLAGS.output_steps)
    test_n = {'data': test_data, 'timestamps': test_timestamps}
    solver.test_1_to_n(test_n, n=FLAGS.output_steps, close=FLAGS.closeness, period=FLAGS.period, trend=FLAGS.trend)

if __name__ == "__main__":
    main()
