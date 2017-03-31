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
tf.app.flags.DEFINE_boolean('use_att', False,
                            """whether to use attention mechanism""")
tf.app.flags.DEFINE_integer('cluster_num', 128,
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
    print('load data...')
    data, timestamps = load_h5data('data/NYC14_M16x8_T60_NewEnd.h5')
    print('preprocess data...')
    # data: [num, row, col, channel]
    data = pre_process.fit_transform(data)
    print('get batch data...')
    x, y = batch_data_cpt_ext(data, timestamps, batch_size=FLAGS.batch_size, close=FLAGS.closeness, period=FLAGS.period, trend=FLAGS.trend)
    train_x = x[:-FLAGS.test_num]
    test_x = x[-FLAGLS.test_num:]
    train_y = y[:-FLAGS.test_num]
    test_y = y[-FLAGS.test_num]
    
    train = {'x': train_x, 'y': train_y}
    test = {'x': test_x, 'y': test_y}

    input_dim = [train_data.shape[1], train_data.shape[2], train_data.shape[3]]
    nb_flow = data.shape[-1]
    row = data.shape[1]
    col = data.shape[2]
    print('build ResNet model...')
    model = ResNet(input_conf=[[[FLAGS.closeness,nb_flow,row,col],[FLAGS.period,nb_flow,row,col],
        [FLAGS.trend,nb_flow,row,col],8]], batch_size=FLAGS.batch_size, 
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
            test_model='model_save/ResNet/model-'+str(FLAGS.n_epochs), log_path='log/ResNet/')

    print('begin training...')
    solver.train()
    solver.test(test)

if __name__ == "__main__":
    main()
