import numpy as np
import tensorflow as tf
import os
import sys
from sklearn.cluster import KMeans
from solver import ModelSolver
# for mac debug
sys.path.append('/Users/frances/Documents/DeepLearning/Code/TaxiPrediction/model/')
sys.path.append('/Users/frances/Documents/DeepLearning/Code/TaxiPrediction/util/')
# for server running
sys.path.append('/home/zx/TaxiPrediction/model/')
sys.path.append('./util/')
sys.path.append('./data/')
from ConvLSTM import *
from autoencoder import *
from AttConvLSTM import *
from ResNet import *
from preprocessing import *
from utils import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('gpu', '0',
                            """which gpu to use: 0 or 1""")

tf.app.flags.DEFINE_integer('input_steps', 10,
                            """num of input_steps""")
tf.app.flags.DEFINE_integer('output_steps', 10,
                            """num of output_steps""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """batch size for training""")
tf.app.flags.DEFINE_integer('n_epochs', 20,
                            """num of epochs""")
tf.app.flags.DEFINE_float('keep_prob', .9,
                            """for dropout""")
tf.app.flags.DEFINE_float('lr', .0002,
                            """for dropout""")
tf.app.flags.DEFINE_string('update_rule', 'adam',
                            """update rule""")
tf.app.flags.DEFINE_integer('save_every', 1,
                            """steps to save""")
# model: ConvLSTM, AttConvLSTM, ResNet
tf.app.flags.DEFINE_string('model', 'ResNet',
                            """which model to train and test""")
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
                            """num of cluster in attention mechanism""")
tf.app.flags.DEFINE_integer('att_nodes', 1024,
                            """num of nodes in attention layer""")
tf.app.flags.DEFINE_integer('use_ae', 1,
                            """whether to use autoencoder to cluster""")
tf.app.flags.DEFINE_string('ae_pretrain', None,
                           """pretrained_model for autoencoder""")
# train/test
tf.app.flags.DEFINE_integer('train', 1,
                            """whether to train""")
tf.app.flags.DEFINE_integer('test', 0,
                            """whether to test""")
tf.app.flags.DEFINE_string('pretrained_model', None,
                           """pretrained_model_name""")

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    # preprocessing class
    pre_process = MinMaxNormalization01()
    print('load train, validate, test data...')
    split = [17520, 4416, 4368]
    data, train_data, val_data, test_data = load_npy_data(filename=['data/citybike/p_map.npy', 'data/citybike/d_map.npy'], split=split)
    # data: [num, row, col, channel]
    print('preprocess train data...')
    pre_process.fit(train_data)
    
    if FLAGS.model=='ResNet':
        pre_index = max(FLAGS.closeness*1, FLAGS.period*7, FLAGS.trend*7*24)
        all_timestamps = gen_timestamps(['2013','2014','2015','2016'])
        all_timestamps = all_timestamps[4344:-4416]
        data = pre_process.transform(data)
        # train_data = train_data
        train_data = data[:split[0]]
        val_data = data[split[0]-pre_index:split[0]+split[1]]
        test_data = data[split[0]+split[1]-pre_index:split[0]+split[1]+split[2]]
        # get train, validate, test timestamps
        train_timestamps = all_timestamps[:split[0]]
        val_timestamps = all_timestamps[split[0]-pre_index:split[0]+split[1]]
        test_timestamps = all_timestamps[split[0]+split[1]-pre_index:split[0]+split[1]+split[2]]
        # get x, y
        train_x, train_y = batch_data_cpt_ext(train_data, train_timestamps, 
            batch_size=FLAGS.batch_size, close=FLAGS.closeness, period=FLAGS.period, trend=FLAGS.trend)
        val_x, val_y = batch_data_cpt_ext(val_data, val_timestamps, 
            batch_size=FLAGS.batch_size, close=FLAGS.closeness, period=FLAGS.period, trend=FLAGS.trend)
        test_x, test_y = batch_data_cpt_ext(test_data, test_timestamps, 
            batch_size=FLAGS.batch_size, close=FLAGS.closeness, period=FLAGS.period, trend=FLAGS.trend)
        train = {'x': train_x, 'y': train_y}
        val = {'x': val_x, 'y': val_y}
        test = {'x': test_x, 'y': test_y}
        nb_flow = train_data.shape[-1]
        row = train_data.shape[1]
        col = train_data.shape[2]
        print('build ResNet model...')
        model = ResNet(input_conf=[[FLAGS.closeness,nb_flow,row,col],[FLAGS.period,nb_flow,row,col],
            [FLAGS.trend,nb_flow,row,col],[8]], batch_size=FLAGS.batch_size, 
            layer=['conv', 'res_net', 'conv'],
            layer_param = [ [[3,3], [1,1,1,1], 64],
            [ 3, [ [[3,3], [1,1,1,1], 64], [[3,3], [1,1,1,1], 64] ] ],
            [[3,3], [1,1,1,1], 2] ])
        print('model solver...')
        solver = ModelSolver(model, train, val, preprocessing=pre_process,
                n_epochs=FLAGS.n_epochs, 
                batch_size=FLAGS.batch_size, 
                update_rule=FLAGS.update_rule,
                learning_rate=FLAGS.lr, save_every=FLAGS.save_every, 
                pretrained_model=FLAGS.pretrained_model, model_path='citybike-results/model_save/ResNet/',
                test_model='citybike-results/model_save/ResNet/model-'+str(FLAGS.n_epochs), log_path='citybike-results/log/ResNet/', 
                cross_val=False, cpt_ext=True)
        print('begin training...')
        test_n = {'data': test_data, 'timestamps': test_timestamps}
        _, test_prediction = solver.train(test, test_n, output_steps=FLAGS.output_steps)
        # get test_target and test_prediction
        i = pre_index
        test_target = []
        while i<len(test_data)-FLAGS.output_steps:
            test_target.append(test_data[i:i+FLAGS.output_steps])
            i+=1
        test_target = np.asarray(test_target)
        #np.save('results/ResNet/test_target.npy', test_target)
        #np.save('results/ResNet/test_prediction.npy', test_prediction)
        #print('begin testing for predicting next 1 step')
        #solver.test(test)
        # test 1 to n
        #print('begin testing for predicting next '+str(FLAGS.output_steps)+' steps')
        #test_n = {'data': test_data, 'timestamps': test_timestamps}
        #solver.test_1_to_n(test_n, n=FLAGS.output_steps, close=FLAGS.closeness, period=FLAGS.period, trend=FLAGS.trend)
    else:
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
        if FLAGS.model=='ConvLSTM':
            print('build ConvLSTM model...')
            model = ConvLSTM(input_dim=input_dim, batch_size=FLAGS.batch_size, 
                layer={'encoder': ['conv', 'conv', 'conv_lstm', 'conv_lstm'], 
                'decoder': ['conv_lstm', 'conv_lstm', 'conv', 'conv']}, 
                layer_param={'encoder': [ [[3,3], [1,1,1,1], 8], 
                [[3,3], [1,1,1,1], 16], 
                [[16,16], [3,3], 64], 
                [[16,16], [3,3], 64] ],
                'decoder': [ [[16,16], [3,3], 64], 
                [[16,16], [3,3], 64], 
                [[3,3], [1,1,1,1], 8], 
                [[3,3], [1,1,1,1], 2] ]}, 
                input_steps=10, output_steps=10)
            print('model solver...')
            solver = ModelSolver(model, train, val, preprocessing=pre_process,
                n_epochs=FLAGS.n_epochs, 
                batch_size=FLAGS.batch_size, 
                update_rule=FLAGS.update_rule,
                learning_rate=FLAGS.lr, save_every=FLAGS.save_every, 
                pretrained_model=FLAGS.pretrained_model, model_path='citybike-results/model_save/ConvLSTM/',
                test_model='citybike-results/model_save/ConvLSTM/model-'+str(FLAGS.n_epochs), log_path='citybike-results/log/ConvLSTM/')
        elif FLAGS.model=='AttConvLSTM':
            # train_data: [num, row, col, channel]
            if FLAGS.use_ae:
                # auto-encoder to cluster train_data
                print('auto-encoder to cluster...')
                ae = AutoEncoder(input_dim=input_dim, z_dim=[4, 4, 16],
                                 layer={'encoder': ['conv', 'conv'],
                                        'decoder': ['conv', 'conv']},
                                 layer_param={'encoder': [[[3,3], [1,2,2,1], 8],
                                                           [[3,3], [1,2,2,1], 16]],
                                              'decoder': [[[3,3], [1,2,2,1], 8],
                                                          [[3,3], [1,2,2,1], 2]]})
                model_path = 'citybike-results/model_save/AEAttConvLSTM/'
                log_path = 'citybike-results/log/AEAttConvLSTM/'
                #ae.train(train_data, batch_size=FLAGS.batch_size, learning_rate=FLAGS.lr, n_epochs=20, model_save_path=model_path, pretrained_model=FLAGS.ae_pretrain)
                train_z_data = ae.get_z(train_data)
                # k-means to cluster train_z_data
                vector_data = np.reshape(train_z_data, (train_data.shape[0], -1))
                kmeans = KMeans(n_clusters=FLAGS.cluster_num, init='random', n_init=FLAGS.kmeans_run_num,
                                tol=0.00000001).fit(vector_data)
                cluster_centroid = kmeans.cluster_centers_
                # reshape to [cluster_num, row, col, channel]
                cluster_centroid = np.reshape(cluster_centroid,
                                              [-1, train_z_data.shape[1], train_z_data.shape[2], train_z_data.shape[3]])
                # decoder to original space
                cluster_centroid = ae.get_y(cluster_centroid)
            else:
                # k-means to cluster train_data
                print('k-means to cluster...')
                vector_data = np.reshape(train_data, (train_data.shape[0], -1))
                #init_vectors = vector_data[:FLAGS.cluster_num, :]
                #cluster_centroid = init_vectors
                kmeans = KMeans(n_clusters=FLAGS.cluster_num, init='random', n_init=FLAGS.kmeans_run_num, tol=0.00000001).fit(vector_data)
                cluster_centroid = kmeans.cluster_centers_
                # reshape to [cluster_num, row, col, channel]
                cluster_centroid = np.reshape(cluster_centroid, (-1, train_data.shape[1], train_data.shape[2], train_data.shape[3]))
                model_path = 'citybike-results/model_save/AttConvLSTM/'
                log_path = 'citybike-results/log/AttConvLSTM/'
            # build model
            print('build AttConvLSTM model...')
            model = AttConvLSTM(input_dim=input_dim, 
                att_inputs=cluster_centroid, att_nodes=FLAGS.att_nodes, 
                batch_size=FLAGS.batch_size, 
                layer={'encoder': ['conv', 'conv', 'conv_lstm', 'conv_lstm'], 
                'decoder': ['conv_lstm', 'conv_lstm', 'conv', 'conv'],
                'attention': ['conv', 'conv']}, 
                layer_param={'encoder': [ [[3,3], [1,1,1,1], 8], 
                [[3,3], [1,1,1,1], 16], 
                [[16,16], [3,3], 64], 
                [[16,16], [3,3], 64] ],
                'decoder': [ [[16,16], [3,3], 64], 
                [[16,16], [3,3], 64], 
                [[3,3], [1,1,1,1], 8], 
                [[3,3], [1,1,1,1], 2] ],
                'attention': [ [[3,3], [1,1,1,1], 8], 
                [[3,3], [1,1,1,1], 16] ]}, 
                input_steps=10, output_steps=10)
            print('model solver...')
            solver = ModelSolver(model, train, val, preprocessing=pre_process,
                n_epochs=FLAGS.n_epochs, 
                batch_size=FLAGS.batch_size, 
                update_rule=FLAGS.update_rule,
                learning_rate=FLAGS.lr, save_every=FLAGS.save_every, 
                pretrained_model=FLAGS.pretrained_model, model_path=model_path,
                test_model=model_path+'model-'+str(FLAGS.n_epochs), log_path=log_path)
        if FLAGS.train:
            print('begin training...')
            test_prediction, _ = solver.train(test)
            test_target = np.asarray(test_y)
        if FLAGS.test:
            print('test trained model...')
            solver.test_model = solver.model_path + FLAGS.pretrained_model
            test_prediction = solver.test(test)
            test_target = np.asarray(test_y)
    # np.save('citybike-results/results/'+FLAGS.model+'/test_target.npy', test_target)
    # np.save('citybike-results/results/'+FLAGS.model+'/test_prediction.npy', test_prediction)

if __name__ == "__main__":
    main()
