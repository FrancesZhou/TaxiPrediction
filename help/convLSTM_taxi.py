
import os.path
import time

import numpy as np
import tensorflow as tf
import scipy.io as sio

import layer_def as ld
import BasicConvLSTMCell

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './checkpoints/train_store_convLSTM_taxi',
                            """dir to store trained net""")
tf.app.flags.DEFINE_integer('seq_length', 20,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('seq_start', 10,
                            """ start of seq generation""")
tf.app.flags.DEFINE_integer('max_step', 200000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', .9,
                            """for dropout""")
tf.app.flags.DEFINE_float('lr', .000001,
                            """for dropout""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .1,
                            """weight init for fully connected layers""")

tf.app.flags.DEFINE_integer('input_height', 64,
                            """height of input frame""")
tf.app.flags.DEFINE_integer('input_width', 64,
                            """width of input frame""")
tf.app.flags.DEFINE_integer('input_channel', 1,
                            """channels of input frame""")




# load data to create train/validation/test data
def load_data(filename_train,filename_validate,filename_test):
  data_train = sio.loadmat(filename_train)
  train_data = data_train['p_map_train']

  data_validate = sio.loadmat(filename_validate)
  validate_data = data_validate['p_map_validate']

  data_test = sio.loadmat(filename_test)
  test_data = data_test['p_map_test']
  # data: [num, input_height, input_width, input_channel]

  return [train_data[:,:,:,np.newaxis], validate_data[:,:,:,np.newaxis], test_data[:,:,:,np.newaxis]]

# create batch data for train/validation/test data
def get_batch_data(data):
  #target = []
  batch_data = []
  # batch_data: [seq_length, batch_size, input_height, input_width, input_channel]
  
  # for i in xrange(FLAGS.batch_size):
  #     target.append(random.randint(0,data.size))
  target = np.random.randint(data.size-FLAGS.seq_length, size=FLAGS.batch_size)
  for i in xrange(FLAGS.seq_length):
      inputs = []
      # inputs: [batch_size, input_height, input_width, input_channel]
      for j in target:
        p = data[i+j]
        inputs.append(p)
      batch_data.append(inputs)

  batch_data = np.asarray(batch_data)
  batch_data = np.transpose(batch_data, [1,0,2,3,4])
  # batch_data: [batch_size, seq_length, input_height, input_width, input_channel]

  return batch_data

def get_batch_validate_and_test_data(data, index):
  batch_data = []
  target = np.arange(index,index+FLAGS.batch_size)
  for i in xrange(FLAGS.seq_length):
    inputs = []
    for j in target:
      p = data[i+j]
      inputs.append(p)
    batch_data.append(inputs)
batch_data = np.asarray(batch_data)
batch_data = np.transpose(batch_data, [1,0,2,3,4])

def train(train_data, validate_data, test_data):
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs
    x = tf.placeholder(tf.float32, [None, FLAGS.seq_length, FLAGS.input_height, FLAGS.input_width, FLAGS.input_channel])

    # possible dropout inside
    #keep_prob = tf.placeholder("float")
    #x_dropout = tf.nn.dropout(x, keep_prob)

    # create network
    x_unwrap = []

    with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
      # BasicConvLSTMCell: (shape, filter_size, num_features, state_is_tuple)
      cell_1 = BasicConvLSTMCell.BasicConvLSTMCell([16,16], [3,3], 64, state_is_tuple=True)
      # new_state: [batch_size, 16, 16, 64] for c and h
      new_state_1 = cell_1.zero_state(FLAGS.batch_size)

      # cell_2
      cell_2 = BasicConvLSTMCell.BasicConvLSTMCell([16,16], [3,3], 64, state_is_tuple=True)
      new_state_2 = cell_2.zero_state(FLAGS.batch_size)

      # cell_3
      cell_3 = BasicConvLSTMCell.BasicConvLSTMCell([16,16], [3,3], 64, state_is_tuple=True)
      #new_state_3 = cell_3.zero_state(FLAGS.batch_size)

      # cell_3
      cell_4 = BasicConvLSTMCell.BasicConvLSTMCell([16,16], [3,3], 64, state_is_tuple=True)
      #new_state_4 = cell_3.zero_state(FLAGS.batch_size)

    # conv network
    # conv_layer: (input, kernel_size, stride, num_features, scope_name_idx, linear, reuseL)
    # BasicConvLSTMCell: __call__(inputs, state, scope, reuseL)
    reuseL = None
    with tf.variable_scope(tf.get_variable_scope()) as scope:
      for i in xrange(FLAGS.seq_length-1):
        # -------------------------- encode ------------------------
        # conv1
        # [16, 64, 64, 1] -> [16, 32, 32, 8]
        if i < FLAGS.seq_start:
          conv1 = ld.conv_layer(x[:,i,:,:,:], 3, 2, 8, "encode_1", reuseL=reuseL)
        else:
          conv1 = ld.conv_layer(x[:,i,:,:,:], 3, 2, 8, "encode_1", reuseL=reuseL)
        # conv2
        # [16, 32, 32, 8] -> [16, 16, 16, 16]
        print conv1.get_shape().as_list()
        conv2 = ld.conv_layer(conv1, 3, 2, 16, "encode_2", reuseL=reuseL)

        # convLSTM 3
        # [16, 16, 16, 16] -> [16, 16, 16, 64]
        print conv2.get_shape().as_list()
        y_0 = conv2
        print y_0.get_shape().as_list()
        y_1, new_state_1 = cell_1(y_0, new_state_1, "encode_3_convLSTM_1", reuseL=reuseL)
        # convLSTM 4
        # [16, 16, 16, 64] -> [16, 16, 16, 64]
        print y_1.get_shape().as_list()
        y_2, new_state_2 = cell_2(y_1, new_state_2, "encode_4_convLSTM_2", reuseL=reuseL)
        print y_2.get_shape().as_list()

        # ------------------------ decode -------------------------
        # convLSTM 5
        # [16, 16, 16, 64] -> [16, 16, 16, 64]
        # copy the initial states and cell outputs from convLSTM 4
        if i==0:
          new_state_3 = new_state_2
        y_3, new_state_3 = cell_3(y_2, new_state_3, "encode_5_convLSTM_3", reuseL=reuseL)
        # convLSTM 6
        # [16, 16, 16, 64] -> [16, 16, 16, 64]
        print y_3.get_shape().as_list()
        if i==0:
          new_state_4 = new_state_1
        y_4, new_state_4 = cell_4(y_3, new_state_4, "encode_6_convLSTM_4", reuseL=reuseL)
        print y_4.get_shape().as_list()
        # conv7
        # [16, 16, 16, 64] -> [16, 32, 32, 8]
        conv6 = y_4
        conv7 = ld.transpose_conv_layer(conv6, 3, 2, 8, "decode_7", reuseL=reuseL)
        # x_1 
        # [16, 32, 32, 8] -> [16, 64, 64, 1]
        print conv7.get_shape().as_list()
        x_1 = ld.transpose_conv_layer(conv7, 3, 2, 1, "decode_8", linear=True, reuseL=reuseL) # set activation to linear
        if i >= FLAGS.seq_start-1:
          x_unwrap.append(x_1)
        # set reuse to true after first go
        if i == 0:
          #tf.get_variable_scope().reuse_variables()
          reuseL = True

    # stack them all together 
    x_unwrap = tf.stack(x_unwrap)
    x_unwrap = tf.transpose(x_unwrap, [1,0,2,3,4])

    # calc total loss (compare x_t to x_t+1)
    loss = tf.nn.l2_loss(x[:,FLAGS.seq_start:,:,:,:] - x_unwrap[:,:,:,:,:])
    tf.summary.scalar('loss', loss)

    # training
    train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
    
    # List of all Variables
    #variables = tf.global_variables()

    # Build a saver
    saver = tf.train.Saver(tf.global_variables())   

    # Summary op
    summary_op = tf.summary.merge_all()
 
    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # --------------------- Start running operations on the Graph ------------------------
    sess = tf.Session()

    # init if this is the very time training
    print("init network from scratch")
    sess.run(init)

    # Summary op
    #graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=sess.graph)

    # ------------------------------- training for train_data -------------------------------------
    print("now training step:")

    for step in xrange(FLAGS.max_step):
      dat = get_batch_data(train_data)
      t = time.time()
      _, loss_r = sess.run([train_op, loss],feed_dict={x:dat})
      elapsed = time.time() - t

      if step%100 == 0 and step != 0:
        summary_str = sess.run(summary_op, feed_dict={x:dat})
        summary_writer.add_summary(summary_str, step) 
        #print("time per batch is " + str(elapsed))
        print("at step " + str(step) + ", loss is " + str(loss_r))
        #print(loss_r)
      
      assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

      if step%1000 == 0 and step != 0:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + FLAGS.train_dir)

        # -------------------------- validation for each 1000 steps ---------------------
        print("at step " + str(step) + ", validate...")
        v_whole_loss = 0
        # [all_size, seq_length, in_height, in_width, in_channel]
        all_batch_v = np.zeros((validate_data.shape[0]-FLAGS.seq_length, FLAGS.seq_length, FLAGS.input_height, FLAGS.input_width, FLAGS.input_channel), dtype=np.float32)
        #all_batch_v = validate_data[v_i, v_i:v_i+FLAGS.seq_length, :, :, :]
        #while v_i <validate_data.shape[0]:
        for v_i in xrange(validate_data.shape[0]-FLAGS.seq_length):
          all_batch_v[v_i] = validate_data[v_i:v_i+FLAGS.seq_length, :, :, :]
        v_i = 0
        while v_i < all_batch_v.shape[0]-FLAGS.batch_size:
          # [batch_size, seq_length, in_height, in_width, channel]
          dat_v = all_batch_v[v_i:v_i+FLAGS.batch_size, :, :, :, :]
          v_loss = sess.run([loss], feed_dict={x:dat_v})
          v_whole_loss += v_loss
        print("validation loss: " + str(v_whole_loss))

    # ---------------------------- for test data -------------------------------------
    print("now testing step:")
    t_whole_loss = 0
    all_batch_t = np.zeros((test_data.shape[0]-FLAGS.seq_length, FLAGS.seq_length, FLAGS.input_height, FLAGS.input_width, FLAGS.input_channel), dtype=np.float32)
    for t_i in xrange(test_data.shape[0]-FLAGS.seq_length):
      all_batch_t[t_i] = test_data[t_i:t_i+FLAGS.seq_length, :, :, :]
    t_i = 0
    while t_i < all_batch_t.shape[0]-FLAGS.batch_size:
      dat_t = all_batch_t[t_i:t_i+FLAGS.batch_size, :, :, :, :]
      predict, t_loss = sess.run([x_unwrap, loss], feed_dict={x:dat_t})
      t_whole_loss += t_loss
    print("test loss: " + str(t_whole_loss))
    np.save('prediction.npy',pred)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train_data, validate_data, test_data = load_data('p_map_train.mat','p_map_validate.mat','p_map_test.mat')
  train(train_data, validate_data, test_data)

if __name__ == '__main__':
  tf.app.run()


