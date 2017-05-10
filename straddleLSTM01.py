import numpy as np 
import tensorflow as tf 

class Model:
    """docstring for Model"""
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self._prediction = None
        self._optimize = None
        self._error = None
        self.prediction
        self.optimize
        self.error

    @property
    def prediction(self):
        if self._prediction is None:
            data_size = int(self.data.get_shape()[1])
            target_size = int(self.target.get_shape()[1])
            num_neurons = 100
            num_layers = 4

            cell = tf.nn.rnn_cell.LSTMCell(num_neurons)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

            val, _ = tf.nn.dynamic_rnn(cell, self.data, dtype=tf.float32)
            val = tf.transpose(val, [1, 0, 2])
            last = tf.gather(val, int(val.get_shape()[0]) - 1)
            W = tf.Variable(tf.truncated_normal([num_neurons, int(self.target.get_shape()[1])]))
            b = tf.Variable(tf.constant(0.1, shape=[self.target.get_shape()[1]]))
            self._prediction = tf.matmul(last, W) + b
        return self._prediction
        
    @property
    def optimize(self):
        if self._optimize is None:
            optimizer = tf.train.AdamOptimizer()
            self._optimize = optimizer.minimize(self.error)
        return self._optimize   

    @property
    def error(self):
        if self._error is None:
            self._error = tf.reduce_mean(tf.squared_difference(self.target, self.prediction)) 
        return self._error


def main():
    input_array = np.load("straddle_input_np.npy")
    output_array = np.load("norm_straddle_labels.npy")
    print("data loaded")
    perm = np.random.permutation(input_array.shape[0])
    input_array = input_array[perm]
    output_array = output_array[perm]
    TEST_SIZE = 5000
    in_train = input_array[TEST_SIZE:]
    in_test = input_array[:TEST_SIZE]
    out_train = output_array[TEST_SIZE:]
    out_test = output_array[:TEST_SIZE]

    data = tf.placeholder(tf.float32, [None, 500, 100])
    target = tf.placeholder(tf.float32, [None, 1])
    model = Model(data, target)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    batch_size = 300
    num_batches = int(len(in_train) / batch_size)
    epoch = 30
    for i in range(epoch):
        ptr = 0
        for j in range(num_batches):
            inp, out = in_train[ptr:ptr+batch_size], out_train[ptr:ptr+batch_size]
            ptr+=batch_size
            _, error = sess.run([model.optimize, model.error], {data: inp, target: out})
            if j == num_batches - 1:
                print("Error - ", str(error))
        print("Epoch - ", str(i))
    saver = tf.train.Saver()
    save = saver.save(sess, "straddleLSTM01_model")
    print("saved in ", save) 
    fin_error = sess.run(model.error, {data: in_test, target: out_test})
    print("Final Error - ", fin_error)   
    sess.close()

if __name__ == "__main__":
    main()