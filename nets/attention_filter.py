import tensorflow as tf


def gaussian_kernel(size,
                    mean,
                    std):
    """Makes 2D gaussian Kernel for convolution."""
    gaussian = tf.contrib.distributions.MultivariateNormalFullCovariance(
        loc=mean,
        covariance_matrix=std)

    _range = tf.range(start=-size, limit=size + 1, dtype=tf.float32)
    X,Y = tf.meshgrid(_range,_range)
    idx = tf.concat([tf.reshape(X, [-1, 1]), tf.reshape(Y, [-1, 1])], axis=1)
    gauss_kernel = tf.reshape(gaussian.prob(idx),[tf.shape(X)[0],tf.shape(X)[1],1])

    return gauss_kernel / tf.reduce_sum(gauss_kernel)

def add_attention_filter(net,end_point,verbose=False,filter_type=None):

    if filter_type == 'l2norm':
        atten_var = tf.compat.v1.get_variable("atten_" + end_point, [net.shape[1], net.shape[2], 1], dtype=tf.float32,
                                    initializer=tf.initializers.ones())
        if verbose:
            print(atten_var)
        atten_var_norm = (atten_var  / tf.norm(atten_var))

    elif filter_type == 'softmax':
        atten_var = tf.compat.v1.get_variable("atten_" + end_point, [net.shape[1], net.shape[2], 1], dtype=tf.float32,
                                    initializer=tf.initializers.ones())
        atten_var_norm = tf.nn.softmax(tf.reshape(atten_var,[1,-1]))
        atten_var_norm = tf.reshape(atten_var_norm ,[-1,net.shape[1], net.shape[2], 1])
    elif filter_type == 'gauss':
        if net.shape[1] != 7:
            print('Warning :: Gauss filter will skip all non 7x7 conv layers')
            return net
        mu = tf.compat.v1.get_variable("atten_" + end_point, [1,2], dtype=tf.float32,
                                    initializer=tf.initializers.zeros())
        cov = [[1.0, 0], [0, 1.0]]
        atten_var_norm = gaussian_kernel(3, mu, cov)
    else:
        raise NotImplementedError('Invalid filter type {}'.format(filter_type))

    atten_var_gate = tf.compat.v1.get_variable("gate_" + end_point, initializer=False,dtype=tf.bool)
    if verbose:
        print(atten_var_gate)
    net = tf.cond(atten_var_gate, lambda: tf.multiply(atten_var_norm, net), lambda: tf.identity(net))

    return net