import tensorflow as tf
# import configuration as config

def poly_lr(global_step,cfg):
    starter_learning_rate = cfg.learning_rate
    end_learning_rate = cfg.end_learning_rate
    decay_steps = cfg.max_iters

    learning_rate = tf.compat.v1.train.polynomial_decay(starter_learning_rate, global_step,
                                              decay_steps, end_learning_rate,
                                              power=1)
    return learning_rate