import os
import cv2
import json
import imageio
import numpy as np
import tensorflow as tf
from pydoc import locate
import constants as const
from nets import nn_utils
from utils import os_utils
from utils import tf_utils
from utils import log_utils
from utils import heatmap_utils
from nets import attention_filter
from config.base_config import BaseConfig
from utils.imagenet_lbls import imagenet_lbls

def normalize_filter(filter_type,_atten_var,filter_height,filter_width):
    if filter_type == 'l2norm':
        frame_mask = np.reshape(np.abs(_atten_var), (filter_height, filter_width))
        # frame_mask = np.reshape(_atten_var, (filter_height, filter_width))
        frame_mask = frame_mask / np.linalg.norm(frame_mask)
    elif filter_type == 'softmax':
        frame_mask = tf.nn.softmax(np.reshape(_atten_var, [1, -1])).eval()
        frame_mask = np.reshape(frame_mask, (filter_height, filter_width))
    elif filter_type == 'gauss':
        # print(_atten_var)
        mu = _atten_var
        cov = [[1.0, 0], [0, 1.0]]
        frame_mask = attention_filter.gaussian_kernel(3, mu, cov).eval()
        frame_mask = frame_mask[:, :, 0]
    else:
        raise NotImplementedError('Invalid filter type {}'.format(filter_type))

    return frame_mask

def main(cfg):

    # cfg.num_classes = 1001
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    output_dir = cfg.output_dir

    os_utils.touch_dir(output_dir)

    args_file = os.path.join(cfg.output_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(cfg), f, ensure_ascii=False, indent=2, sort_keys=True)

    log_file = os.path.join(cfg.output_dir, cfg.log_filename + '.txt')

    logger = log_utils.create_logger(log_file)

    img_name_ext = cfg.img_name
    img_name,_ = os.path.splitext(img_name_ext)
    datasets_dir = './input_imgs'
    test_img = imageio.imread('{}/{}'.format(datasets_dir,img_name_ext))
    test_img = cv2.resize(test_img,(const.frame_height, const.frame_height))
    with tf.Graph().as_default():

        images_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, const.frame_height, const.frame_height,
                                                       const.num_channels), name='input_img')
        lbls_ph = tf.compat.v1.placeholder(tf.int32, shape=(None, cfg.num_classes), name='class_lbls')
        logits_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, cfg.num_classes), name='logits_lbls')
        per_class_logits_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, cfg.num_classes), name='logits_lbls')
        input_ph = nn_utils.adjust_color_space(images_ph,cfg.preprocess_func)
        network_class = locate(cfg.network_name)
        model = network_class(cfg, images_ph=input_ph, lbls_ph=lbls_ph)

        pre_atten_feat_map_tf = tf.compat.v1.get_default_graph().get_tensor_by_name(cfg.replicate_net_at)
        pre_atten_feat_map_tf_shape = pre_atten_feat_map_tf.shape
        sub_feat_map_ph = tf.compat.v1.placeholder(tf.float32,
                                         shape=[None, pre_atten_feat_map_tf_shape[1], pre_atten_feat_map_tf_shape[2],
                                                pre_atten_feat_map_tf_shape[3]], name='feat_map_input')
        sub_network_class = locate(cfg.sub_network_name)
        sub_model = sub_network_class(cfg, images_ph=sub_feat_map_ph, lbls_ph=lbls_ph)
        sub_logits = sub_model.val_logits


        logits = model.val_logits


        sess = tf.compat.v1.InteractiveSession()

        atten_filter_position = cfg.atten_filter_position

        tf_atten_var = [v for v in tf.compat.v1.global_variables() if atten_filter_position.format('atten') in v.name][-1]
        ## Didn't make a difference for tf_atten_var becuase tf_atten_var is created using get_varibale, i.e., shared
        tf_gate_atten_var = [v for v in tf.compat.v1.global_variables() if atten_filter_position.format('gate') in v.name][-1]
        # print(tf_gate_atten_var)
        # optimizer = tf.train.AdamOptimizer(0.01)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        logger.info('Learning rate {} {}'.format(cfg.learning_rate,cfg.max_iters))
        learning_rate = tf_utils.poly_lr(global_step, cfg)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)

        class_specific = True if cfg.caf_variant == 'cls_specific' else False
        if class_specific:
            logger.info('Solving class specific optimization problem -- classification network')

            mult_logits_2 = per_class_logits_ph * sub_logits
            loss_sub = tf.reduce_sum(mult_logits_2)
            grads = optimizer.compute_gradients(loss_sub, var_list=[tf_atten_var])
            train_op = optimizer.apply_gradients(grads, global_step=global_step)
        else:
            raise NotImplementedError('cls_oblivious version implemented yet')


        # train_op = optimizer.minimize(loss, var_list=[tf_atten_var])
        tf.compat.v1.global_variables_initializer().run()
        ckpt_file = tf.train.latest_checkpoint(output_dir)
        logger.info('Model Path {}'.format(ckpt_file))
        saver = tf.compat.v1.train.Saver()  # saves variables learned during training
        load_model_msg = model.load_model(output_dir,ckpt_file,sess,saver, load_logits=True)
        logger.info(load_model_msg)

        class_predictions,ground_logits = sess.run([model.val_class_prediction,logits],
                                                                feed_dict={images_ph:np.expand_dims(test_img,0)})


        class_predictions = class_predictions[0]
        # print('Class Prediction {}'.format(imagenet_lbls[class_predictions]))

        k = 1
        top_k = np.argsort(np.squeeze(ground_logits))[::-1][:k]
        # top_k = [235,282,94,1,225]
        logger.info('Top K={} {}'.format(k, [imagenet_lbls[i] for i in top_k]))


        filter_type = cfg.filter_type
        if filter_type == 'gauss':
            rand_initilzalier = np.random.normal(0, 1, (tf_atten_var.shape[0], tf_atten_var.shape[1]))
        else:
            rand_initilzalier = np.random.normal(0, 1, (tf_atten_var.shape[0], tf_atten_var.shape[1], 1))

        close_gate = tf.compat.v1.assign(tf_gate_atten_var, False)
        open_gate = tf.compat.v1.assign(tf_gate_atten_var, True)
        random_init = tf.compat.v1.assign(tf_atten_var, rand_initilzalier)
        lr_reset = tf.compat.v1.assign(global_step, 0)
        MAX_INT = np.iinfo(np.int16).max
        # output_dir = cfg.output_dir
        for top_i in top_k:
            # top_i  = 207  # To control which top_i to work on directly
            sess.run([open_gate, random_init, lr_reset])
            # sess.run(open_gate)

            iteration = 0
            prev_loss = MAX_INT
            event_gif_images = []
            per_class_maximization = np.ones((1,cfg.num_classes))
            per_class_maximization[0,top_i] = -1

            while iteration < cfg.max_iters:

                if iteration == 0:
                    sess.run([close_gate])
                    _pre_atten_feat_map_tf, _atten_var = sess.run(
                        [pre_atten_feat_map_tf, tf_atten_var],
                        feed_dict={
                            # sub_feat_map_ph: _pre_atten_feat_map_tf,
                            images_ph:  np.expand_dims(test_img, 0),
                            per_class_logits_ph: per_class_maximization
                        })
                    sess.run([open_gate])
                _atten_var, _sub_logits, _loss, _ = sess.run([tf_atten_var, sub_logits, loss_sub, train_op],
                                                             feed_dict={
                                                                 sub_feat_map_ph: _pre_atten_feat_map_tf,
                                                                 # images_ph:np.expand_dims(img_crops[crop_idx,:,:,:],0),
                                                                 per_class_logits_ph: per_class_maximization
                                                             })


                if iteration % 50 == 0:
                    logger.info('Iter {0:2d}: {1:.5f} Top {2:3d} {3}'.format(iteration, _loss,top_i,imagenet_lbls[top_i]))
                    # print(np.round(np.reshape(_atten_var,(7,7)),2))
                    if cfg.save_gif:
                        frame_mask = normalize_filter(filter_type,_atten_var,tf_atten_var.shape[0], tf_atten_var.shape[1])
                        if class_specific:
                            #
                            # heatmap_utils.save_heatmap(frame_mask,save=output_dir + img_name +'_msk_cls_{}_{}.png'.format(top_i,filter_type))
                            plt = heatmap_utils.apply_heatmap(test_img / 255.0, frame_mask, alpha=0.7,
                                                    save=output_dir + img_name +'_cls_{}_{}.png'.format(top_i,filter_type), axis='off', cmap='bwr')
                        else:
                            plt = heatmap_utils.apply_heatmap(test_img / 255.0, frame_mask, alpha=0.7,
                                                    save=output_dir + img_name +'_{}.png'.format(filter_type), axis='off', cmap='bwr')

                        fig = plt.gcf()
                        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                        w, h = fig.canvas.get_width_height()
                        data_img = data.reshape((h, w, 3))
                        event_gif_images.append(data_img)
                        # imageio.imwrite(dump_dir + '{}_test.jpg'.format(iteration),data_img)
                        plt.close()

                    if np.abs(_loss - prev_loss) < 10e-5:
                        break

                    prev_loss = _loss

                iteration+=1

            frame_mask = normalize_filter(filter_type, _atten_var, tf_atten_var.shape[0], tf_atten_var.shape[1])
            if class_specific:
                # imageio.imwrite(output_dir + img_name + '_msk_cls_{}_{}.png'.format(top_i, filter_type), frame_mask)
                heatmap_utils.apply_heatmap(test_img / 255.0, frame_mask, alpha=0.6,
                                                  save=output_dir + img_name + '_cls_{}_{}.png'.format(top_i,
                                                                                                       filter_type),
                                                  axis='off', cmap='bwr')
            else:
                heatmap_utils.apply_heatmap(test_img / 255.0, frame_mask, alpha=0.6,
                                                  save=output_dir + img_name + '_{}.png'.format(filter_type),
                                                  axis='off', cmap='bwr')
            if cfg.save_gif:
                if class_specific:
                    imageio.mimsave(output_dir + img_name+'_cls_{}_{}.gif'.format(top_i,atten_filter_position[:-2].format('').replace('/','')),event_gif_images, duration=1.0)
                else:
                    imageio.mimsave(output_dir + img_name + '_cls_{}_{}.gif'.format(filter_type,atten_filter_position[:-2].format('').replace('/','')), event_gif_images, duration=1.0)

                # break ## === TOP 1 always


if __name__ == '__main__':
    arg_db_name = 'imagenet'
    arg_net = 'densenet161' #[densenet161,inceptionv1,resnet50] ## For NOW only denseNet is supported
    args = [
        '--gpu', '0',
        '--output_dir',  './output_heatmaps/',
        '--db_name', arg_db_name,
        '--img_name', 'ILSVRC2012_val_00000021.JPEG', #[ILSVRC2012_val_00000021.JPEG,cute_dog.jpg]
        # '--print_filter_name',
        '--net', arg_net,
        '--caf_variant','cls_specific', #[cls_oblivious,cls_specific]
        '--learning_rate','0.5',
        '--max_iters','1000',
        '--filter_type','l2norm', #['l2norm,softmax,gauss]
        '--replicate_net_at','densenet161/dense_block4/conv_block24/concat:0',
        '--atten_filter_position','dense_block4/{}_conv_block24:0' # last conv DenseNet
    ]
    cfg = BaseConfig().parse(args)
    assert cfg.net == 'densenet161'
    main(cfg)