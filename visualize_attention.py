import os
import cv2
import json
import imageio
import numpy as np
import tensorflow as tf
from pydoc import locate
import constants as const
from nets import nn_utils
from utils import heatmap_utils
from utils import os_utils
from utils import tf_utils
from utils import log_utils
from nets import attention_filter
from config.base_config import BaseConfig
from utils.imagenet_lbls import imagenet_lbls

def normalize_filter(filter_type,_atten_var,filter_height,filter_width):
    if filter_type == 'l2norm':
        frame_mask = np.reshape(np.abs(_atten_var), (filter_height, filter_width))
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
    test_img = cv2.resize(test_img,(const.max_frame_size, const.max_frame_size))
    with tf.Graph().as_default():

        images_ph = tf.placeholder(tf.float32, shape=(None, const.max_frame_size, const.max_frame_size,
                                                       const.num_channels), name='context_input')
        lbls_ph = tf.placeholder(tf.int32, shape=(None, cfg.num_classes), name='class_lbls')
        logits_ph = tf.placeholder(tf.float32, shape=(None, cfg.num_classes), name='logits_lbls')

        per_class_logits_ph = tf.placeholder(tf.float32, shape=(None, cfg.num_classes), name='logits_lbls')

        input_ph = nn_utils.center_crop(images_ph,cfg.preprocess_func)
        print(input_ph)
        network_class = locate(cfg.network_name)
        print(network_class,cfg.preprocess_func)

        # vis_input = False
        # if vis_input:
        #     input_ph_in = attention_filter.add_attention_filter(input_ph,end_point='input',verbose=True)
        #     model = network_class(cfg, images_ph=input_ph_in, lbls_ph=lbls_ph)
        # else:
        model = network_class(cfg, images_ph=input_ph, lbls_ph=lbls_ph)


        logits = model.val_logits


        sess = tf.InteractiveSession()

        atten_filter_position = cfg.atten_filter_position
        # if cfg.net == 'mobile':
        #     atten_filter_position = '{}_Conv2d_13_pointwise:0' # mobilenet
        # elif cfg.net == 'densenet161':
        #     atten_filter_position = cfg.atten_filter_position
        # elif cfg.net == 'inc1':
        #     atten_filter_position = cfg.atten_filter_position  # inceptionV1

        tf_atten_var = [v for v in tf.global_variables() if atten_filter_position.format('atten') in v.name][-1]
        ## I have train and val siamese networks (Train do batch norm while val apply learned normalization)
        ## Didn't make a difference for tf_atten_var becuase tf_atten_var is created using get_varibale, i.e., shared
        tf_gate_atten_var = [v for v in tf.global_variables() if atten_filter_position.format('gate') in v.name][-1]
        print(tf_gate_atten_var)
        # optimizer = tf.train.AdamOptimizer(0.01)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        print('Learning rate {} {}'.format(cfg.learning_rate,cfg.max_iters))
        learning_rate = tf_utils.poly_lr(global_step, cfg)
        optimizer = tf.train.AdamOptimizer(learning_rate)

        class_specific = True
        if class_specific:
            mult_logits = per_class_logits_ph * logits
            loss = tf.reduce_sum(mult_logits)
            grads = optimizer.compute_gradients(loss, var_list=[tf_atten_var])
            train_op = optimizer.apply_gradients(grads, global_step=global_step)
        else:
            loss = tf.reduce_mean(tf.square(logits_ph - logits))
            grads = optimizer.compute_gradients(loss, var_list=[tf_atten_var])
            train_op = optimizer.apply_gradients(grads, global_step=global_step)

        # train_op = optimizer.minimize(loss, var_list=[tf_atten_var])
        tf.global_variables_initializer().run()
        ckpt_file = tf.train.latest_checkpoint(output_dir)
        logger.info('Model Path {}'.format(ckpt_file))
        saver = tf.train.Saver()  # saves variables learned during training
        load_model_msg = model.load_model(output_dir,ckpt_file,sess,saver, load_logits=True)
        logger.info(load_model_msg)

        class_predictions,ground_logits = sess.run([model.val_class_prediction,logits],
                                                                feed_dict={images_ph:np.expand_dims(test_img,0)})


        class_predictions = class_predictions[0]
        # print('Class Prediction {}'.format(imagenet_lbls[class_predictions]))

        k = 1
        top_k = np.argsort(np.squeeze(ground_logits))[::-1][:k]
        # top_k = [235,282,94,1,225]
        print('Top K={} {}'.format(k, [imagenet_lbls[i] for i in top_k]))

        # k = 1
        # top_k = np.argsort(np.squeeze(ground_logits))[::-1][:k]
        # print('Top K={} {}'.format(k,[imagenet_lbls[i] for i in top_k ]))

        filter_type = cfg.filter_type
        if filter_type == 'gauss':
            rand_initilzalier = np.random.normal(0, 1, (tf_atten_var.shape[0], tf_atten_var.shape[1]))
        else:
            rand_initilzalier = np.random.normal(0, 1, (tf_atten_var.shape[0], tf_atten_var.shape[1], 1))

        # close_gate = tf.assign(tf_gate_atten_var, False)
        open_gate = tf.assign(tf_gate_atten_var, True)
        random_init = tf.assign(tf_atten_var, rand_initilzalier)
        lr_reset = tf.assign(global_step, 0)

        # output_dir = cfg.output_dir
        for top_i in top_k:
            # top_i  = 207  # To control which top_i to work on directly
            sess.run([open_gate, random_init, lr_reset])
            # sess.run(open_gate)

            iteration = 0
            event_gif_images = []
            per_class_maximization = np.ones((1,cfg.num_classes))
            per_class_maximization[0,top_i] = -1

            while iteration < cfg.max_iters:
                if class_specific:

                    class_predictions, wrong_logits, _atten_var, _loss, _ = sess.run(
                        [model.val_class_prediction, logits, tf_atten_var, loss, train_op],
                        feed_dict={images_ph: np.expand_dims(test_img, 0),
                                   per_class_logits_ph: per_class_maximization
                                   })
                else:

                    class_predictions, wrong_logits, _atten_var, _loss, _ = sess.run(
                        [model.val_class_prediction, logits, tf_atten_var, loss, train_op],
                        feed_dict={images_ph: np.expand_dims(test_img, 0),
                                   logits_ph:ground_logits
                                   })
                # _mult_logits = sess.run(mult_logits,
                #     {per_class_logits_ph: np.ones((1, cfg.num_classes)), logits_ph: ground_logits}
                #         )

                if cfg.save_gif and iteration % 10 == 0:
                    print('Iter {0:2d}: {1:.5f} Top {2:3d} {3} logit value {4:.2f}'.format(iteration, _loss,top_i,imagenet_lbls[top_i],wrong_logits[0,top_i]))
                    # print(np.round(np.reshape(_atten_var,(7,7)),2))

                    frame_mask = normalize_filter(filter_type,_atten_var,tf_atten_var.shape[0], tf_atten_var.shape[1])
                    if class_specific:
                        # imageio.imwrite(output_dir + img_name +'_msk_cls_{}_{}.png'.format(top_i,filter_type), frame_mask)
                        heatmap_utils.save_heatmap(frame_mask,save=output_dir + img_name +'_msk_cls_{}_{}.png'.format(top_i,filter_type))
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

                iteration+=1

            frame_mask = normalize_filter(filter_type, _atten_var, tf_atten_var.shape[0], tf_atten_var.shape[1])
            if class_specific:
                # imageio.imwrite(output_dir + img_name +'_msk_cls_{}_{}.png'.format(top_i,filter_type), frame_mask)
                # img_90 = np.percentile(frame_mask, 90)
                # img_50 = np.percentile(frame_mask, 60)


                # frame_mask = cv2.blur(frame_mask, (3, 3))
                # heatmap_utils.save_heatmap(frame_mask,
                #                            save=output_dir + img_name + '_sigmoid_{}_{}.png'.format(top_i, filter_type))

                # frame_mask[frame_mask < img_50] = np.amin(frame_mask)
                # heatmap_utils.save_heatmap(frame_mask,
                #                            save=output_dir + img_name + '_cam_cls_{}_{}.png'.format(top_i, filter_type))
                #
                #
                # drop_mask = np.ones(frame_mask.shape)
                #
                # drop_mask[frame_mask > img_90] = 0
                # heatmap_utils.save_heatmap(drop_mask,
                #                           save=output_dir + img_name + '_drop_mask_{}_{}.png'.format(top_i, filter_type))
                # # frame_mask_copy = frame_mask.copy()
                # print(np.percentile(frame_mask, 98))
                # # frame_mask[frame_mask < img_25] = img_min
                heatmap_utils.apply_heatmap(test_img / 255.0, frame_mask, alpha=0.7,
                                                  save=output_dir + img_name + '_cls_{}_{}.png'.format(top_i,
                                                                                                       filter_type),
                                                  axis='off', cmap='bwr')
                # print(np.percentile(frame_mask, 98))
                # img_98 = np.percentile(frame_mask, 98)
                # img_25 = np.percentile(frame_mask, 25)
                # img_min = np.amin(frame_mask)
                # frame_mask[frame_mask > img_98] = img_min
                #
                # heatmap_utils.apply_heatmap(test_img / 255.0, frame_mask, alpha=0.7,
                #                             save=output_dir + img_name + '_cls_2_{}_{}.png'.format(top_i,
                #                                                                                  filter_type),
                #                             axis='off', cmap='bwr')
                #
                # frame_mask_copy[frame_mask_copy > img_98] = img_98
                # frame_mask_copy[frame_mask_copy < img_25] = img_min
                # heatmap_utils.apply_heatmap(test_img / 255.0, frame_mask_copy, alpha=0.7,
                #                             save=output_dir + img_name + '_cls_3_{}_{}.png'.format(top_i,
                #                                                                                    filter_type),
                #                             axis='off', cmap='bwr')

            else:
                heatmap_utils.apply_heatmap(test_img / 255.0, frame_mask, alpha=0.7,
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
    arg_net = 'densenet161'
    arg_ckpt = 'test_{}_{}'.format(arg_db_name, arg_net)
    args = [
        '--gpu', '0',
        '--output_dir',  './output_heatmaps/',
        '--db_name', arg_db_name,
        '--img_name', 'cute_dog.jpg',
        # '--print_filter_name',
        '--net', arg_net,
        '--learning_rate','0.5', # Rule of thumb bigger attention map need bigger learning rate and vice versa
        '--max_iters','300',
        '--filter_type','l2norm',
        '--replicate_net_at','',
        '--atten_filter_position','dense_block4/{}_conv_block24:0'
        # '--atten_filter_position','InceptionV1/{}_Mixed_5c:0'
        # '--atten_filter_position','dense_block3/{}_conv_block36:0'
    ]
    cfg = BaseConfig().parse(args)
    main(cfg)