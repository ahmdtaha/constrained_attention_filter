import os
import getpass
import argparse



# import constants as const



class BaseConfig:

    def __init__(self):

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--gpu', type=str, default='0',
                                 help='which gpu')
        self.parser.add_argument('--process_shift', type=int, default=0,
                                 help='Shift to split image list')

        self.parser.add_argument('--atten_var_name', type=str,help='attention var name')
        self.parser.add_argument('--output_dir', type=str, default=None,
                                 help='where to save experiment log and model')
        self.parser.add_argument('--replicate_net_at', type=str, required=True,
                                 help='Which Layer to replicate the network at?')
        self.parser.add_argument('--net', type=str, default='resnet50',
                                 help='Which networks? resnet50, inc4,densenet161')
        self.parser.add_argument('--max_iters', type=int, default=40000,
                                 help='Number of parallel threads')
        self.parser.add_argument('--atten_filter_position', type=str,
                                 help='Where to insert the attention filter into the network?'
                                      'For denseNet_last conv layer: dense_block4/{}_conv_block24:0'
                                      'For denseNet_last intermediate layer: dense_block4/{}_conv_block20:0'
                                        
                                      'For InceptionNetV1_last conv layer: InceptionV1/{}_Mixed_5c:0'
                                      
                                    'I print all the layer name in order to make selecting a layer easier'
                                 )

        self.parser.add_argument('--db_name', type=str, default='imagenet',
                                 help='Database name')
        self.parser.add_argument('--checkpoint_suffix', type=str, default='base_config',
                                 help='Number of parallel threads')

        self.parser.add_argument('--checkpoint_filename', type=str, default='model.ckpt',
                                 help='Number of parallel threads')

        self.parser.add_argument('--learning_rate', type=float, default=0.01,
                                 help='Number of parallel threads')
        self.parser.add_argument('--end_learning_rate', type=float, default=0,
                                 help='Number of parallel threads')

        self.parser.add_argument('--log_filename', type=str, default='logger',
                                 help='Number of parallel threads')

        self.parser.add_argument('--filter_type', type=str, default='l2norm',choices=['l2norm','softmax','gauss'],
                                 help='What is the constraint to apply during optimization')

        self.parser.add_argument('--img_name', type=str,
                                 help='specific sample image inside the input_imgs folder')

        self.parser.add_argument('--save_gif',  action='store_true', default=False,
                                 help='Whether to save prepare gif of the converging process or not'
                                      'Please note that enabling this will slow the code in order to keep a stack of heatmaps during optimization')

        self.parser.add_argument('--print_filter_name', action='store_true', default=False,
                                 help='By default a filter is added after every conv layer. '
                                      'param: atten_var_name determine which filter to be optimized'
                                      'If not sure about the conv_filter, use this option and all names will be printed')

    def _load_user_setup(self):
        username = getpass.getuser()
        # if username == 'ataha':
        logging_threshold = 500
        local_datasets_dir = '/mnt/data/datasets/'
        pretrained_weights_dir = '/mnt/data/pretrained/'
        training_models_dir = '.'
        # else:  ##
        #     raise NotImplementedError('base_config:: User not found')
        ##TODO : Make sure to fill these values
        # local_datasets_dir = ''  # where is the dataset
        # pretrained_weights_dir = local_datasets_dir + 'Model/'  # where the imagenet pre-trained weight
        # training_models_dir = ''  # where to save the trained models
        # dump_path = ''  ## I use this to dump files during debuging. I ought not use it here
        # logging_threshold = 500
        # batch_size = 32
        # caffe_iter_size = 12
        # debug_mode = False

        return local_datasets_dir,pretrained_weights_dir,training_models_dir,logging_threshold

    def parse(self,args):
        cfg = self.parser.parse_args(args)

        local_datasets_dir, pretrained_weights_dir, training_models_dir, logging_threshold = self._load_user_setup()
        cfg.num_classes, cfg.db_path, cfg.db_tuple_loader, cfg.train_csv_file, cfg.val_csv_file, cfg.test_csv_file    = self.db_configuration(cfg.db_name,local_datasets_dir)
        cfg.network_name,cfg.sub_network_name, cfg.imagenet_weights_filepath, cfg.preprocess_func = self._load_net_configuration(cfg.net,pretrained_weights_dir)


        cfg.output_dir = os.path.join(training_models_dir,cfg.output_dir)

        if cfg.db_name == 'imagenet':
            if cfg.net in ['mobile','inceptionv1','resnet50']:
                cfg.num_classes = cfg.num_classes + 1 ## add 1 for background class


        return cfg

    def _load_net_configuration(self,model,pretrained_weights_dir):
        if model == 'resnet50':
            network_name = 'nets.resnet_v2.ResNet50'
            sub_network_name = 'nets.sub_resnet_v2.ResNet50'
            imagenet_weights_filepath = pretrained_weights_dir + 'resnet_v2_50/resnet_v2_50.ckpt'
            preprocess_func = 'inception_v1'
        elif model == 'resnet50_v1':
            network_name = 'nets.resnet_v1.ResNet50'
            imagenet_weights_filepath = pretrained_weights_dir + 'resnet_v1_50/resnet_v1_50.ckpt'
            preprocess_func = 'vgg'
        elif model == 'densenet161':
            network_name = 'nets.densenet161.DenseNet161'
            sub_network_name = 'nets.sub_densenet161.DenseNet161'
            imagenet_weights_filepath = pretrained_weights_dir + 'tf-densenet161/tf-densenet161.ckpt'
            preprocess_func = 'densenet'

        elif model == 'inc4':
            network_name = 'nets.inception_v4.InceptionV4'
            imagenet_weights_filepath = pretrained_weights_dir + 'inception_v4/inception_v4.ckpt'
            preprocess_func = 'inception_v1'

        elif model == 'inceptionv1':
            network_name = 'nets.inception_v1.InceptionV1'
            sub_network_name = 'nets.sub_inception_v1.InceptionV1'
            imagenet_weights_filepath = pretrained_weights_dir + 'inception_v1/inception_v1.ckpt'
            preprocess_func = 'inception_v1'


        elif model == 'inc3':
            network_name = 'nets.inception_v3.InceptionV3'
            imagenet_weights_filepath = pretrained_weights_dir + 'inception_v3.ckpt'
            preprocess_func = 'inception_v1'

        elif model == 'mobile':
            network_name = 'nets.mobilenet_v1.MobileV1'
            imagenet_weights_filepath = pretrained_weights_dir + 'mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt'
            preprocess_func = 'inception_v1'
            sub_network_name = 'nets.sub_mobilenet_v1.MobileV1'
        elif model == 'vgg16':
            network_name = 'nets.vgg.VGG'
            sub_network_name = 'nets.sub_vgg.VGG'
            imagenet_weights_filepath = pretrained_weights_dir + 'vgg_16/vgg_16.ckpt'
            preprocess_func = 'vgg'
        elif model == 'cam_vgg16':
            network_name = 'nets.cam_vgg.CAMVGG'
            sub_network_name = 'nets.sub_cam_vgg.CAMVGG'
            imagenet_weights_filepath = pretrained_weights_dir + 'cam_vgg/'
            preprocess_func = 'cam_vgg'
        else:
            raise NotImplementedError('network name not found')

        return network_name,sub_network_name,imagenet_weights_filepath,preprocess_func

    def db_configuration(self, dataset_name, datasets_dir):

        if dataset_name == 'imagenet':
            num_classes = 1000
            # db_path =   '/mnt/data/imagenet/ILSVRC'
            db_path = datasets_dir + 'imagenet/ILSVRC'
            db_tuple_loader = 'data_sampling.imagenet_tuple_loader.ImageNetTupleLoader'
            train_csv_file = '/lists/train_all_sub_list.csv'
            val_csv_file = '/lists/val_all_sub_list.csv'
            test_csv_file = '/lists/val_all_sub_list.csv'
        else:
            raise NotImplementedError('dataset_name not found')

        return num_classes,db_path,db_tuple_loader,train_csv_file,val_csv_file,test_csv_file


if __name__ == '__main__':
    args = [
        '--db_name','flowers'
    ]
    cfg = BaseConfig().parse(args)
    print(cfg.num_classes,cfg.train_csv_file)
    if hasattr(cfg,'abc'):
        print(cfg.abc)
    else:
        print('Something is wrong')

