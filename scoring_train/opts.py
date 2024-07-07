import argparse
from pathlib import Path


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',
                        default='./prostate_dataset/Case_input_aligned',
                        type=str,
                        help='Root directory path')
    parser.add_argument('--root_test_path',
                        default='./prostate_dataset/1113_segmentation_input',
                        type=str,
                        help='Root directory path')
    parser.add_argument('--train_txt_file',
                        default='./data/train_PI-RADS_ISUP_manxi.txt',
                        type=str,
                        help='Root directory path')
    parser.add_argument('--test_txt_file',
                        default='./data/test_PI-RADS_ISUP_manxi.txt',
                        type=str,
                        help='Root directory path')
    parser.add_argument('--inf_txt_file',
                        default='./data/test_PI-RADS_ISUP_manxi.txt',
                        type=str,
                        help='Root directory path')
    parser.add_argument("--used_ins",
                        default=False,
                        type=bool)
    parser.add_argument('--result_path',
                        default=None,
                        type=Path,
                        help='Result directory path')
    parser.add_argument('--used_dataset',
                        default='private',
                        type=str,
                        help='public or private')

    parser.add_argument(
        '--datasampler',
        default=False,
        type=bool,
        help='resample the dataset'
    )
    parser.add_argument(
        '--n_classes',
        default=3,
        type=int,
        help=
        'Number of classes (activitynet: 200, kinetics: 400 or 600, ucf101: 101, hmdb51: 51)'
    )
    parser.add_argument('--pretrain_path',
                        default="pretrain/r3d34_KM_200ep.pth",
                        type=Path,
                        help='Pretrained model path (.pth).')
    parser.add_argument(
        '--ft_begin_module',
        default='',
        type=str,
        help=('Module name of beginning of fine-tuning'
              '(conv1, layer1, fc, denseblock1, classifier, ...).'
              'The default means all layers are fine-tuned.'))
    parser.add_argument('--sample_size',
                        default=112,
                        type=int,
                        help='Height and width of inputs')
    parser.add_argument('--sample_duration',
                        default=4,
                        type=int,
                        help='Temporal duration of inputs')
    parser.add_argument('--loss_weight',
                        default=0.2,
                        type=float,
                        help='kd loss weight')
    parser.add_argument('--kd_loss',
                        default='KL',
                        type=str,
                        help='KL or CE')
    parser.add_argument(
        '--center_crop',
        default=False,
        type=bool,
        help='data augmentation'
    )
    parser.add_argument(
        '--flip',
        default=True,
        type=bool,
        help='data augmentation'
    )
    parser.add_argument(
        '--rot',
        default=True,
        type=bool,
        help='data augmentation'
    )
    parser.add_argument(
        '--resize_select',
        default=False,
        type=bool,
        help='data augmentation'
    )
    parser.add_argument('--learning_rate',
                        default=0.001,
                        type=float,
                        help=('Initial learning rate'
                              '(divided by 10 while training by lr scheduler)'))
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--loss_select',
                        default="CE",
                        type=str,
                        help='CE, BalanceCE, multiFocal')
    parser.add_argument('--focalweight',
                        default=[1.0,1.0],
                        type=float,
                        nargs='+',
                        help='weight of focal loss')
    parser.add_argument('--focalgamma',
                        default=2,
                        type=int,
                        help='gamma of focal loss')
    parser.add_argument('--dampening',
                        default=0.0,
                        type=float,
                        help='dampening of SGD')
    parser.add_argument('--weight_decay',
                        default=1e-3,
                        type=float,
                        help='Weight Decay')
    parser.add_argument('--nesterov',
                        action='store_true',
                        help='Nesterov momentum')
    parser.add_argument('--optimizer',
                        default='sgd',
                        type=str,
                        help='Currently only support SGD')
    parser.add_argument('--lr_scheduler',
                        default='ExponentialLR',
                        type=str,
                        help='Type of LR scheduler (multistep | plateau | ExponentialLR)')
    parser.add_argument(
        '--multistep_milestones',
        default=[50, 100, 150],
        type=int,
        nargs='+',
        help='Milestones of LR scheduler. See documentation of MultistepLR.')
    parser.add_argument(
        '--overwrite_milestones',
        action='store_true',
        help='If true, overwriting multistep_milestones when resuming training.'
    )
    parser.add_argument(
        '--plateau_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )
    parser.add_argument('--batch_size',
                        default=4,
                        type=int,
                        help='Batch Size')
    parser.add_argument(
        '--inference_batch_size',
        default=0,
        type=int,
        help='Batch Size for inference. 0 means this is the same as batch_size.'
    )
    parser.add_argument(
        '--batchnorm_sync',
        action='store_true',
        help='If true, SyncBatchNorm is used instead of BatchNorm.')
    parser.add_argument('--n_epochs',
                        default=200,
                        type=int,
                        help='Number of total epochs to run')
    parser.add_argument('--n_val_samples',
                        default=3,
                        type=int,
                        help='Number of validation samples for each activity')
    parser.add_argument('--resume_path',
                        default=None,
                        type=Path,
                        help='Save data (.pth) of previous training')
    parser.add_argument('--no_train',
                        action='store_true',
                        help='If true, training is not performed.')
    parser.add_argument('--no_val',
                        action='store_true',
                        help='If true, validation is not performed.')
    parser.add_argument('--inference',
                        action='store_true',
                        help='If true, inference is performed.')
    parser.add_argument('--no_cuda',
                        action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--n_threads',
                        default=4,
                        type=int,
                        help='Number of threads for multi-thread loading')
    parser.add_argument('--checkpoint',
                        default=10,
                        type=int,
                        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help=
        '(resnet | resnet2p1d | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--n_input_channels',
                        default=3,
                        type=int,
                        help='channels')
    parser.add_argument('--model_depth',
                        default=34,
                        type=int,
                        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--conv1_t_size',
                        default=7,
                        type=int,
                        help='Kernel size in t dim of conv1.')
    parser.add_argument('--conv1_t_stride',
                        default=2,
                        type=int,
                        help='Stride in t dim of conv1.')
    parser.add_argument('--dilation_flag',
                        default=True,
                        type=bool,
                        help='define the dilation in ResNet')
    parser.add_argument('--no_max_pool',
                        action='store_true',
                        help='If true, the max pooling after conv1 is removed.')
    parser.add_argument('--resnet_shortcut',
                        default='B',
                        type=str,
                        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--resnet_widen_factor',
        default=1.0,
        type=float,
        help='The number of feature maps of resnet is multiplied by this value')
    parser.add_argument('--wide_resnet_k',
                        default=2,
                        type=int,
                        help='Wide resnet k')
    parser.add_argument('--resnext_cardinality',
                        default=32,
                        type=int,
                        help='ResNeXt cardinality')
    parser.add_argument('--input_type',
                        default='rgb',
                        type=str,
                        help='(rgb | flow)')
    parser.add_argument('--manual_seed',
                        default=1,
                        type=int,
                        help='Manually set random seed')
    parser.add_argument('--accimage',
                        action='store_true',
                        help='If true, accimage is used to load images.')
    parser.add_argument('--output_topk',
                        default=5,
                        type=int,
                        help='Top-k scores are saved in json file.')
    parser.add_argument('--file_type',
                        default='jpg',
                        type=str,
                        help='(jpg | hdf5)')
    parser.add_argument('--tensorboard',
                        action='store_true',
                        help='If true, output tensorboard log file.')
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Use multi-processing distributed training to launch '
        'N processes per node, which has N GPUs.')
    parser.add_argument('--dist_url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--world_size',
                        default=-1,
                        type=int,
                        help='number of nodes for distributed training')

    args = parser.parse_args()

    return args
