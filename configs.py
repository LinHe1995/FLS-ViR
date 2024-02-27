import argparse

def get_trainmae_config():
    """Returns the train configuration."""
    parser = argparse.ArgumentParser(description='PyTorch Example')#创建一个ArgumentParser对象
    parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='input batch size for training')
    parser.add_argument('--freeze-epoch', type=int, default=0, help='Freezing the first few backbones of modeling')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs for training')
    parser.add_argument('--is-save', type=bool, default=True, help='save model')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum')
    parser.add_argument('--label-smooth', type=float, default=1e-3, help='label smoothing to reduce over fitting')
    parser.add_argument('--seed', type=int, default=123, metavar='S', help='fixed random seed')
    parser.add_argument('--is-use-cuda', type=bool, default=True, help='use cuda')
    parser.add_argument('--gpu_ids', type=str, default='0,1')
    parser.add_argument('--data-dir', type=str, default='./data/mini-imagenet/', help='data path for training')
    parser.add_argument('--is-resume', type=bool, default=True, metavar='R', help='use the pretrained model')
    parser.add_argument('--model-dir', type=str, default='./model/', help='store the model weight')
    parser.add_argument('--val-num', type=float, default=0.3, help='percentage of validate data')
    parser.add_argument('--project_name', type=str, default='transformer classify', help='project name')
    parser.add_argument('--is-use-aug', type=bool, default=True, help='use data augmentation')
    parser.add_argument('--image-size', type=int, default=224, choices=[224, 224], help='image size')#/14
    parser.add_argument('--patch-size', type=int, default=16, help='number of segmentation blocks')#16
    parser.add_argument('--num-class', type=int, default=27, help='number of classes')
    parser.add_argument('--emb-dim', type=int, default=768, help='embedded dimension')#1024 768
    parser.add_argument('--mlp-dim', type=int, default=3072, help='mlp dimension')#4096
    parser.add_argument('--seq-len', type=int, default=99, help='length of a fixed-length sequence')
    parser.add_argument('--num-heads', type=int, default=16, help='number of multi heads')#16 12
    parser.add_argument('--num-layers', type=int, default=12, help='number of layers')#12
    parser.add_argument('--attn-dropout-rate', type=float, default=0.7, help='attention dropout rate')
    parser.add_argument('--dropout-rate', type=float, default=0.7, help='dropout rate')
    parser.add_argument('--layer-dropout', type=float, default=0.0, help='layer dropout rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')#mae_visualize_vit_base.pthcheckpoint-806
    parser.add_argument('--pretrained-weight', type=str, default="./weight/mae_visualize_vit_base.pth", help="the path of the pytorch weight(.pth)")#imagenet21k+imagenet2012_ViT-B_16
    parser.add_argument('--mask_regular', action='store_true', help='Uniform sampling for supporting pyramid-based vits')
    parser.add_argument('--token_size', default=int(224 / 16), type=int, help='number of patch (in one dimension), usually input_size//16')  # for mask generator
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='Masking ratio (percentage of removed patches).')##########
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--bf16', action='store_true', help='whether to use bf16')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--de_emb_dim', type=int, default=512, help='decoder embedded dimension')
    parser.add_argument('--de_num_layers', type=int, default=8, help='decoder number of layers')
    parser.add_argument('--de_num_heads', type=int, default=16, help='decoder number of heads')
    parser.add_argument('--de_mlp_ratio', type=float, default=4., help='mlp ratio')

    return parser

def get_train_config():
    """Returns the train configuration."""
    parser = argparse.ArgumentParser(description='PyTorch Example')#创建一个ArgumentParser对象
    parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='input batch size for training')
    parser.add_argument('--freeze-epoch', type=int, default=0, help='Freezing the first few backbones of modeling')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs for training')
    parser.add_argument('--is-save', type=bool, default=True, help='save model')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum')
    parser.add_argument('--label-smooth', type=float, default=1e-3, help='label smoothing to reduce over fitting')
    parser.add_argument('--seed', type=int, default=123, metavar='S', help='fixed random seed')
    parser.add_argument('--is-use-cuda', type=bool, default=True, help='use cuda')
    parser.add_argument('--gpu_ids', type=str, default='0,1')
    parser.add_argument('--data-dir', type=str, default='./data/UCMerced_LandUse/', help='data path for training')
    parser.add_argument('--is-resume', type=bool, default=True, metavar='R', help='use the pretrained model')
    parser.add_argument('--model-dir', type=str, default='./model/', help='store the model weight')
    parser.add_argument('--val-num', type=float, default=0.3, help='percentage of validate data')
    parser.add_argument('--project_name', type=str, default='transformer classify', help='project name')
    parser.add_argument('--is-use-aug', type=bool, default=True, help='use data augmentation')
    parser.add_argument('--image-size', type=int, default=224, choices=[512, 512], help='image size')#/14
    parser.add_argument('--patch-size', type=int, default=16, help='number of segmentation blocks')#16
    parser.add_argument('--num-class', type=int, default=27, help='number of classes')
    parser.add_argument('--emb-dim', type=int, default=768, help='embedded dimension')#1024 768
    parser.add_argument('--mlp-dim', type=int, default=3072, help='mlp dimension')#4096
    parser.add_argument('--seq-len', type=int, default=75, help='length of a fixed-length sequence')
    parser.add_argument('--num-heads', type=int, default=16, help='number of multi heads')#16 12
    parser.add_argument('--num-layers', type=int, default=12, help='number of layers')
    parser.add_argument('--attn-dropout-rate', type=float, default=0.7, help='attention dropout rate')
    parser.add_argument('--dropout-rate', type=float, default=0.7, help='dropout rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')#mae_visualize_vit_base.pth
    parser.add_argument('--pretrained-weight', type=str, default="./weight/Epoch-192-Test_loss-15.248.pth", help="the path of the pytorch weight(.pth)")#imagenet21k+imagenet2012_ViT-B_16
    parser.add_argument('--mask_regular', action='store_true', help='Uniform sampling for supporting pyramid-based vits')
    parser.add_argument('--token_size', default=int(224 / 16), type=int, help='number of patch (in one dimension), usually input_size//16')  # for mask generator
    parser.add_argument('--mask_ratio', default=0.0, type=float, help='Masking ratio (percentage of removed patches).')##########
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--bf16', action='store_true', help='whether to use bf16')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')

    return parser
