'''
    author: peng chen
    date: 2021.8.4
'''
from argparse import ArgumentParser

def parser():
    '''
        add arguments
    '''
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='/media/chiyukunpeng/CHENPENG01/dataset/nuscenes', 
        help='Nuscenes data root')
    parser.add_argument(
        '--seg_config',
        type=str,
        default='configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py', 
        help='segmentation model config file')
    parser.add_argument(
        '--seg_checkpoint',
        type=str,
        default='checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes.pth',
        help='segmentation model checkpoint file')
    parser.add_argument(
        '--device',
        type=str, 
        default='cuda:0', 
        help='device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--show_semantic',
        action='store_true',
        help='whether to show segmentation map'
    )
    parser.add_argument(
        '--write_semantic',
        action='store_true',
        help='whether to write segmentation map'
    )
    parser.add_argument(
        '--out_path',
        type=str,
        default='results',
        help='save out files, such as segmentation map'
    )
    parser.add_argument(
        '--show_projected_points',
        action='store_true',
        help='whether to show projected points'
    )
    parser.add_argument(
        '--write_projected_points',
        action='store_true',
        help='whether to write projected points'
    )
    parser.add_argument(
        '--nusc_scene',
        type=int,
        default=0,
        help='nuscenes mini dataset scene number, in [0, 9]'
    )
    parser.add_argument(
        '--show_painted_points',
        action='store_true',
        help='whether to show painted points'
    )
    
    args = parser.parse_args()
    return args
