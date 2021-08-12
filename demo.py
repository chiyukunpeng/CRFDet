'''
    author: peng chen
    date: 2021.8.2
'''
import os
import numpy as np
import time

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

from nuscenes.utils.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.utils.data_classes import RadarPointCloud

from utils.utils import *
from utils.args import parser


def main():
    args = parser()

    print('### loading images and radar points from nuscenes ###')
    nusc = NuScenes(dataroot=args.dataset)
    nusc_exp = NuScenesExplorer(nusc)
    scene = nusc.scene[args.nusc_scene]
    sample = nusc.get('sample', scene['first_sample_token'])
    
    cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    image_path = os.path.join(args.dataset, cam_front_data['filename'])
    
    pc_data = nusc.get('sample_data', sample['data']['RADAR_FRONT'])
    pc_path = os.path.join(args.dataset, pc_data['filename'])
    points = RadarPointCloud.from_file(pc_path).points
    ramap = points2RAMap(points.T, args.ramap_width, args.ramap_height)
    if args.show_ramap:
        show_ramap(ramap, args.write_ramap, image_path, args.out_path)

    print('\n### semantic map inference ###')
    model = init_segmentor(args.seg_config, args.seg_checkpoint, device=args.device)
    init_time = time.time()
    semantic = inference_segmentor(model, image_path)[0]
    semantic_time = time.time() - init_time
    print('-> semantic inference time:{}'.format(semantic_time))
    
    # show semantic map 
    if args.show_semantic:
        show_semantic(
            image_path, 
            semantic,
            args.write_semantic,
            args.out_path)
    
    print('\n### project radar points onto the image ###')
    init_time = time.time()
    projected_points, _, _ = nusc_exp.map_pointcloud_to_image(
                                                                    pointsensor_token=sample['data']['RADAR_FRONT'],
                                                                    camera_token=sample['data']['CAM_FRONT'])
    projected_points = projected_points.astype(np.int32)
    project_time = time.time() - init_time
    print('-> project time:{}'.format(project_time))
    
    # show image with projected radar points
    if args.show_projected_points:
        show_projected_points(
            projected_points, 
            image_path, 
            args.write_projected_points,
            args.out_path)
    
    print('\n### paint radar points ###')
    init_time = time.time()
    painted_points = paint(semantic, projected_points.T, points.T)
    paint_time = time.time() - init_time
    print('-> paint time:{}'.format(paint_time))
    
    # show painted points
    if args.show_painted_points:
        show_painted_points(painted_points)
    
    print('\n### map painted points to ramap ###')
    init_time = time.time()
    painted_ramap = map_points_to_ramap(ramap, painted_points)
    painted_ramap_time  = time.time() - init_time
    print('-> paint ramap time:{}'.format(painted_ramap_time))
    
    # show painted ramap
    if args.show_painted_ramap:
        show_painted_ramap(
            painted_ramap, 
            args.write_painted_ramap, 
            image_path, 
            args.out_path)
    

if __name__ == '__main__':
    main()
    
