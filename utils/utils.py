'''
    author: peng chen
    date: 2021.8.3
'''
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
            [0, 80, 100], [0, 0, 230], [119, 11, 32]]

def paint(semantic, projected_points, points):
    '''
        paint points based on seg map(add semantic channel)
        
        args:
            semantic (list[list[int]]): seg map [900, 1600]
            projected_points (list[list[int]]): projected radar point clouds [N, 3]
            points (list[list[float]]): radar points [N, 18]
        
        returns:
            painted points (list[list[float]]): [N, 4]
    '''
    semantic_channel = semantic[projected_points[:, 1], projected_points[:, 0]].reshape(-1, 1)
    assert len(semantic_channel) == len(points)
    painted_points = np.hstack((points[:, :3], semantic_channel))
    
    return painted_points

def show_semantic(
    img_path, 
    semantic, 
    write_semantic,
    out_path,
    palette=PALETTE, 
    opacity=0.5
    ):
    '''
        show segmentation map.
        
        args:
            img_path(str): raw RGB image path
            semantic (list[list[int]]): seg map [900, 1600]
            write_semantic (bool): whether to write segmentation map.
            out_path (str): segmentation map save_path.
            palette (list[list[int]]): the palette of segmentation map.
            opacity (float): opacity of segmentation map. default=0.5.
    '''
    img = cv2.imread(img_path)
    raw_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    palette =  np.array(palette)
    color_seg = np.zeros((semantic.shape[0], semantic.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(palette):
        color_seg[semantic==idx, :] = color
    color_seg = color_seg[..., ::-1]
    
    painted_img = (img * (1-opacity) + color_seg * opacity).astype(np.uint8)
    painted_img = cv2.cvtColor(painted_img, cv2.COLOR_BGR2RGB)
    
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(raw_img)
    axs[0].set_title('raw image')
    axs[0].axis('off')
    axs[1].imshow(painted_img)
    axs[1].set_title('semantic map')
    axs[1].axis('off')
    plt.show()
    
    if write_semantic:
        semantic_name = 'semantic_' + img_path.split('/')[-1]
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        semantic_path = os.path.join(out_path, semantic_name)
        plt.savefig(semantic_path)
        print('-> semantic map saved to {}'.format(semantic_path))
    
def show_projected_points(
    projected_points,
    img_path,
    write_projected_points,
    out_path
    ):
    '''
        show image with projected points
        
        args:
            projected_points (list[list[int]]): [3, N]
            img_path (str): raw RGB image path
            write_projected_points (bool): whether to write projected points
            out_path (str): image with projected points save_path
    '''
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img)
    ax.scatter(projected_points[0, :], projected_points[1, :])
    ax.set_title('projected radar points')
    ax.axis('off')
    plt.show()
    
    if write_projected_points:
        projected_points_name = 'projected_points_' + img_path.split('/')[-1]
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        projected_points_path = os.path.join(out_path, projected_points_name)
        plt.savefig(projected_points_path)
        print('-> projected points saved to {}'.format(projected_points_path))

def show_painted_points(painted_points):
    '''
        show painted radar points in 3D world
        
        args:
            painted_points (list[list[float]]): painted radar points. [N, 4]
    '''
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=1280, height=720)
    pcd = o3d.geometry.PointCloud()
    visualizer.add_geometry(pcd)
    
    opt = visualizer.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    
    semantic = painted_points[:, 3]
    points = painted_points[:, :3]
    colors = np.zeros((semantic.shape[0], 3))
    for idx, color in enumerate(PALETTE):
        colors[semantic==idx] = (color[0]/255, color[1]/255, color[2]/255)
    
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


    
    