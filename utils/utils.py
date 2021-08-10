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
                [0]: x
                [1]: y
                [2]: z
                [3]: dyn_drop
                [4]: id
                [5]: rcs
                [6]: vx
                [7]: vy
                [8]: vx_comp
                [9]: vy_comp
                [10]: is_quality_valid
                [11]: ambig_state
                [12]: x_rms
                [13]: y_rms
                [14]: invalid_state
                [15]: pdh0
                [16]: vx_rms
                [17]: vy_rms
        
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

def points2RAMap(points, ramap_width=128, ramap_height=128):
    '''
        transform radar points to RAMap
        
        args:
            points (list[list[float]]): radar points [N, 18]
            ramap_width (int): ramap azimuth size
            ramap_height (int): ramap range size
            
        returns:
            ramap (ndarray): radar RAMap [W, H, 3] 
    '''
    dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    rcs = points[:, 5]
    
    ramap_mask = ramapEncoder(azimuth, dist, rcs, ramap_width, ramap_height)
    ramap = np.zeros((ramap_width, ramap_height, 3), dtype=float)
    
    for i in range(len(ramap_mask)):
        w = int(ramap_mask[i][0])
        w = w if w < ramap_width else ramap_width - 1
        h = int(ramap_mask[i][1])
        h = h if h < ramap_height else ramap_height - 1
        r = ramap_mask[i][2]
        g = ramap_mask[i][3]
        b = ramap_mask[i][4]
        ramap[w][h][0] = r
        ramap[w][h][1] = g
        ramap[w][h][2] = b
    
    return ramap

def ramapEncoder(azimuth, distance, rcs, ramap_width, ramap_height):
    '''
        encode (azimuth, distance, rcs) to (x, y, rgb)
        
        args:
            azimuth (list[float]): azimuth between radar and object
            dist (list[float]): range between radar and object
            rcs (list[float]): radar cross section. rcs in db = 10*log(rcs in m**2)
            ramap_width (int): ramap azimuth size
            ramap_height (int): ramap range size
            
        returns:
            ramap_mask (list[list[float]]): ramap mask informations: x, y, r, g, b. [N, 5]
    '''
    azimuth = [i + abs(min(azimuth)) for i in azimuth]
    azimuth_length = max(azimuth) - min(azimuth)
    azimuth_factor = ramap_width / azimuth_length
    x = [i * azimuth_factor for i in azimuth]
    x = np.expand_dims(x, axis=1)
    
    distance_length = max(distance) - min(distance)
    distance_factor = ramap_height / distance_length
    y = [i * distance_factor for i in distance]
    y = np.expand_dims(y, axis=1)
    
    rcs = [i + abs(min(rcs)) for i in rcs]
    rcs_factor = [i / max(rcs) for i in rcs]
    r = [int(255 * i) for i in rcs_factor]
    r = np.expand_dims(r, axis=1)
    g = [0] * len(r)
    g = np.expand_dims(g, axis=1)
    b = [0] * len(r)
    b = np.expand_dims(b, axis=1)
    
    ramap_mask = np.hstack((x, y, r, g, b))
    
    return ramap_mask
    
def show_ramap(ramap, write_ramap, image_path, out_path):
    '''
        visualize radar RAMap
        
        args:
            ramap (ndarray): range azimuth map[w, h, 3]
            write_ramap (bool): whether to write ramap
            image_path (str): RGB image path
            out_path (str): ramap save path
    '''
    fig, ax = plt.subplots(1, 1)
    ax.imshow(ramap)
    ax.set_title('RAMap')
    plt.show()
    
    if  write_ramap:
        ramap_name =   'ramap_' + image_path.split('/')[-1]
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        ramap_path = os.path.join(out_path, ramap_name)
        plt.savefig(ramap_path)
        print('-> ramap saved to {}'.format(ramap_path))

