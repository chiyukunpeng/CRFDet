'''
    author: peng chen
    date: 2021.8.3
'''
import cv2
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import open3d as o3d
from numpy.lib.scimath import log10


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
            painted points (list[list[float]]): [N, 19]
    '''
    semantic_channel = semantic[projected_points[:, 1], projected_points[:, 0]].reshape(-1, 1)
    assert len(semantic_channel) == len(points)
    painted_points = np.hstack((points, semantic_channel))
    
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
            painted_points (list[list[float]]): painted radar points. [N, 19]
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
            points (list[list[float]]): radar points [N, 19]
            ramap_width (int): ramap azimuth size
            ramap_height (int): ramap range size
            
        returns:
            ramap (ndarray): radar RAMap [W, H, 3] 
    '''
    dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    rcs = points[:, 5]
    
    ramap_mask = ramapEncoder(azimuth, dist, rcs, ramap_width, ramap_height)
    ramap = np.zeros((ramap_width, ramap_height, 3), dtype=np.int32)
    
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

def map_points_to_ramap(ramap, points):
    '''
        map painted points to ramap
        
        args:
            ramap (ndarray): range azimuth map [w, h, 3]
            points (list[list[float]]): painted points [N, 19]
            
        returns:
            painted_ramap (ndarray): ramap with painted points
    '''
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    azimuth = [i + abs(min(azimuth)) for i in azimuth]
    azimuth_length = max(azimuth) - min(azimuth)
    azimuth_factor = ramap.shape[0] / azimuth_length
    x = [i * azimuth_factor for i in azimuth]
    x = np.expand_dims(x, axis=1)
    
    distance = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    distance_length = max(distance) - min(distance)
    distance_factor = ramap.shape[1] / distance_length
    y = [i * distance_factor for i in distance]
    y = np.expand_dims(y, axis=1)
    
    for i in range(len(points)):
        w = int(x[i]) if int(x[i]) < ramap.shape[0] else ramap.shape[0] - 1
        h = int(y[i]) if int(y[i]) < ramap.shape[1] else ramap.shape[1] - 1
        c = PALETTE[int(points[i][3])]
        
        ramap[w][h][0] = c[0]
        ramap[w][h][1] = c[1]
        ramap[w][h][2] = c[2]
    
    painted_ramap = ramap.astype(np.int32)
    
    return painted_ramap
        
def show_painted_ramap(painted_ramap, write_painted_ramap, image_path, out_path):
    '''
    show painted ramap

    args:
        painted_ramap (ndarray): painted range azimuth map [w, h, 3]
        write_painted_ramap (bool): whether to write painted ramap.
        image_path (str): image path
        out_path (str): painted ramap save path
    '''
    fig, ax = plt.subplots(1, 1)
    ax.imshow(painted_ramap)
    ax.set_title('painted ramap')
    plt.show()

    if write_painted_ramap:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        painted_ramap_name = 'painted_ramap_' + image_path.split('/')[-1]
        painted_ramap_path = os.path.join(out_path, painted_ramap_name)
        plt.savefig(painted_ramap_path)
        print('-> painted ramap saved to {}'.format(painted_ramap_path))
    
def point2rdmap(points, radar_param):
    '''
        transform points to range doppler map

        args:
            points (list[list[float]]): radar points [N, 18]
            radar_param (dict): radar parameters
        
        returns:
            rdmap (list[list[float]]): range doppler map [W, H, 3]
    '''
    distance = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    velocity = np.sqrt(points[:, 8]**2 + points[:, 9]**2)

    bandwidth = 3e8/(2*radar_param['range_res'])
    t_chirp = 5.5*2*radar_param['max_range']/3e8
    slope = bandwidth/t_chirp

    total_time = np.linspace(0, 
                            radar_param['num_doppler']*t_chirp,
                            radar_param['num_range']*radar_param['num_doppler'])
    t_x = np.zeros((len(points), len(total_time)))
    r_x = np.zeros((len(points), len(total_time)))
    mix = np.zeros((len(points), len(total_time)))
    r_t = np.zeros((len(points), len(total_time)))
    t_d = np.zeros((len(points), len(total_time)))
    
    for i in range(len(points)):
        for j in range(len(total_time)):
            r_t[i][j] = distance[i] + velocity[i]*total_time[j]
            t_d[i][j] = 2*r_t[i][j]/3e8

            t_x[i][j] = math.cos(2*math.pi*(radar_param['freq'])*total_time[j] + \
                                slope*((total_time[j]**2)/2))
            r_x[i][j] = math.cos(2*math.pi*(radar_param['freq'])*(total_time[j]-t_d[i][j])) + \
                                slope*(((total_time[j]-t_d[i][j])**2)/2)
            mix[i][j] = np.dot(t_x[i][j], r_x[i][j])
    
    # mix = np.sum(mix, axis=0)
    mix = np.max(mix, axis=0)
    # mix = np.mean(mix, axis=0)
    mix = np.expand_dims(mix, axis=0)
    reshaped_mix = mix.reshape((radar_param['num_doppler'], radar_param['num_range']))
    sig_fft2 = np.fft.fft2(reshaped_mix, (radar_param['num_doppler'], radar_param['num_range']))
    sig_fft2 = np.fft.fftshift(sig_fft2)
    mask = 10*np.log(np.abs(sig_fft2))
    mask /= np.max(mask)
    # mask = cfar(mask, radar_param['num_range'], radar_param['num_doppler'])
    mask = (255*mask).astype(np.int32)
    rdmap = np.zeros((radar_param['num_doppler'], radar_param['num_range'], 3), dtype=np.int32)
    rdmap[:, :, 2] = mask
    
    return rdmap

def cfar(mask, num_range=128, num_doppler=128, num_train=8, num_guard=4):
    '''
    detect peaks with 2D CFAR algorithm.
    
    args:
        mask (list[list[float]]): rdmap mask.
        num_range (int): number of range
        num_doppler (int): number of doppler
        num_train (int): number of training cells.
        num_guard (int): number of guard cells.
        
    returns:
        mask (list[list[float]]): rdmap mask after cfar
    '''
    for i in range(num_train+num_guard+1, num_range-num_train-num_guard):
        for j in range(num_train+num_guard+1, num_doppler-num_train-num_guard):
            noise = np.zeros((1,1))
            for p in range(i-num_train-num_guard, i+num_train+num_guard):
                for q in range(j-num_train-num_guard, j+num_train+num_guard):
                    if abs(i-p) > num_guard or abs(j-q) > num_guard:
                        noise += 10**(mask[p][q]/10)
            threshold = 10*log10(noise/(4*(num_train+num_guard+1)**2-num_guard**2-1))
            mask[i][j] = 0 if mask[i][j] < threshold else 1
    
    return mask

def show_rdmap(rdmap, write_rdmap, image_path, out_path):
    '''
        visualize radar RDMap
        
        args:
            rdmap (ndarray): range doppler map[w, h, 3]
            write_rdmap (bool): whether to write rdmap
            image_path (str): RGB image path
            out_path (str): rdmap save path
    '''
    fig, ax = plt.subplots(1, 1)
    ax.imshow(rdmap)
    ax.set_title('RDMap')
    plt.show()
    
    if  write_rdmap:
        rdmap_name =   'rdmap_' + image_path.split('/')[-1]
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        rdmap_path = os.path.join(out_path, rdmap_name)
        plt.savefig(rdmap_path)
        print('-> rdmap saved to {}'.format(rdmap_path))

def map_points_to_rdmap(points, rdmap, radar_param):
    '''
        map painted points to rdmap
        
        args:
            points (list[list[float]]): painted points [N, 19]
            rdmap (ndarray): range doppler map [w, h, 3]
            radar_param (dict): radar parameters
            
        returns:
            mapped_rdmap (ndarray): mapped rdmap [w, h, 3]
    '''
    distance = np.sqrt(points[:, 0]**2 + points[:, 1]**2).astype(np.int32)
    distance = np.where(rdmap.shape[0]>distance, distance, rdmap.shape[0]-1)
    distance = np.expand_dims(distance, axis=1)
    
    velocity = np.sqrt(points[:, 8]**2 + points[:, 9]**2)
    doppler = (velocity * radar_param['freq'] / 3e8).astype(np.int32)
    doppler = np.where(rdmap.shape[1]>doppler, doppler, rdmap.shape[1]-1)
    doppler = np.expand_dims(doppler, axis=1)
    
    rdmap[:, :, :] = (127, 0, 255)
    for i in range(len(points)):
        w, h = doppler[i], distance[i]
        c = PALETTE[int(points[i][18])]
        rdmap[w, h, :] = c
    
    return rdmap

def show_painted_rdmap(painted_rdmap, write_painted_rdmap, image_path, out_path):
    '''
    show painted rdmap

    args:
        painted_rdmap (ndarray): painted range doppler map [w, h, 3]
        write_painted_rdmap (bool): whether to write painted rdmap.
        image_path (str): image path
        out_path (str): painted ramap save path
    '''
    fig, ax = plt.subplots(1, 1)
    ax.imshow(painted_rdmap)
    ax.set_title('painted rdmap')
    plt.show()

    if write_painted_rdmap:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        painted_rdmap_name = 'painted_ramap_' + image_path.split('/')[-1]
        painted_rdmap_path = os.path.join(out_path, painted_rdmap_name)
        plt.savefig(painted_rdmap_path)
        print('-> painted rdmap saved to {}'.format(painted_rdmap_path))
    
