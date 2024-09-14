# (B,C,T,V,M)

# uav graph
# (10, 8), (8, 6), (9, 7), (7, 5), # arms
# (15, 13), (13, 11), (16, 14), (14, 12), # legs
# (11, 5), (12, 6), (11, 12), (5, 6), # torso
# (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2) # nose, eyes and ears


#                             3                   4
#                                  1        2
#                                      0
#
#                                5    ---     6
#                         /       |            |         \
#                    7                                        8
#               /
#          9                      |             |                    \
#                                                                       10
#
#                                |             |
#                                11  ------   12
#
#                               /               \
#
#                           13                      14
#                           /                          \

#                        15                               16

import numpy as np
from matplotlib import pyplot as plt
import os
from multiprocessing import Pool
import matplotlib.cm as cm
import cv2

head = [0, 1, 2, 3, 4]
arm = [5, 7, 9, 6, 8, 10]
leg = [11, 12, 13, 14, 15, 16]

graph = (
(10, 8), (8, 6), (9, 7), (7, 5), (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (11, 12), (5, 6), (5, 0),
(6, 0), (1, 0), (2, 0), (3, 1), (4, 2))

n = 0  # 从第n帧开始展示
m = 300  # 到第m帧结束，n<m<row

data = np.load('/home/gait/wuRenji/111/data/train_joint.npy')
N, C, T, V, M = data.shape


data = data.permute(4, 0, 2, 3, 1)# M, N, T, V, C
# data0 = data[0, 0]
# #T, V, C
# data0 = data0[:,:,[0,2]]
# for i in range(m-1):
#     xmin = np.min(data0[i,:,0])
#     xmax = np.max(data0[i,:,0])
#     ymin = np.min(data0[i,:,1])
#     ymax = np.max(data0[i,:,1])
    
#     data0[i,:,0] = ((data0[i,:,0] - xmin)/(xmax - xmin))*50 
#     data0[i,:,1] = ((data0[i,:,1] - ymin)/(ymax - ymin))*50 
#     # data0[i,:,1] -= data0[i,16,1]
    # data0[i,:,:] = np.clip(data0[i,:,:], 0, 50)



def save_fig(batch, i, fmt = 'png'):
    if os.path.exists(f"/home/gait/wuRenji/{fmt}/{batch}") == 0:
        os.makedirs(f"/home/gait/wuRenji/{fmt}/{batch}", exist_ok=True)
    plt.savefig(f'/home/gait/wuRenji/{fmt}/{batch}/{i}.png', dpi=50, bbox_inches='tight')


def p2d(batch):
    # print(batch)
    point = data[0, batch]
    for i in range(n, m // 3):
        i = i * 3
        plt.cla()
        plt.axis('off')
        plt.scatter(point[i, :, 0], point[i, :, 2], c="red")
        for (v1, v2) in graph:
            plt.plot((point[i, v1, 0], point[i, v2, 0]),
                     (point[i, v1, 2], point[i, v2, 2]),
                     c='green', lw=2.0)
        ax = plt.gca()
        ax.set_aspect('equal')
        save_fig(batch, i // 3)


def p3d():
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')
    for batch in range(1):
        point = data[0, batch]
        for i in range(n, 1):
            ax.scatter(point[i, :, 0], point[i, :, 1], point[i, :, 2], c="red")
            for (v1, v2) in graph:
                ax.plot3D([point[i, v1, 0], point[i, v2, 0]],
                          [point[i, v1, 1], point[i, v2, 1]],
                          [point[i, v1, 2], point[i, v2, 2]],
                          c='green', lw=2.0)
        ax.set_aspect('equal')
        save_fig(batch, i)


class GeneratePoseTarget:
    """Generate pseudo heatmaps based on joint coordinates and confidence.
    Required keys are "keypoint", "img_shape", "keypoint_score" (optional),
    added or modified keys are "imgs".
    Args:
        sigma (float): The sigma of the generated gaussian map. Default: 0.6.
        use_score (bool): Use the confidence score of keypoints as the maximum
            of the gaussian maps. Default: True.
        with_kp (bool): Generate pseudo heatmaps for keypoints. Default: True.
        with_limb (bool): Generate pseudo heatmaps for limbs. At least one of
            'with_kp' and 'with_limb' should be True. Default: False.
        skeletons (tuple[tuple]): The definition of human skeletons.
            Default: ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7), (7, 9),
                      (0, 6), (6, 8), (8, 10), (5, 11), (11, 13), (13, 15),
                      (6, 12), (12, 14), (14, 16), (11, 12)),
            which is the definition of COCO-17p skeletons.
        double (bool): Output both original heatmaps and flipped heatmaps.
            Default: False.
        left_kp (tuple[int]): Indexes of left keypoints, which is used when
            flipping heatmaps. Default: (1, 3, 5, 7, 9, 11, 13, 15),
            which is left keypoints in COCO-17p.
        right_kp (tuple[int]): Indexes of right keypoints, which is used when
            flipping heatmaps. Default: (2, 4, 6, 8, 10, 12, 14, 16),
            which is right keypoints in COCO-17p.
        left_limb (tuple[int]): Indexes of left limbs, which is used when
            flipping heatmaps. Default: (1, 3, 5, 7, 9, 11, 13, 15),
            which is left limbs of skeletons we defined for COCO-17p.
        right_limb (tuple[int]): Indexes of right limbs, which is used when
            flipping heatmaps. Default: (2, 4, 6, 8, 10, 12, 14, 16),
            which is right limbs of skeletons we defined for COCO-17p.
    """

    def __init__(self,
                 sigma=0.6,
                 use_score=False,
                 with_kp=True,
                 with_limb=False,
                 skeletons=((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7),
                            (7, 9), (0, 6), (6, 8), (8, 10), (5, 11), (11, 13),
                            (13, 15), (6, 12), (12, 14), (14, 16), (11, 12)),
                 double=False,
                 left_kp=(1, 3, 5, 7, 9, 11, 13, 15),
                 right_kp=(2, 4, 6, 8, 10, 12, 14, 16),
                 left_limb=(0, 2, 4, 5, 6, 10, 11, 12),
                 right_limb=(1, 3, 7, 8, 9, 13, 14, 15),
                 scaling=1.,
                 eps=1e-3,
                 img_h=128,
                 img_w=128):

        self.sigma = sigma
        self.use_score = use_score
        self.with_kp = with_kp
        self.with_limb = with_limb
        self.double = double
        self.eps = eps

        assert self.with_kp + self.with_limb == 1, ('One of "with_limb" and "with_kp" should be set as True.')
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.skeletons = skeletons
        self.left_limb = left_limb
        self.right_limb = right_limb
        self.scaling = scaling
        self.img_h = img_h
        self.img_w = img_w

    def generate_a_heatmap(self, arr, centers, max_values, point_center):
        """Generate pseudo heatmap for one keypoint in one frame.
        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
            centers (np.ndarray): The coordinates of corresponding keypoints (of multiple persons). Shape: 1 * 2.
            max_values (np.ndarray): The max values of each keypoint. Shape: (1, ).
            point_center: Shape: (1, 2)
        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        sigma = self.sigma
        img_h, img_w = arr.shape

        for center, max_value in zip(centers, max_values):
            if max_value < self.eps:
                continue

            mu_x, mu_y = center[0], center[1]

            tmp_st_x = int(mu_x - 3 * sigma)
            tmp_ed_x = int(mu_x + 3 * sigma)
            tmp_st_y = int(mu_y - 3 * sigma)
            tmp_ed_y = int(mu_y + 3 * sigma)

            st_x = max(tmp_st_x, 0)
            ed_x = min(tmp_ed_x + 1, img_w)
            st_y = max(tmp_st_y, 0)
            ed_y = min(tmp_ed_y + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma ** 2)
            patch = patch * max_value

            arr[st_y:ed_y, st_x:ed_x] = np.maximum(arr[st_y:ed_y, st_x:ed_x], patch)

    def generate_a_limb_heatmap(self, arr, starts, ends, start_values, end_values, point_center):
        """Generate pseudo heatmap for one limb in one frame.
        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
            starts (np.ndarray): The coordinates of one keypoint in the corresponding limbs. Shape: 1 * 2.
            ends (np.ndarray): The coordinates of the other keypoint in the corresponding limbs. Shape: 1 * 2.
            start_values (np.ndarray): The max values of one keypoint in the corresponding limbs. Shape: (1, ).
            end_values (np.ndarray): The max values of the other keypoint in the corresponding limbs. Shape: (1, ).
        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        sigma = self.sigma
        img_h, img_w = arr.shape

        for start, end, start_value, end_value in zip(starts, ends, start_values, end_values):
            value_coeff = min(start_value, end_value)
            if value_coeff < self.eps:
                continue

            min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
            min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

            tmp_min_x = int(min_x - 3 * sigma)
            tmp_max_x = int(max_x + 3 * sigma)
            tmp_min_y = int(min_y - 3 * sigma)
            tmp_max_y = int(max_y + 3 * sigma)

            min_x = max(tmp_min_x, 0)
            max_x = min(tmp_max_x + 1, img_w)
            min_y = max(tmp_min_y, 0)
            max_y = min(tmp_max_y + 1, img_h)

            x = np.arange(min_x, max_x, 1, np.float32)
            y = np.arange(min_y, max_y, 1, np.float32)

            if not (len(x) and len(y)):
                continue

            y = y[:, None]
            x_0 = np.zeros_like(x)
            y_0 = np.zeros_like(y)

            # distance to start keypoints
            d2_start = ((x - start[0]) ** 2 + (y - start[1]) ** 2)

            # distance to end keypoints
            d2_end = ((x - end[0]) ** 2 + (y - end[1]) ** 2)

            # the distance between start and end keypoints.
            d2_ab = ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

            if d2_ab < 1:
                self.generate_a_heatmap(arr, start[None], start_value[None], point_center)
                continue

            coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

            a_dominate = coeff <= 0
            b_dominate = coeff >= 1
            seg_dominate = 1 - a_dominate - b_dominate

            position = np.stack([x + y_0, y + x_0], axis=-1)
            projection = start + np.stack([coeff, coeff], axis=-1) * (end - start)
            d2_line = position - projection
            d2_line = d2_line[:, :, 0] ** 2 + d2_line[:, :, 1] ** 2
            d2_seg = a_dominate * d2_start + b_dominate * d2_end + seg_dominate * d2_line

            patch = np.exp(-d2_seg / 2. / sigma ** 2)
            patch = patch * value_coeff

            arr[min_y:max_y, min_x:max_x] = np.maximum(arr[min_y:max_y, min_x:max_x], patch)

    def generate_heatmap(self, arr, kps, max_values):
        """Generate pseudo heatmap for all keypoints and limbs in one frame (if
        needed).
        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: V * img_h * img_w.
            kps (np.ndarray): The coordinates of keypoints in this frame. Shape: 1 * V * 2.
            max_values (np.ndarray): The confidence score of each keypoint. Shape: 1 * V.
        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        point_center = kps.mean(1)

        if self.with_kp:
            num_kp = kps.shape[1]
            for i in range(num_kp):
                self.generate_a_heatmap(arr[i], kps[:, i], max_values[:, i], point_center)

        if self.with_limb:
            for i, limb in enumerate(self.skeletons):
                start_idx, end_idx = limb
                starts = kps[:, start_idx]
                ends = kps[:, end_idx]

                start_values = max_values[:, start_idx]
                end_values = max_values[:, end_idx]
                self.generate_a_limb_heatmap(arr[i], starts, ends, start_values, end_values, point_center)

    def gen_an_aug(self, pose_data):
        """Generate pseudo heatmaps for all frames.
        Args:
            pose_data (array): [1, T, V, C]
        Returns:
            list[np.ndarray]: The generated pseudo heatmaps.
        """

        all_kps = pose_data[..., :2]
        kp_shape = pose_data.shape  # [1, T, V, 2]

        if pose_data.shape[-1] == 3:
            all_kpscores = pose_data[..., -1]  # [1, T, V]
        else:
            all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)

        # scale img_h, img_w and kps
        img_h = int(self.img_h * self.scaling + 0.5)
        img_w = int(self.img_w * self.scaling + 0.5)
        all_kps[..., :2] *= self.scaling

        num_frame = kp_shape[1]
        num_c = 0
        if self.with_kp:
            num_c += all_kps.shape[2]
        if self.with_limb:
            num_c += len(self.skeletons)
        ret = np.zeros([num_frame, num_c, img_h, img_w], dtype=np.float32)

        for i in range(num_frame):
            # 1, V, C
            kps = all_kps[:, i]
            # 1, V
            kpscores = all_kpscores[:, i] if self.use_score else np.ones_like(all_kpscores[:, i])

            self.generate_heatmap(ret[i], kps, kpscores)
        return ret

    def __call__(self, pose_data):
        """
        pose_data: (T, V, C=3/2)
        1: means person number
        """
        pose_data = pose_data[None, ...]  # (1, T, V, C=3/2)
        # print(pose_data[0,0])
        heatmap = self.gen_an_aug(pose_data)

        if self.double:
            indices = np.arange(heatmap.shape[1], dtype=np.int64)
            left, right = (self.left_kp, self.right_kp) if self.with_kp else (self.left_limb, self.right_limb)
            for l, r in zip(left, right):  # noqa: E741
                indices[l] = r
                indices[r] = l
            heatmap_flip = heatmap[..., ::-1][:, indices]
            heatmap = np.concatenate([heatmap, heatmap_flip])
        return heatmap

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sigma={self.sigma}, '
                    f'use_score={self.use_score}, '
                    f'with_kp={self.with_kp}, '
                    f'with_limb={self.with_limb}, '
                    f'skeletons={self.skeletons}, '
                    f'double={self.double}, '
                    f'left_kp={self.left_kp}, '
                    f'right_kp={self.right_kp})')
        return repr_str


class HeatmapToImage:
    """
    Convert the heatmap data to image data.
    """

    def __init__(self) -> None:
        self.cmap = cm.gray

    def __call__(self, heatmaps):
        """
        heatmaps: (T, 17, H, W)
        return images: (T, 1, H, W)
        """
        heatmaps = [x.transpose(1, 2, 0) for x in heatmaps]
        h, w, _ = heatmaps[0].shape
        newh, neww = int(h), int(w)
        heatmaps = [np.max(x, axis=-1) for x in heatmaps]
        heatmaps = [(self.cmap(x)[..., :3] * 255).astype(np.uint8) for x in heatmaps]
        heatmaps = [cv2.resize(x, (neww, newh)) for x in heatmaps]
        return np.ascontiguousarray(np.mean(np.array(heatmaps), axis=-1, keepdims=True).transpose(0, 3, 1, 2))

def save_heat(batch, i, heat, fmt = 'png'):
    if os.path.exists(f"/home/gait/wuRenji/{fmt}/{batch}") == 0:
        os.makedirs(f"/home/gait/wuRenji/{fmt}/{batch}", exist_ok=True)
    # plt.savefig(f'/home/gait/wuRenji/{fmt}/{batch}/{i}.png', dpi=50, bbox_inches='tight')
    plt.imsave(f'/home/gait/wuRenji/{fmt}/{batch}/{i}.png',heat[i])

def get_norm(input):
    imax = np.max(input)
    imin = np.min(input)
    input = (input-imin)/(imax - imin)
    return input

def Heatmap(batch):
    joint_data = data[0, batch]
    joint_data = joint_data[:,:,[0,2]]
    for i in range(m):
        xmin = np.min(joint_data[i,:,0])
        xmax = np.max(joint_data[i,:,0])
        ymin = np.min(joint_data[i,:,1])
        ymax = np.max(joint_data[i,:,1])
    
        joint_data[i,:,0] = 128 - ((joint_data[i,:,0] - xmin)/(xmax - xmin))*120 
        joint_data[i,:,1] = 130 - ((joint_data[i,:,1] - ymin)/(ymax - ymin))*120
    
    
    heatmap_kp = GeneratePoseTarget(with_kp=1,with_limb=0,sigma=8)
    heatmap_lm = GeneratePoseTarget(with_kp=0,with_limb=1,sigma=8)
    
    
    heat_data_kp = heatmap_kp(joint_data) 
    heat_data_lm = heatmap_lm(joint_data)
    # print(heat_data.shape)
    heat_img_manager = HeatmapToImage()
    heat_img_kp = heat_img_manager(heat_data_kp)
    heat_img_lm = heat_img_manager(heat_data_lm)
    heat_img_kp = get_norm(heat_img_kp[:,0])
    heat_img_lm = get_norm(heat_img_lm[:,0])
    
    #T,H,Wnp.zeros_like(heat_img_kp)
    heat_img = np.stack([heat_img_lm,heat_img_kp,np.zeros_like(heat_img_kp)],axis=-1)
    # print(heat_img.shape)
    for i in range(n,m//10):
        save_heat(batch,i,heat_img,"heat")
        
    


def multi(task):
    with Pool(processes=os.cpu_count()) as pool:  # 使用所有可用CPU核心
        pool.map(task, range(2000))

multi(Heatmap)
# heatmap_kp = GeneratePoseTarget(with_kp=1,with_limb=0,sigma=1)
# heatmap_lm = GeneratePoseTarget(with_kp=0,with_limb=1,sigma=0.2)

# heat_data = heatmap_kp(data0) + heatmap_lm(data0)
# # print(heat_data.shape)
# heat_img_manager = HeatmapToImage()
# heat_img = heat_img_manager(heat_data)
# # print(heat_img[0,0])
# plt.imsave(f"./demo.png",heat_img[0,0])
# # print(heat_img.shape)
# # print(heat_data.shape)
# multi(p2d)
