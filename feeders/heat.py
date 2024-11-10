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
                 eps=0,
                 img_h=182//2,
                 img_w=182):

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
                if kps[:, i,1] == np.inf:
                    import pdb 
                    pdb.set_trace()
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


def get_norm(input):
    imax = np.max(input)
    imin = np.min(input)
    input = (input-imin)/(imax - imin)
    return input

def pooling(data, pool_size=300//13):
    c, n, m, l  = data.shape
    pooled_data = []
    for i in range(0, n, pool_size):
        pooled_data.append((np.mean(data[:,i:i+pool_size], axis=1)+np.max(data[:,i:i+pool_size], axis=1))/2)
    return np.array(pooled_data)

def norm(joint_data):
    min = np.min(joint_data,axis=0)
    max = np.max(joint_data,axis=0)
    joint_data = (joint_data-min)
    ma = (max-min)
    ma[ma==0] = 1
    joint_data = joint_data/ma
    joint_data[:,:,0] = 151 - 120*joint_data[:,:,0]
    joint_data[:,:,1] = 75 - 60*joint_data[:,:,1]
    return joint_data
    

def Heatmap(joint_data):
    # TMVC
    joint_data = joint_data.transpose(3,1,2,0)
    joint_data = pooling(joint_data)
    joint_data_l = joint_data[:,0]
    joint_data_r = joint_data[:,1]
    
    
    
    #T V C
    joint_data1 = np.stack((joint_data_l[:,:,0],joint_data_l[:,:,1]),axis=-1)
    joint_data2 = np.stack((joint_data_l[:,:,0],joint_data_l[:,:,2]),axis=-1)
    joint_data3 = np.stack((joint_data_l[:,:,1],joint_data_l[:,:,2]),axis=-1)
    joint_data4 = np.stack((joint_data_r[:,:,0],joint_data_r[:,:,1]),axis=-1)
    joint_data5 = np.stack((joint_data_r[:,:,0],joint_data_r[:,:,2]),axis=-1)
    joint_data6 = np.stack((joint_data_r[:,:,1],joint_data_r[:,:,2]),axis=-1)
    
    joint_data1 = norm(joint_data1)
    joint_data2 = norm(joint_data2)
    joint_data3 = norm(joint_data3)
    joint_data4 = norm(joint_data4)
    joint_data5 = norm(joint_data5)
    joint_data6 = norm(joint_data6)
    
    heatmap_kp = GeneratePoseTarget(with_kp=1,with_limb=0,sigma=8)
    heatmap_lm = GeneratePoseTarget(with_kp=0,with_limb=1,sigma=1)
    
    
    heat_data_kp1 = heatmap_kp(joint_data1) 
    heat_data_kp2 = heatmap_kp(joint_data2)
    heat_data_kp3 = heatmap_kp(joint_data3)
    heat_data_kp4 = heatmap_kp(joint_data4)
    heat_data_kp5 = heatmap_kp(joint_data5)
    heat_data_kp6 = heatmap_kp(joint_data6)
    
    heat_data_lm1 = heatmap_lm(joint_data1)
    heat_data_lm2 = heatmap_lm(joint_data2)
    heat_data_lm3 = heatmap_lm(joint_data3)
    heat_data_lm4 = heatmap_lm(joint_data4)
    heat_data_lm5 = heatmap_lm(joint_data5)
    heat_data_lm6 = heatmap_lm(joint_data6)
  
    heat_img_manager = HeatmapToImage()
    heat_img_kp1 = heat_img_manager(heat_data_kp1)
    heat_img_kp2 = heat_img_manager(heat_data_kp2)
    heat_img_kp3 = heat_img_manager(heat_data_kp3)
    heat_img_kp4 = heat_img_manager(heat_data_kp4)
    heat_img_kp5 = heat_img_manager(heat_data_kp5)
    heat_img_kp6 = heat_img_manager(heat_data_kp6)
    
    kp1 = np.concatenate([heat_img_kp1,heat_img_kp4],axis=2)
    kp2 = np.concatenate([heat_img_kp2,heat_img_kp5],axis=2)
    kp3 = np.concatenate([heat_img_kp3,heat_img_kp6],axis=2)
    
    kp1 = get_norm(kp1[:,0])
    kp2 = get_norm(kp2[:,0])
    kp3 = get_norm(kp3[:,0])
   
    heat_img_lm1 = heat_img_manager(heat_data_lm1)
    heat_img_lm2 = heat_img_manager(heat_data_lm2)
    heat_img_lm3 = heat_img_manager(heat_data_lm3)
    heat_img_lm4 = heat_img_manager(heat_data_lm4)
    heat_img_lm5 = heat_img_manager(heat_data_lm5)
    heat_img_lm6 = heat_img_manager(heat_data_lm6)
    
    lm1 = np.concatenate([heat_img_lm1,heat_img_lm4],axis=2)
    lm2 = np.concatenate([heat_img_lm2,heat_img_lm5],axis=2)
    lm3 = np.concatenate([heat_img_lm3,heat_img_lm6],axis=2)
    
    lm1 = get_norm(lm1[:,0])
    lm2 = get_norm(lm2[:,0])
    lm3 = get_norm(lm3[:,0])
    
    im1 = (kp1-lm1+1)/2
    im2 = (kp2-lm2+1)/2
    im3 = (kp3-lm3+1)/2
    
    heat_img = np.stack([im1,im2,im3],axis=-1)
    heat_img = heat_img.transpose(3,0,1,2)
    return heat_img


if __name__ == '__main__':
    import tqdm
    sets = {'train', 'test', 'val'}
    for set in sets:
        data = np.load(f"/home/niyunfei/workspace/wuRenji/wuRenji/data/{set}_joint.npy")
        for i in tqdm.tqdm(range(data.shape[0])):
            heat = Heatmap(data[i])
            np.save(f'/home/niyunfei/workspace/wuRenji/wuRenji/data/heat/{set}_{i}.npy',heat)
    # heat = []
    # import tqdm
    # for i in tqdm.tqdm(range(data.shape[0])):
    #     heat.append(Heatmap(data[i]))
    # heat = np.stack(heat,axis=0)
# data = np.ones((3,300,17,2))
# data = Heatmap(data)

