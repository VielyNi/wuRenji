import numpy as np

x = np.load('/home/gait/wuRenji/src/data/test_joint_A.npy',mmap_mode='r')
print(x.shape)

x = np.load('/home/gait/wuRenji/src/data/test_label_A.npy',mmap_mode='r')
print(x.shape)
print(x.max())

#x = np.load('data/test_joint_B.npy',mmap_mode='r')
#print(x.shape)