import torch
import torch.nn as nn
import torch.optim as op
import numpy as np
from torch.utils.data import Dataset,DataLoader
import os.path as osp
import os
import time
import logging

from model.joint import Joint_model
from model.gaitgraph.gaitgraph2 import GaitGraph2
from model.tdgcn import Model as TD
from model.ctrgcn import CTRGCN
model_list = {
  "TD":TD,
  "CTR":CTRGCN,
  "MIX":CTRGCN,
}


train_data_joint = np.load('./data/train_joint.npy').copy()
train_data_bone = np.load('./data/train_bone.npy').copy()
train_data_joint_motion = np.load('./data/train_joint_motion.npy').copy()
train_data_bone_motion = np.load('./data/train_bone_motion.npy').copy()

test_data_joint = np.load('./data/test_joint.npy').copy()
test_data_bone = np.load('./data/test_bone.npy').copy()
test_data_joint_motion = np.load('./data/test_joint_motion.npy').copy()
test_data_bone_motion = np.load('./data/test_bone_motion.npy').copy()


train_label = np.load('./data/train_label.npy').copy()
test_label = np.load('./data/test_label.npy').copy()
N, C, T, V, M = train_data_joint.shape


def get_logger(name):
  logger = logging.getLogger(__name__)
  logger.setLevel(level = logging.INFO)  
  handler = logging.FileHandler(name+".txt")
  handler.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  handler.setFormatter(formatter)
  
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  console.setFormatter(formatter)
 
  logger.addHandler(handler)
  logger.addHandler(console)
  return logger


class PreNormalize3D:
    """PreNormalize for NTURGB+D 3D keypoints (x, y, z). Codes adapted from https://github.com/lshiwjx/2s-AGCN. """

    def unit_vector(self, vector):
        """Returns the unit vector of the vector. """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """Returns the angle in radians between vectors 'v1' and 'v2'. """
        if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
            return 0
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def rotation_matrix(self, axis, theta):
        """Return the rotation matrix associated with counterclockwise rotation
        about the given axis by theta radians."""
        if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
            return np.eye(3)
        axis = np.asarray(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def __init__(self, zaxis=[0, 1], xaxis=[8, 4], align_spine=True, align_center=True):
        self.zaxis = zaxis
        self.xaxis = xaxis
        self.align_spine = align_spine
        self.align_center = align_center

    def __call__(self, ipts):
      # C, T, V, M
        skeleton = np.transpose(ipts,(3,1,2,0))

        M, T, V, C = skeleton.shape
       
        if skeleton.sum() == 0:
            return ipts

        T_new = skeleton.shape[1]

        if self.align_center:
            if skeleton.shape[2] == 25:
                main_body_center = skeleton[0, 0, 1].copy()
            else:
                main_body_center = skeleton[0, 0, -1].copy()
            skeleton = skeleton - main_body_center

        if self.align_spine:
            joint_bottom = skeleton[0, 0, self.zaxis[0]]
            joint_top = skeleton[0, 0, self.zaxis[1]]
            axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
            angle = self.angle_between(joint_top - joint_bottom, [0, 0, 1])
            matrix_z = self.rotation_matrix(axis, angle)
            skeleton = np.einsum('abcd,kd->abck', skeleton, matrix_z)

            joint_rshoulder = skeleton[0, 0, self.xaxis[0]]
            joint_lshoulder = skeleton[0, 0, self.xaxis[1]]
            axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            angle = self.angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            matrix_x = self.rotation_matrix(axis, angle)
            skeleton = np.einsum('abcd,kd->abck', skeleton, matrix_x)

        return np.transpose(skeleton,(3,1,2,0))

def norm(x):
  normlize = PreNormalize3D()
  for i in range(x.shape[0]):
    x[i] = normlize(x[i])
  return x

class data_loader(Dataset):
  def __init__(self):
    
    self.label = torch.from_numpy(train_label).cuda()
    self.joint = torch.from_numpy(norm(train_data_joint)).cuda().half()
    self.bone = torch.from_numpy(norm(train_data_bone)).cuda().half()
    self.joint_motion = torch.from_numpy(norm(train_data_joint_motion)).cuda().half()
    self.bone_motion = torch.from_numpy(norm(train_data_bone_motion)).cuda().half()
    # self.mix = np.concatenate((self.joint,self.bone,self.joint_motion),axis=1)
  
  def __getitem__(self,item):
    # N T V M C
    return {
            "joint":self.joint[item],
            "bone":self.bone[item],
            "joint_motion":self.joint_motion[item],
            "bone_motion":self.bone_motion[item],
            # "mix":self.norm(self.mix[item]),
            },self.label[item]
  def __len__(self):
    return len(self.label)
  
class test_loader(Dataset):
  def __init__(self) -> None:
    super().__init__()
    
    self.label = torch.from_numpy(test_label).cuda()
    self.joint = torch.from_numpy(norm(test_data_joint)).cuda().half()
    self.bone = torch.from_numpy(norm(test_data_bone)).cuda().half()
    self.joint_motion = torch.from_numpy(norm(test_data_joint_motion)).cuda().half()
    self.bone_motion = torch.from_numpy(norm(test_data_bone_motion)).cuda().half()
    # self.mix = np.concatenate((self.joint,self.bone,self.joint_motion),axis=1)
    
 
    
  def __getitem__(self,item):
    # N T V M C
    return {
            "joint":self.joint[item],
            "bone":self.bone[item],
            "joint_motion":self.joint_motion[item],
            "bone_motion":self.bone_motion[item],
            # "mix":self.norm(self.mix[item]),
            },self.label[item]
  def __len__(self):
    return len(self.label)
 

def get_result(model, mod):
    model.eval()
    testloader = DataLoader(test_loader(),batch_size = 20)
    res = 0
    with torch.no_grad():
      for batch, (train_data, _)in enumerate(testloader):
        pred = model(train_data[mod])
        pred = torch.argmax(pred,dim=1)
        res = torch.cat((res,pred),dim=-1)
    if os.path.exists("./res") == False:
        os.mkdir("./res")
    res = np.array(res)
    np.save("./res/"+mod+"_"+model+".npy",res)
        
    
def test(model, batch_size, mod):
    model.eval().half()
    correct = 0
    total = 0
    
    testloader = DataLoader(test_loader(),batch_size = batch_size)
    
    with torch.no_grad():
      for batch, (test_data, test_label)in enumerate(testloader):
        pred = model(test_data[mod])
        pred = torch.argmax(pred,dim=1)
        correct += torch.sum(pred==test_label)
        total += len(test_label)
    print("acc:",(correct/total).item())
    return (correct/total).item()

def train(epoch,batch_size, model_name, mod, args=3):
  
  model = model_list[model_name](in_channels=args).train().cuda().half()
  
  dataloader = DataLoader(data_loader(),batch_size = batch_size,shuffle=True)
  optimizer = op.SGD(model.parameters(), lr=0.01,momentum=0.9,weight_decay=0.0004)
  # schedule = op.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 1/(1+epoch))
  loss_fn =  nn.CrossEntropyLoss()
  
  logger = get_logger("./log/"+model_name+"_"+mod+"_"+time.strftime("%d %H:%M:%S"))
  logger.info("start training")
  
  print("start training")
  for e in range(epoch):
    logger.info(f"epoch:{e}")
    for batch, (train_data, train_label)in enumerate(dataloader):
      
      train_data = train_data[mod]
      
      optimizer.zero_grad()
      label = torch.zeros((batch_size,155)).cuda()
      for i in range(batch_size):
        label[i,train_label[i]]=1
      pred = model(train_data)
    
      loss = loss_fn(pred,label)
      
      loss.backward()
      optimizer.step()
      
      if batch%100 == 0:
        logger.info(f"batch:{batch}"+"   "+f"loss:{loss.item()}")
    
    # if e < 10:
    #   schedule.step()
    
    test(model, 16, mod)
    
    if e%16 == 15:
      logger.info(test(model, 16, mod))
      if os.path.exists("./ckpt") == False:
        os.mkdir("./ckpt")
      torch.save(model.state_dict(),osp.join(f"./ckpt/{mod}_{model_name}_{e}.pth"))  
 
 
# train(64,16, "CTR","joint") 
train(64,16, "CTR","bone")      
# train(64,16, "MIX","mix",9)
      