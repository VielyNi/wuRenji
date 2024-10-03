import torch
import torch.nn as nn
import torch.optim as op

import numpy as np
import os
import os.path as osp
import logging 
from torch.utils.data import Dataset,DataLoader

from model.tdgcn import Model as TD
from model.ctrgcn import CTRGCN

model_list = {
  "TD":TD,
  "CTR":CTRGCN,
}


test_data_joint = np.load('./data/test_joint.npy').copy()
test_data_bone = np.load('./data/test_bone.npy').copy()
test_data_joint_motion = np.load('./data/test_joint_motion.npy').copy()
test_data_bone_motion = np.load('./data/test_bone_motion.npy').copy()

train_data_joint = np.load('./data/train_joint.npy').copy()
train_data_bone = np.load('./data/train_bone.npy').copy()
train_data_joint_motion = np.load('./data/train_joint_motion.npy').copy()
train_data_bone_motion = np.load('./data/train_bone_motion.npy').copy()

train_label = np.load('./data/train_label.npy').copy()
test_label = np.load('./data/test_label.npy').copy()
label = torch.from_numpy(test_label).cuda()
train_vm_label = torch.from_numpy(train_label).cuda()

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

# class test_loader(Dataset):
#   def __init__(self) -> None:
#     super().__init__()
#     self.label = torch.from_numpy(test_label).cuda()
#     self.joint = torch.from_numpy(test_data_joint).cuda()
#     self.bone = torch.from_numpy(test_data_bone).cuda()
#     self.joint_motion = torch.from_numpy(test_data_joint_motion).cuda()
#     self.bone_motion = torch.from_numpy(test_data_bone_motion).cuda()

#   def __getitem__(self,index):
#     # N T V M C
#     return {
#             "joint":self.joint[index],
#             "bone":self.bone[index],
#             "joint_motion":self.joint_motion[index],
#             "bone_motion":self.bone_motion[index],
#             },self.label[index]
#   def __len__(self):
#     return len(self.label)

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

class att(nn.Module):
  def __init__(self,in_ch):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Linear(in_ch,in_ch//2),
        nn.ReLU(),
        nn.Linear(in_ch//2,1)
    ) 

  def forward(self,x):
    return self.conv(x)


def get_result(model_names, mods, models):
    model = []
    logger = get_logger("test")
    for i in range(len(model_names)):
        model.append(models[i])
        model[i].load_state_dict(torch.load(f"./ckpt/{model_names[i]}.pth"))
        model[i].eval().cuda().half()
 
    testloader = DataLoader(test_loader(),batch_size = 20)

    with torch.no_grad():
        pred = torch.zeros((len(model),2000,155)).cuda()
        for i in range(len(model)):
            for batch, (train_data, _)in enumerate(testloader):
                pred[i,batch*20:(batch+1)*20] = model[i](train_data[mods[i]].half())
            idx = torch.argmax(pred[i],dim=1)
            acc = torch.sum(idx==label)/2000
            logger.info(f"{model_names[i]}:{acc}")
            
        # p = torch.sum(pred,dim=0)
        # p = torch.argmax(p,dim=1)
        
    # # self att
    # loss_fn =  nn.CrossEntropyLoss()
    # att_add = att(len(model)).cuda()
    # optimizer = op.SGD(att_add.parameters(), lr=0.01,momentum=0.9,weight_decay=0.0004)
        
    # label_one_hot = torch.zeros((2000,155)).cuda()
    # for i in range(2000):
    #   label_one_hot[i,label[i]] = 1
        
    # for i in range(5000):
    #   optimizer.zero_grad()
    #   p = att_add(pred.permute(2,1,0)).permute(2,1,0).squeeze(0)
    #   loss = loss_fn(p,label_one_hot)
    #   loss.backward()
    #   optimizer.step()
    
    att_add = att(len(model)).cuda()   
    att_add.load_state_dict(torch.load("./ckpt/vm.pth"))   
    p = att_add(pred.permute(2,1,0)).permute(2,1,0).squeeze(0)
    p = p.argmax(dim=1)
     
    acc = torch.sum(p==label)/2000
    logger.info(f"pred:{acc}")
    res = pred.cpu().numpy()
   
    if os.path.exists("./res") == False:
        os.mkdir("./res")
    res = np.array(res)
    if len(model_names) == 1:
        name = model_names[0]
    else :
        name = "vm"
    np.save(f"./res/{name}.npy",res)

def train_vm(model_names, mods, models):
    model = []
    logger = get_logger("test")
    for i in range(len(model_names)):
        model.append(models[i])
        model[i].load_state_dict(torch.load(f"./ckpt/{model_names[i]}.pth"))
        model[i].eval().cuda().half()
 
    testloader = DataLoader(data_loader(),batch_size = 20)

    with torch.no_grad():
        pred = torch.zeros((len(model),16432,155)).cuda()
        for i in range(len(model)):
            for batch, (train_data, _)in enumerate(testloader):
                pred[i,batch*20:(batch+1)*20] = model[i](train_data[mods[i]].half())
            idx = torch.argmax(pred[i],dim=1)
            acc = torch.sum(idx==train_vm_label)/16432
            logger.info(f"{model_names[i]}:{acc}")
            
        # p = torch.sum(pred,dim=0)
        # p = torch.argmax(p,dim=1)
        
    # self att
    loss_fn =  nn.CrossEntropyLoss()
    att_add = att(len(model)).cuda()
    optimizer = op.SGD(att_add.parameters(), lr=0.01,momentum=0.9,weight_decay=0.0004)
        
    label_one_hot = torch.zeros((16432,155)).cuda()
    for i in range(16432):
      label_one_hot[i,train_vm_label[i]] = 1
        
    for i in range(2000):
      optimizer.zero_grad()
      p = att_add(pred.permute(2,1,0)).permute(2,1,0).squeeze(0)
      loss = loss_fn(p,label_one_hot)
      loss.backward()
      optimizer.step()
          
    p = att_add(pred.permute(2,1,0)).permute(2,1,0).squeeze(0)
    p = p.argmax(dim=1)
     
    acc = torch.sum(p==train_vm_label)/16432
    logger.info(f"pred:{acc}")
    res = pred.cpu().numpy()
   
    if os.path.exists("./res") == False:
        os.mkdir("./res")
    res = np.array(res)
    if len(model_names) == 1:
        name = model_names[0]
    else :
        name = "vm"
    np.save(f"./res/{name}.npy",res)
    torch.save(att_add.state_dict(),"./ckpt/vm.pth")  
    logger.info("end train vm")
    
    
    
    

names = ["bone_CTR_78","joint_CTR_70",]
        #  "joint_motion_CTR_70",
        # "bone_motion_CTR_31",
        #  "bone_TD_31"]
models = [CTRGCN(),CTRGCN(),]
          # CTRGCN(),
          # CTRGCN(),
          # TD()]
mods = ["bone","joint",]
        # "joint_motion",
        # "bone_motion",
        # "bone"]

train_vm(names,mods,models)
get_result(names,mods,models)  