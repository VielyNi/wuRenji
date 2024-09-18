import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import logging 
from torch.utils.data import Dataset,DataLoader

from model.joint import Joint_model
from model.gaitgraph.gaitgraph2 import GaitGraph2
from model.tdgcn import Model as TD
from model.ctrgcn import CTRGCN

model_list = {
  "joint":Joint_model,
  "graph":GaitGraph2,
  "TD":TD,
  "CTR":CTRGCN,
}


test_data_joint = np.load('./data/test_joint.npy').copy()
test_data_bone = np.load('./data/test_bone.npy').copy()
test_data_joint_motion = np.load('./data/test_joint_motion.npy').copy()
test_data_bone_motion = np.load('./data/test_bone_motion.npy').copy()

test_label = np.load('./data/test_label.npy').copy()
label = torch.from_numpy(test_label).cuda()

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

class test_loader(Dataset):
  def __init__(self) -> None:
    super().__init__()
    self.label = torch.from_numpy(test_label).cuda()
    self.joint = torch.from_numpy(test_data_joint).cuda()
    self.bone = torch.from_numpy(test_data_bone).cuda()
    self.joint_motion = torch.from_numpy(test_data_joint_motion).cuda()
    self.bone_motion = torch.from_numpy(test_data_bone_motion).cuda()

  def __getitem__(self,index):
    # N T V M C
    return {
            "joint":self.joint[index],
            "bone":self.bone[index],
            "joint_motion":self.joint_motion[index],
            "bone_motion":self.bone_motion[index],
            },self.label[index]
  def __len__(self):
    return len(self.label)
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
            
        p = torch.sum(pred,dim=0)
        p = torch.argmax(p,dim=1)
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

names = ["bone_CTR_60","joint_CTR_60","joint_TD_60"]
models = [CTRGCN(),CTRGCN(),TD()]
mods = ["bone","joint","joint"]


get_result(names,mods,models)  