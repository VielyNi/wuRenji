import torch
import torch.nn as nn
import torch.optim as op
import numpy as np
from torch.utils.data import Dataset,DataLoader
import os.path as osp
import os
import logging

from model.joint import Joint_model
from model.gaitgraph.gaitgraph2 import GaitGraph2
from model.tdgcn import Model as TD
from model.ctrgcn import CTRGCN
model_list = {
  "joint":Joint_model,
  "graph":GaitGraph2,
  "TD":TD,
  "CTR":CTRGCN
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

class data_loader(Dataset):
  def __init__(self):
    self.label = torch.from_numpy(train_label).cuda()
    self.joint = torch.from_numpy(train_data_joint).cuda()
    self.bone = torch.from_numpy(train_data_bone).cuda()
    self.joint_motion = torch.from_numpy(train_data_joint_motion).cuda()
    self.bone_motion = torch.from_numpy(train_data_bone_motion).cuda()
  
  def __getitem__(self,item):
    # N T V M C
    return {
            "joint":self.joint[item],
            "bone":self.bone[item],
            "joint_motion":self.joint_motion[item],
            "bone_motion":self.bone_motion[item],
            },self.label[item]
  def __len__(self):
    return len(self.label)
  
class test_loader(Dataset):
  def __init__(self) -> None:
    super().__init__()
    self.label = torch.from_numpy(test_label).cuda()
    self.joint = torch.from_numpy(test_data_joint).cuda()
    self.bone = torch.from_numpy(test_data_bone).cuda()
    self.joint_motion = torch.from_numpy(test_data_joint_motion).cuda()
    self.bone_motion = torch.from_numpy(test_data_bone_motion).cuda()
  def __getitem__(self,item):
    # N T V M C
    return {
            "joint":self.joint[item],
            "bone":self.bone[item],
            "joint_motion":self.joint_motion[item],
            "bone_motion":self.bone_motion[item],
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
        pred = model(test_data[mod].cuda().half())
        pred = torch.argmax(pred,dim=1)
        correct += torch.sum(pred==test_label.cuda())
        total += len(test_label)
    print("acc:",correct/total)

def train(epoch,batch_size, model_name, mod):
  model = model_list[model_name]().train().cuda().half()
  dataloader = DataLoader(data_loader(),batch_size = batch_size,shuffle=True)
  optimizer = op.SGD(model.parameters(), lr=0.1,momentum=0.9,weight_decay=0.0004)
  schedule = op.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 32/(epoch+32))
  loss_fn =  nn.CrossEntropyLoss()
  
  logger = get_logger("./log/"+model_name+"_"+mod)
  logger.info("start training")
  
  print("start training")
  for e in range(epoch):
    logger.info(f"epoch:{e}")
    for batch, (train_data, train_label)in enumerate(dataloader):
      
      optimizer.zero_grad()
      label = torch.zeros((batch_size,155)).cuda()
      for i in range(batch_size):
        label[i,train_label[i]]=1
      pred = model(train_data[mod].half())
    
      loss = loss_fn(pred,label)
      
      loss.backward()
      optimizer.step()
      if batch%batch_size == 0:
        logger.info(f"batch:{batch}"+"   "+f"loss:{loss.item()}")
    schedule.step()
    test(model, 16, mod)
    if e%10 == 0:
      test(model, 16, mod)
      if os.path.exists("./ckpt") == False:
        os.mkdir("./ckpt")
      torch.save(model.state_dict(),osp.join(f"./ckpt/{mod}_{model_name}_{e}.pth"))  
  #get_result(model, mod)
train(64,16, "CTR","joint")
      
      
