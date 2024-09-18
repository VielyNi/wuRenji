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
from model.tdgcn_GCL import Model
from model.ctrgcn import CTRGCN
from model.loss_GCL import InfoNCEGraph

model_list = {
  "joint":Joint_model,
  "graph":GaitGraph2,
  "TD":Model,
  "CTR":CTRGCN,
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
  
  def __getitem__(self,index):
    # N T V M C
    return {
            "joint":self.joint[index],
            "bone":self.bone[index],
            "joint_motion":self.joint_motion[index],
            "bone_motion":self.bone_motion[index],
            },self.label[index],index
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
  def __getitem__(self,index):
    # N T V M C
    return {
            "joint":self.joint[index],
            "bone":self.bone[index],
            "joint_motion":self.joint_motion[index],
            "bone_motion":self.bone_motion[index],
            },self.label[index],index
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
    model.eval()
    correct = 0
    total = 0
    
    testloader = DataLoader(test_loader(),batch_size = batch_size)
    
    with torch.no_grad():
      for batch, (test_data, test_label,index)in enumerate(testloader):
        pred,_ = model(test_data[mod])
        pred = torch.argmax(pred,dim=1)
        correct += torch.sum(pred==test_label)
        total += len(test_label)
    print("acc:",correct/total)

def train(epoch,batch_size, model_name, mod):
  model = model_list[model_name]().train().cuda()

  
  dataloader = DataLoader(data_loader(),batch_size = batch_size,shuffle=True)
  optimizer = op.SGD(model.parameters(), lr=0.01,momentum=0.9,weight_decay=0.0004)
  #schedule = op.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 64/(epoch+64))
  loss_fn =  nn.CrossEntropyLoss()
  graphContrast = InfoNCEGraph(in_channels=3*17*17, out_channels=256, class_num=155,\
            mem_size=dataloader.dataset.__len__(), label_all=dataloader.dataset.label, T=0.8).cuda()
  logger = get_logger("./log/"+model_name+"_"+mod)
  logger.info("start training")
  
  print("start training")
  for e in range(epoch):
    logger.info(f"epoch:{e}")
    for batch, (train_data, train_label, index)in enumerate(dataloader):
      
      label = torch.zeros((batch_size,155)).cuda()
      for i in range(batch_size):
        label[i,train_label[i]]=1
      pred,graph = model(train_data[mod])
    
      loss = loss_fn(pred,label)
      if graph is not None:
          contrast_loss = graphContrast(graph, label, index)
      else:
          contrast_loss = torch.zeros(1, device=pred.device)
      if contrast_loss > 0:
          loss = loss + contrast_loss
      optimizer.zero_grad()    
      loss.backward()
      optimizer.step()
      #schedule.step()
      if batch%batch_size == 0:
        logger.info(f"batch:{batch}"+"   "+f"loss:{loss.item()}")
    test(model, 16, mod)
    if e%10 == 0:
      test(model, 16, mod)
      if os.path.exists("./ckpt") == False:
        os.mkdir("./ckpt")
      torch.save(model.state_dict(),osp.join(f"./ckpt/{mod}_{model_name}_{e}_GCL.pth"))  
  #get_result(model, mod)
train(64,16, "TD","joint")
      
      
