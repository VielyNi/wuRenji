import torch
import torch.nn as nn
import torch.optim as op
import numpy as np
from torch.utils.data import Dataset,DataLoader
import os.path as osp

from model.joint import Joint_model
from model.gaitgraph.gaitgraph2 import GaitGraph2
from model.ctrgcn import Model

model_list = {
  "joint":Joint_model,
  "graph":GaitGraph2,
  "CTR":Model
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

class data_loader(Dataset):
  def __init__(self):
    self.label = torch.from_numpy(train_label).cuda()
    self.joint = torch.from_numpy(train_data_joint).cuda()
    self.bone = torch.from_numpy(train_data_bone).cuda()
    self.joint_motion = torch.from_numpy(train_data_joint_motion).cuda()
    self.bone_motion = torch.from_numpy(train_data_bone_motion).cuda()
  
  def __getitem__(self,item):
    # N T V M C
    return self.joint[item],self.label[item]
  def __len__(self):
    return len(self.label)
  
class test_loader(Dataset):
  def __init__(self) -> None:
    super().__init__()
    self.label = torch.from_numpy(test_label).cuda()
    self.joint = torch.from_numpy(test_data_joint).cuda()
    self.bone = torch.from_numpy(test_data_bone).cuda()
    self.joint_motion = torch.from_numpy(test_data_joint_motion).permute()
  def __getitem__(self,item):
    # N T V M C
    return self.joint[item],self.label[item]
  def __len__(self):
    return len(self.label)

def test(model, batch_size):
    model.eval()
    correct = 0
    total = 0
    
    testloader = DataLoader(test_loader(),batch_size = batch_size)
    
    with torch.no_grad():
      for batch, (train_data, train_label)in enumerate(testloader):
        pred = model(train_data)
        pred = torch.argmax(pred,dim=1)
        correct += torch.sum(pred==train_label)
        total += len(train_label)
    print(correct/total)

def train(epoch,batch_size, model_name):
  model = model_list[model_name]().train().cuda()
  dataloader = DataLoader(data_loader(),batch_size = batch_size)
  optimizer = op.Adam(model.parameters(), lr=0.01)
  schedule = op.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 1/(epoch+1))
  loss_fn =  nn.CrossEntropyLoss()
  
  print("start training")
  for e in range(epoch):
    print("epoch:",e)
    for batch, (train_data, train_label)in enumerate(dataloader):
      
      optimizer.zero_grad()
      label = torch.zeros((batch_size,155)).cuda()
      for i in range(batch_size):
        label[i,train_label[i]]=1
      # b, t, v, m, c  = train_data.shape
      pred = model(train_data)
    
      loss = loss_fn(pred,label)
      
      loss.backward()
      optimizer.step()
      schedule.step()
      
    if batch%10 == 0:
        print("loss:",loss)
      
    if e%10 == 0:
      test(model)
      torch.save(model.state_dict(),osp.join("./ckpt",model_name+"_{e}"+".pth"))  
train(10,16, "CTR")
      
      
