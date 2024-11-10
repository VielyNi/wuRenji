import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as op
import pdb

directory = './ver' 

skip = []
data_list = []
name_list = []
count = -1
for root, dirs, files in os.walk(directory):
    for file in files:
       
        if file.endswith('.npy'):
            count += 1
            file_path = os.path.join(root, file)
           
            data = np.load(file_path)
            # print(f"{count}Loaded {file_path}, shape: {data.shape}")
            if count in skip:
              continue
            print(f"{len(data_list)}Loaded {file_path}, shape: {data.shape}")
            data_list.append(data)
            name_list.append(file_path)


label = np.stack(data_list,axis=2)
label = torch.from_numpy(label).cuda()

def get_vote(weight=[1 for _ in range(len(data_list))]):
  vote = 0
  for i in range(len(data_list)):
      vote += data_list[i]*weight[i]
  return vote
  
def save_pred(weight=[1 for _ in range(len(data_list))]):
  vote = get_vote(weight)
  np.save('pred.npy',vote)
  print(vote.shape)
  
def get_single_acc(truth, data,index):
  data = data.argmax(axis=1)
  acc = np.sum(data==truth)/len(truth)
  print(f'{index}  {name_list[index]} acc:{acc}')
  
def get_acc(weight=[1 for _ in range(len(data_list))],show = False):
  truth = np.load('./data/val_label.npy')
  vote = get_vote(weight)
  vote = vote.argmax(axis=1)
  acc = np.sum(vote==truth)/len(truth)
  
  for i in range(len(data_list)):
    get_single_acc(truth,data_list[i],i)
  
  if show:
    print(f'acc:{acc}')
  return acc
  
def search(epoch,weight=[1 for _ in range(len(data_list))],acc=0):
  acc_max = acc
  weight_max = weight
  for _ in range(epoch):
    weight = [weight[i]+np.random.uniform(-0.3,0.3) for i in range(len(weight))]
    acc = get_acc(weight)
    if acc > acc_max:
      acc_max = acc
      weight_max = weight
      print(f'best acc:{acc_max},\nbest weight:{weight_max}')
  # pdb.set_trace()
      

if __name__ == '__main__':
    save_pred()
    # w = [2.01, 0.72, 1.89, 0.47, 1.85, 0.74, 0.70, 1.58, 0.93, 0.76, 0.21, -0.13, 0.77, 1.79]
    # get_acc(show=True)
    # search(w,100000)
