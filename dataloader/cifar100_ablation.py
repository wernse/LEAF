import os,sys
import numpy as np
import torch
# import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle

cf100_dir = './data/'
file_dir = './data/binary_cifar100_ablation'

def get(seed=0,pc_valid=0.10):
    data={}
    taskcla=[]
    size=[3,32,32]
    data1={}
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]

        # CIFAR100
        dat={}
        dat['train']=datasets.CIFAR100(cf100_dir,train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR100(cf100_dir,train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        # dat['train'] = datasets.CIFAR100(cf100_dir,train=True,download=False,transform=transforms.Compose([transforms.ToTensor()]))
        # dat['test']  = datasets.CIFAR100(cf100_dir,train=False,download=False,transform=transforms.Compose([transforms.ToTensor()]))
        for n in range(10):
            data[n]={}
            data[n]['name']='cifar100'
            data[n]['ncla']=10
            data[n]['train']={'x': [],'y': []}
            data[n]['test']={'x': [],'y': []}

        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
            for image,target in loader:
                n=target.numpy()[0]
                nn=(n//10)
                data[nn][s]['x'].append(image) # 255 
                data[nn][s]['y'].append(n%10)
        # none uniform setting
        # overlap
        # 0-9  5-14 10-19 20-29, 25-34, 30-39, 40-49
        # for t in range(10):
        #     for s in ['train','test']:
        for i in range(7):
            data1[i] = {}
            data1[i]['ncla']=10
            data1[i]['train']={'x': [],'y': []}
            data1[i]['test']={'x': [],'y': []}
            if i == 0:
                data1[i] = data[i]
            elif i == 1:
                for s in ['train','test']:
                    for idx in range(len(data[0][s]['y'])):
                        if data[i][s]['y'][idx] <  6:
                            data1[i][s]['x'].append(data[0][s]['x'][idx])
                            data1[i][s]['y'].append(data[0][s]['y'][idx])
                    for idx in range(len(data[1][s]['y'])):
                        if data[i][s]['y'][idx] >  5:
                            data1[i][s]['x'].append(data[1][s]['x'][idx])
                            data1[i][s]['y'].append(data[1][s]['y'][idx])
            elif i == 2:
                data1[i] = data[1]
            elif i == 3:
                data1[i] = data[2]               
            elif i == 4:
                for s in ['train','test']:
                    for idx in range(len(data[2][s]['y'])):
                        if data[i][s]['y'][idx] <  6:
                            data1[i][s]['x'].append(data[2][s]['x'][idx])
                            data1[i][s]['y'].append(data[2][s]['y'][idx])
                    for idx in range(len(data[3][s]['y'])):
                        if data[i][s]['y'][idx] >  5:
                            data1[i][s]['x'].append(data[3][s]['x'][idx])
                            data1[i][s]['y'].append(data[3][s]['y'][idx])
            elif i == 5:
                data1[i] = data[3]
            elif i == 6:
                data1[i] = data[4]                
        # "Unify" and save
        for t in data1.keys():
            for s in ['train','test']:
                data1[t][s]['x']=torch.stack(data[t][s]['x']).view(-1,size[0],size[1],size[2])
                data1[t][s]['y']=torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)
                torch.save(data1[t][s]['x'], os.path.join(os.path.expanduser(file_dir),'data'+str(t)+s+'x.bin'))
                torch.save(data1[t][s]['y'], os.path.join(os.path.expanduser(file_dir),'data'+str(t)+s+'y.bin'))

    # Load binary files
    data={}
    # ids=list(shuffle(np.arange(5),random_state=seed))
    ids=list(np.arange(7))
    print('Task order =',ids)
    for i in range(7):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser(file_dir),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser(file_dir),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))
        if data[i]['ncla']==2:
            data[i]['name']='cifar10-'+str(ids[i])
        else:
            data[i]['name']='cifar100-'+str(ids[i])

    # Validation
    for t in data.keys():
        r=np.arange(data[t]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
        data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size
