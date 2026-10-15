import pandas as pd
import numpy as np
import os
import warnings
import pickle
from tqdm import tqdm
import torch; 
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv,GCNConv,Linear,GATConv,SuperGATConv,SAGEConv,TransformerConv,TAGConv,ARMAConv,SGConv
from torch_geometric.nn import GraphConv,ChebConv,AGNNConv,GatedGraphConv,ResGatedGraphConv,SSGConv,MFConv,MixHopConv,LEConv
from torch_geometric.nn import FAConv,EGConv,DNAConv
from torch import nn as torchNN
from torchvision.models.resnet import resnet18
from random import randint
from torch.nn import MSELoss
from torch.optim import Adam
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import cv2 as cv
import argparse
    
"""Parse input arguments"""
parser = argparse.ArgumentParser(description='Train EBM model')

parser.add_argument('--deviceStr', default='cuda:0', type=str, help='Specify the pytorch device')
parser.add_argument('--model_name', default='', type=str, help='a partially trained model for continue learning')
parser.add_argument('--model_path', default='', type=str, help='path for saving models')
parser.add_argument('--spatial_type', default='gcn', type=str , help='whether a GMM model or not (priority: 1), the highest')
parser.add_argument('--temporal_type', default='gcn', type=str , help='whether a GMM model or not (priority: 1), the highest')


parser.add_argument('--hidden_channels', default=100, type=int,help='hidden_channels for TAG layers') 
parser.add_argument('--hidden_channels2', default=200, type=int,help='hidden_channels for output head') 
parser.add_argument('--num_layers', default=4, type=int,help='number of TAGCN layers') 
parser.add_argument('--components', default=10, type=int,help='GMM components') 

# training
parser.add_argument('--senario_num', default='03', type=str, help='scene index')
parser.add_argument('--data_path', default='../Data/rounD/data/', type=str, help='path to the rounD data')
parser.add_argument('--h_sec', default=1, type=int, help='History in seconds')
parser.add_argument('--f_sec', default=5, type=int, help='Horizons in seconds')
parser.add_argument('--img_width', default=128, type=int, help='The map segment wifth')
parser.add_argument('--resolution', default=1, type=int,help='combined pixels in the map segment')    

# training
parser.add_argument('--num_steps', default=4, type=int,help='number of steps to change the learning rate ') 
parser.add_argument('--init_epoch', default=40, type=int,help='The initial number of epoches')  
parser.add_argument('--reduce_epoch', default=10, type=int,help='The reduction in the number of epoches per step')  
parser.add_argument('--init_divisor', default=1, type=float ,help='initial divisior to the learning rate ')    
parser.add_argument('--update_divisor', default=10, type=float ,help='multiplicant to the learning rate divisor ')    

def load(path):
    with open(path+'/framesInfo.pkl', 'rb') as f2:
        framesInfo=pickle.load(f2)
    with open(path+'/nodes.pkl', 'rb') as f2:
        nodes=pickle.load(f2)
    with open(path+'/spatial_edges.pkl', 'rb') as f2:
        spatial_edges=pickle.load(f2)
    with open(path+'/temporal_edges.pkl', 'rb') as f2:
        temporal_edges=pickle.load(f2)
    with open(path+'/output.pkl', 'rb') as f2:
        output=pickle.load(f2)
    with open(path+'/output_mask.pkl', 'rb') as f2:
        output_mask=pickle.load(f2)
    with open(path+'/max_abs_nodes.pkl', 'rb') as f2:
        max_abs_nodes=pickle.load(f2)
    with open(path+'/min_output.pkl', 'rb') as f2:
        min_output=pickle.load(f2)
    with open(path+'/max_output.pkl', 'rb') as f2:
        max_output=pickle.load(f2)
    recordingMeta=pd.read_csv(path+'_recordingMeta.csv')
    scale=recordingMeta['orthoPxToMeter'][0]*10
    return framesInfo,nodes,spatial_edges,temporal_edges,output, output_mask,max_abs_nodes,min_output,max_output,scale

def slice(nodes,max_abs_nodes,path,nodes_beg,nodes_end,flag,scale):
    image = cv.imread(flag.data_path+flag.senario_num+'_background.png')
    img_half_act_width=(flag.img_width*flag.resolution)//2
    sub_images=np.zeros((nodes_end-nodes_beg,3,flag.img_width,flag.img_width))
    sub_nodes= nodes[nodes_beg:nodes_end,:]
    for i in range(nodes_end-nodes_beg):
        T1 = np.float32([[1, 0, -sub_nodes[i,0]/scale], [0, 1, sub_nodes[i,1]/scale], [0,0,1]])
        T3 = np.float32([[1, 0, img_half_act_width], [0, 1, img_half_act_width], [0,0,1]])
        T=T3@T1
        img_translation = cv.warpAffine(image, T[0:2,:], (image.shape[1],image.shape[0]))
        img_translation=img_translation[0:img_half_act_width*2,0:img_half_act_width*2,:]
        img_translation=cv.resize(img_translation,(flag.img_width,flag.img_width),interpolation = cv.INTER_AREA)
        image2=np.swapaxes(img_translation, 0, 2)
        sub_images[i,:,:,:]=image2
    sub_images=(sub_images-128)/256
    return sub_nodes/max_abs_nodes,sub_images

def saveModel(model, model_path):
    torch.save(model.state_dict(), model_path)

def loadModel(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model;
    
def create_training_sets(framesInfo,flag,history):
    n=(framesInfo.shape[0]-(history+1)*2)
    np.random.seed(123)
    samplesPer=np.random.permutation(n)
    n1=n*3//10;
    n2=n*1//10;
    training=samplesPer[:-n1]
    validating=samplesPer[-n1:-n2]
    return training,validating
    
    
def create_testing_set(framesInfo,flag,history):
    n=(framesInfo.shape[0]-(history+1)*2)
    np.random.seed(123)
    samplesPer=np.random.permutation(n)
    n2=n*1//10;
    testing=samplesPer[-n2:]
    return testing
 
class DTAG_NL(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_channels2, out_channels, num_layers,flag):
        super().__init__()
        if flag.spatial_type=='supergat2' or flag.temporal_type=='supergat2':
            hidden_channels=(hidden_channels//4)*4
        if flag.spatial_type=='mix' or flag.temporal_type=='mix':
            hidden_channels=(hidden_channels//3)*3
        self.resModel=resnet18()
        self.num_layers=num_layers
        self.out_channels=out_channels
        self.resModel.fc=torchNN.Linear(in_features=512, out_features=2*hidden_channels, bias=True)
        self.convs1 = torch.nn.ModuleList()
        self.convs2 = torch.nn.ModuleList()
        self.lins1 = torch.nn.ModuleList()
        self.lins2 = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            if flag.spatial_type=='gcn':
                print('gcn')
                conv1 = HeteroConv({
                    ('agent', 'space', 'agent'): GCNConv(hidden_channels*2+14,hidden_channels)
                }, aggr='mean')
            elif flag.spatial_type=='gat':
                print('gat')
                conv1 = HeteroConv({
                    ('agent', 'space', 'agent'): GATConv(hidden_channels*2+14,hidden_channels)
                }, aggr='mean')
            elif flag.spatial_type=='supergat':
                print('supergat')
                conv1 = HeteroConv({
                    ('agent', 'space', 'agent'): SuperGATConv(hidden_channels*2+14,hidden_channels)
                }, aggr='mean')
            elif flag.spatial_type=='supergat2':
                print('supergat2')
                conv1 = HeteroConv({
                    ('agent', 'space', 'agent'): SuperGATConv(hidden_channels*2+14,hidden_channels//4,heads=4)
                }, aggr='mean')
            elif flag.spatial_type=='sage':
                print('sage')
                conv1 = HeteroConv({
                    ('agent', 'space', 'agent'): SAGEConv(hidden_channels*2+14,hidden_channels)
                }, aggr='mean')
            elif flag.spatial_type=='transformer':
                print('transformer')
                conv1 = HeteroConv({
                    ('agent', 'space', 'agent'): TransformerConv(hidden_channels*2+14,hidden_channels//4,4)
                }, aggr='mean')
            elif flag.spatial_type=='tag':
                print('tag')
                conv1 = HeteroConv({
                    ('agent', 'space', 'agent'): TAGConv(hidden_channels*2+14,hidden_channels)
                }, aggr='mean')
            elif flag.spatial_type=='arma':
                print('arma')
                conv1 = HeteroConv({
                    ('agent', 'space', 'agent'): ARMAConv(hidden_channels*2+14,hidden_channels)
                }, aggr='mean')
            elif flag.spatial_type=='sg':
                print('sg')
                conv1 = HeteroConv({
                    ('agent', 'space', 'agent'): SGConv(hidden_channels*2+14,hidden_channels,K=3)
                }, aggr='mean')
            elif flag.spatial_type=='graph':
                print('graph')
                conv1 = HeteroConv({
                    ('agent', 'space', 'agent'): GraphConv(hidden_channels*2+14,hidden_channels)
                }, aggr='mean')
            elif flag.spatial_type=='cheb':
                print('cheb')
                conv1 = HeteroConv({
                    ('agent', 'space', 'agent'): ChebConv(hidden_channels*2+14,hidden_channels,K=3)
                }, aggr='mean')
            elif flag.spatial_type=='agnn':
                print('agnn')
                conv1 = HeteroConv({
                    ('agent', 'space', 'agent'): AGNNConv()
                }, aggr='mean')
                self.lins1.append(Linear(hidden_channels*2+14, hidden_channels))
            elif flag.spatial_type=='gru':
                print('gru')
                self.lins1.append(Linear(hidden_channels*2+14, hidden_channels))
                conv1 = HeteroConv({
                    ('agent', 'space', 'agent'): GatedGraphConv(hidden_channels, 1,'mean')
                }, aggr='mean')
            elif flag.spatial_type=='gru2':
                print('gru2')
                self.lins1.append(Linear(hidden_channels*2+14, hidden_channels))
                conv1 = HeteroConv({
                    ('agent', 'space', 'agent'): GatedGraphConv(hidden_channels*2+14, 1,'mean')
                }, aggr='mean')
            elif flag.spatial_type=='resgru':
                print('resgru')
                conv1 = HeteroConv({
                    ('agent', 'space', 'agent'): ResGatedGraphConv(hidden_channels*2+14,hidden_channels)
                }, aggr='mean')
            elif flag.spatial_type=='ssg':
                print('ssg')
                conv1 = HeteroConv({
                    ('agent', 'space', 'agent'): SSGConv(hidden_channels*2+14,hidden_channels,alpha=0.5,K=3)
                }, aggr='mean')
            elif flag.spatial_type=='mf':
                print('mf')
                conv1 = HeteroConv({
                    ('agent', 'space', 'agent'): MFConv(hidden_channels*2+14,hidden_channels)
                }, aggr='mean')
            elif flag.spatial_type=='mix':
                print('mix')
                conv1 = HeteroConv({
                    ('agent', 'space', 'agent'): MixHopConv(hidden_channels*2+14,hidden_channels//3)
                }, aggr='mean')
            elif flag.spatial_type=='le':
                print('le')
                conv1 = HeteroConv({
                    ('agent', 'space', 'agent'): LEConv(hidden_channels*2+14,hidden_channels)
                }, aggr='mean')
            elif flag.spatial_type=='fa':
                print('fa')
                self.lins1.append(Linear(hidden_channels*2+14, hidden_channels))
                conv1 = HeteroConv({
                    ('agent', 'space', 'agent'): FAConv(hidden_channels*2+14)
                }, aggr='mean')
            elif flag.spatial_type=='eg':
                print('eg')
                conv1 = HeteroConv({
                    ('agent', 'space', 'agent'): EGConv(hidden_channels*2+14,hidden_channels,['mean'],4)
                }, aggr='mean')
            elif flag.spatial_type=='dna':
                print('fa')
                self.lins1.append(Linear(hidden_channels*2+14, hidden_channels))
                conv1 = HeteroConv({
                    ('agent', 'space', 'agent'): DNAConv(hidden_channels*2+14,4)
                }, aggr='mean')
                
            if flag.temporal_type=='gcn':
                print('gcn')
                conv2 = HeteroConv({
                    ('agent', 'time', 'agent'): GCNConv(hidden_channels*2+14,hidden_channels)
                }, aggr='mean')
            elif flag.temporal_type=='gat':
                print('gat')
                conv2 = HeteroConv({
                    ('agent', 'time', 'agent'): GATConv(hidden_channels*2+14,hidden_channels)
                }, aggr='mean')
            elif flag.temporal_type=='supergat':
                print('supergat')
                conv2 = HeteroConv({
                    ('agent', 'time', 'agent'): SuperGATConv(hidden_channels*2+14,hidden_channels)
                }, aggr='mean')
            elif flag.temporal_type=='supergat2':
                print('supergat2')
                conv2 = HeteroConv({
                    ('agent', 'time', 'agent'): SuperGATConv(hidden_channels*2+14,hidden_channels//4,heads=4)
                }, aggr='mean')
            elif flag.temporal_type=='sage':
                print('sage')
                conv2 = HeteroConv({
                    ('agent', 'time', 'agent'): SAGEConv(hidden_channels*2+14,hidden_channels)
                }, aggr='mean')
            elif flag.temporal_type=='transformer':
                print('transformer')
                conv2 = HeteroConv({
                    ('agent', 'time', 'agent'): TransformerConv(hidden_channels*2+14,hidden_channels//4,4)
                }, aggr='mean')
            elif flag.temporal_type=='tag':
                print('tag')
                conv2 = HeteroConv({
                    ('agent', 'time', 'agent'): TAGConv(hidden_channels*2+14,hidden_channels)
                }, aggr='mean')
            elif flag.temporal_type=='arma':
                print('arma')
                conv2 = HeteroConv({
                    ('agent', 'time', 'agent'): ARMAConv(hidden_channels*2+14,hidden_channels)
                }, aggr='mean')
            elif flag.temporal_type=='sg':
                print('sg')
                conv2 = HeteroConv({
                    ('agent', 'time', 'agent'): SGConv(hidden_channels*2+14,hidden_channels,K=3)
                }, aggr='mean')
            elif flag.temporal_type=='graph':
                print('graph')
                conv2 = HeteroConv({
                    ('agent', 'time', 'agent'): GraphConv(hidden_channels*2+14,hidden_channels)
                }, aggr='mean')
            elif flag.temporal_type=='cheb':
                print('cheb')
                conv2 = HeteroConv({
                    ('agent', 'time', 'agent'): ChebConv(hidden_channels*2+14,hidden_channels,K=3)
                }, aggr='mean')  
            elif flag.temporal_type=='agnn':
                print('agnn')
                conv2 = HeteroConv({
                    ('agent', 'time', 'agent'): AGNNConv()
                }, aggr='mean')
                self.lins2.append(Linear(hidden_channels*2+14, hidden_channels))  
            elif flag.temporal_type=='gru':
                print('gru')
                self.lins2.append(Linear(hidden_channels*2+14, hidden_channels))
                conv2 = HeteroConv({
                    ('agent', 'time', 'agent'): GatedGraphConv(hidden_channels, 1,'mean')
                }, aggr='mean')
            elif flag.temporal_type=='gru2':
                print('gru2')
                self.lins2.append(Linear(hidden_channels*2+14, hidden_channels))
                conv2 = HeteroConv({
                    ('agent', 'time', 'agent'): GatedGraphConv(hidden_channels*2+14, 1,'mean')
                }, aggr='mean')
            elif flag.temporal_type=='resgru':
                print('resgru')
                conv2 = HeteroConv({
                    ('agent', 'time', 'agent'): ResGatedGraphConv(hidden_channels*2+14,hidden_channels)
                }, aggr='mean')
            elif flag.temporal_type=='ssg':
                print('ssg')
                conv2 = HeteroConv({
                    ('agent', 'time', 'agent'): SSGConv(hidden_channels*2+14,hidden_channels,alpha=0.5,K=3)
                }, aggr='mean')
            elif flag.temporal_type=='mf':
                print('mf')
                conv2 = HeteroConv({
                    ('agent', 'time', 'agent'): MFConv(hidden_channels*2+14,hidden_channels)
                }, aggr='mean')
            elif flag.temporal_type=='mix':
                print('mix')
                conv2 = HeteroConv({
                    ('agent', 'time', 'agent'): MixHopConv(hidden_channels*2+14,hidden_channels//3)
                }, aggr='mean')
            elif flag.temporal_type=='le':
                print('le')
                conv2 = HeteroConv({
                    ('agent', 'time', 'agent'): LEConv(hidden_channels*2+14,hidden_channels)
                }, aggr='mean')
            elif flag.temporal_type=='fa':
                print('fa')
                self.lins2.append(Linear(hidden_channels*2+14, hidden_channels))
                conv2 = HeteroConv({
                    ('agent', 'time', 'agent'): FAConv(hidden_channels*2+14)
                }, aggr='mean')
            elif flag.temporal_type=='eg':
                print('eg')
                conv2 = HeteroConv({
                    ('agent', 'time', 'agent'): EGConv(hidden_channels*2+14,hidden_channels,['mean'],4)
                }, aggr='mean')
            elif flag.temporal_type=='dna':
                print('fa')
                self.lins2.append(Linear(hidden_channels*2+14, hidden_channels))
                conv2 = HeteroConv({
                    ('agent', 'time', 'agent'): DNAConv(hidden_channels*2+14,4)
                }, aggr='mean')
            self.convs1.append(conv1)
            self.convs2.append(conv2)

        self.lin = Linear(hidden_channels*2+14, hidden_channels2)
        self.lin2 = Linear(hidden_channels2, out_channels)

    def forward(self, x_dict,x2_dict, edge_index_dict):
        temp=self.resModel(x2_dict['agent'])
        x_dict['agent']=torch.cat((x_dict['agent'][:,:14],temp),dim=1)
        if flag.spatial_type=='fa' or flag.temporal_type=='fa':
            x0_dict={'agent':x_dict['agent']}
        for i in range(self.num_layers):
            conv2=self.convs2[i];
            conv1=self.convs1[i];
            if flag.spatial_type=='gru' or flag.spatial_type=='agnn':
                temp1={'agent':self.lins1[i](x_dict['agent'])}
                temp1 = conv1(temp1, edge_index_dict)
            elif flag.spatial_type=='gru2' or flag.spatial_type=='dna':
                temp1 = conv1(x_dict, edge_index_dict)
                temp1['agent']=self.lins1[i](temp1['agent'])
            elif flag.spatial_type=='fa':
                temp1 = conv1(x_dict,x0_dict, edge_index_dict)
                temp1['agent']=self.lins1[i](temp1['agent'])
            else:
                temp1 = conv1(x_dict, edge_index_dict)
                
            if flag.temporal_type=='gru' or flag.temporal_type=='agnn':
                temp2={'agent':self.lins2[i](x_dict['agent'])}
                temp2 = conv2(temp2, edge_index_dict)    
            elif flag.temporal_type=='gru2' or flag.temporal_type=='dna':
                temp2 = conv2(x_dict, edge_index_dict)
                temp2['agent']=self.lins2[i](temp2['agent'])
            elif flag.temporal_type=='fa':
                temp2 = conv2(x_dict, x0_dict, edge_index_dict)
                temp2['agent']=self.lins2[i](temp2['agent'])
            else:
                temp2 = conv2(x_dict, edge_index_dict)
                
            x_dict['agent']=torch.cat((x_dict['agent'][:,:14],temp1['agent'].relu(),temp2['agent'].relu()),dim=1)
            #x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin2(self.lin(x_dict['agent']).tanh()).sigmoid()


def test_one_epoch(model,nodes,max_abs_nodes,min_output,max_output,path,epoch_index,testing,output,output_mask,flag,scale,spatial_edges,temporal_edges,framesInfo):
    criterion = MSELoss(reduction='sum').to(flag.deviceStr)
    #global running_loss
    running_loss = 0.
    stdv_max_P =0.;
    minDE =0.;
    DE_max_P =0.;
    expected_DE =0.;
    alphas_max_P=torch.zeros((flag.components,))
    last_loss = 0.
    horizons=model.out_channels
    horizon=horizons//2
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    #print('teet')
    pbar = tqdm(testing)
    total_count=0;
    output_dims=output.shape[1]
    range_torch=torch.tensor(max_output-min_output)
    with torch.no_grad():
        model.train(False)
        for i, ind in enumerate(pbar):
            frame=ind+6;
            graph = HeteroData()
            sub_nodes,sub_images=slice(nodes,max_abs_nodes,path,framesInfo[frame-6,0],framesInfo[frame+1,0],flag,scale)
            graph['agent'].x=torch.Tensor(sub_nodes)
            graph['agent'].x2=torch.Tensor(sub_images)
            graph['agent','space','agent'].edge_index =torch.Tensor(spatial_edges[:,framesInfo[frame-6,1]:framesInfo[frame+1,1]]-framesInfo[frame-6,0]).type(dtype=torch.LongTensor)
            graph['agent','time','agent'].edge_index =torch.Tensor(temporal_edges[:,framesInfo[frame-6+1,2]:framesInfo[frame+1,2]]-framesInfo[frame-6,0]).type(dtype=torch.LongTensor)
            labels=(output[framesInfo[frame-6,0]:framesInfo[frame+1,0]]-min_output)/(max_output-min_output)
            mask=output_mask[framesInfo[frame-6,0]:framesInfo[frame+1,0],0].copy()
            mask[:framesInfo[frame,0]-framesInfo[frame-6,0]]=False
            
            nodes_per_frame=np.sum(mask==True)
            if(nodes_per_frame==0):
                continue;
                
            total_count=total_count+nodes_per_frame;
            graph.pin_memory()
            graph = graph.to(flag.deviceStr, non_blocking=True)
            
            
            outputs = model(graph.x_dict,graph.x2_dict, graph.edge_index_dict)
            loss = criterion(outputs[mask==True,:], torch.tensor(labels[mask==True,:]).float().to(flag.deviceStr))
            running_loss += loss.detach().sum()
            DE=(outputs[mask==True,:].cpu()-torch.tensor(labels[mask==True,:])).abs()*range_torch
            DE2=(DE[:,:horizon]**2+DE[:,horizon:]**2)**0.5;
            minDE+=DE2.sum(axis=0);
            current_loss = running_loss / (total_count*output_dims) # loss per batch
            pbar.set_description(f"training MSE: {current_loss} , Error: {100*(current_loss**0.5)} %")
            
    return running_loss / max(1,total_count*output_dims),minDE.cpu()/ max(1,total_count)


def create_model(flag,horizon):
   
    model = DTAG_NL(hidden_channels=flag.hidden_channels, hidden_channels2=flag.hidden_channels2, out_channels=2*horizon,num_layers=flag.num_layers,flag=flag).to(flag.deviceStr, non_blocking=True)
    return model;

def stat(DE,flag):
    print("FDE:"+str(DE[flag.frame_rate-1::flag.frame_rate]))
    ADE=np.zeros((flag.frame_rate,))
    for i in range(1,1+flag.frame_rate):
        ADE[i-1]=DE[:i*flag.frame_rate].mean()
    print("ADE: "+str(ADE))
    print()
    
def evaluate(model,nodes,max_abs_nodes,min_output,max_output,path,testing,output,output_mask,flag,scale,spatial_edges,temporal_edges,framesInfo):
    epoch_index=-1;
    model_path=flag.model_name;
    print(path+"/"+model_path)
    model=loadModel(model,path+"/"+model_path)
    
    loss,minDE =test_one_epoch(model,nodes,max_abs_nodes,min_output,max_output,path,epoch_index,testing,output,output_mask,flag,scale,spatial_edges,temporal_edges,framesInfo)
    print("running loss: "+str(loss.item()))
    print("DE:\n"+str(minDE))
    stat(minDE,flag)
   
        
def main(flag):
    print("available cuda devices:")
    print([(i, torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())])
    path=flag.data_path+flag.senario_num
    model_path=flag.model_path;
    if model_path=="":
        model_path=flag.data_path+'/models/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    flag.frame_rate=5;
    history=flag.h_sec*flag.frame_rate
    horizon=flag.f_sec*flag.frame_rate;
    print()
    print("Loading the graph")
    framesInfo,nodes,spatial_edges,temporal_edges,output, output_mask,max_abs_nodes,min_output,max_output,scale=load(path)
    print("Creating the model")
    model=create_model(flag,horizon)
    print("creating the testing")
    testing=create_testing_set(framesInfo,flag,history)
    #training,validating=create_training_sets(framesInfo,flag,history)
    evaluate(model,nodes,max_abs_nodes,min_output,max_output,model_path,testing,output,output_mask,flag,scale,spatial_edges,temporal_edges,framesInfo)
            
if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=Warning)
    flag=parser.parse_args()
    print(flag)
    main(flag)
