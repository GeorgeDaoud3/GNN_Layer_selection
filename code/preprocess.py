import pandas as pd
import numpy as np
import os
import warnings
import pickle
from tqdm import tqdm
import argparse
    
"""Parse input arguments"""
parser = argparse.ArgumentParser(description='Train EBM model')

parser.add_argument('--senario_num', default='03', type=str, help='scene index')
parser.add_argument('--data_path', default='../Data/rounD/data/', type=str, help='path to the rounD data')
parser.add_argument('--h_sec', default=1, type=int, help='History in seconds')
parser.add_argument('--f_sec', default=5, type=int, help='Horizons in seconds')
parser.add_argument('--distance', default=30, type=float, help='maximum distance for spatial edges')

# samples the frames to 5 sample/second
def resample(tracksMeta,tracks):
    tracks=tracks[tracks['frame']%5==0]
    tracks['frame']=tracks['frame']//5
    tracks.sort_values('frame',inplace=True)
    tracks['ind']=np.arange(len(tracks))
    tracks.set_index('ind',inplace=True)
    tracksMeta['initialFrame']=(np.ceil(tracksMeta['initialFrame']/5)).astype(int)
    tracksMeta['finalFrame']=(np.floor(tracksMeta['finalFrame']/5)).astype(int)
    tracksMeta['numFrames']=tracksMeta['finalFrame']-tracksMeta['initialFrame']+1
    return tracksMeta,tracks

#initialize framesInfo , setting framesInfo[f,0]= start node index for frame f
def init_framesInfo(tracks):
    framesInfo=np.zeros((tracks.frame.max()+1,3),dtype=int)
    ind=-1;
    for i,row in tracks.iterrows():
        if ind!=row.frame:
            ind=row.frame
            framesInfo[int(row.frame),0]=i;
    #fill gabs in framesInfo
    wrongInd=np.where(framesInfo[:,0]==0)
    for wInd in wrongInd[0]:
        if(wInd!=0):
            framesInfo[wInd,0]=framesInfo[wInd-1,0]     
    return framesInfo

def get_nodes_features(tracks,tracksMeta):
    tracks=tracks.drop(["recordingId","lonVelocity","latVelocity","lonAcceleration","latAcceleration"], axis=1,inplace=False)
    tracksMeta_updated=tracksMeta.drop(["recordingId","width","length","numFrames"], axis=1,inplace=False)
    tracks_full=tracks.join(tracksMeta_updated,on="trackId",rsuffix="_").drop("trackId_",axis=1)
    classes=tracksMeta_updated["class"].unique()
    classes.sort()
    for c in classes:
        tracks_full[c]=(tracks_full["class"]==c)*1
    tracks_full.drop(["class",classes[0]], axis=1,inplace=True)
    tracks_full.drop(["trackLifetime","initialFrame","finalFrame"], axis=1,inplace=True)
    tracks_full['wcos']=tracks_full['width']* np.cos(tracks_full['heading']/180*np.pi)
    tracks_full['wsin']=tracks_full['width']* np.sin(tracks_full['heading']/180*np.pi)
    tracks_full['lcos']=tracks_full['length']* np.cos(tracks_full['heading']/180*np.pi)
    tracks_full['lsin']=tracks_full['length']* np.sin(tracks_full['heading']/180*np.pi)
    tracks_full['dcos']=np.cos(tracks_full['heading']/180*np.pi)
    tracks_full['dsin']=np.sin(tracks_full['heading']/180*np.pi)
    tracks_full.drop(["width","length","heading"], axis=1,inplace=True)
    tracks_full.drop(["xVelocity","yVelocity","xAcceleration","yAcceleration"], axis=1,inplace=True)
    nodes=tracks_full.drop(["trackId","frame"],axis=1).to_numpy()
    return nodes,tracks,tracks_full

def calculate_edge_lists_lengths(tracks_full,framesInfo,distance):
    # framesInfo[frame,1]=starting index of spatial edges of frame;
    # framesInfo[frame,2]=starting index of temporal edges of frame;
    maxLen=distance**2;
    tracks_min=tracks_full[["xCenter","yCenter","trackId","frame"]]
    spatial_count=0;
    temporal_count=0;
    frameInd=-1
    for frame in tqdm(range(tracks_min['frame'].min(),tracks_min['frame'].max()+1)):
        if frameInd!=frame:
            frameInd=frame
            framesInfo[frame,1]=spatial_count;
            framesInfo[frame,2]=temporal_count;
        frame0=tracks_min[tracks_min['frame']==frame]
        frame1=frame0[["xCenter","yCenter"]]
        frame2=frame0[["trackId"]]
        inds=frame0.index;
        for i in range(len(inds)):
            for j in range(i+1,len(inds)):
                if (((frame1.iloc[i].to_numpy()-frame1.iloc[j].to_numpy())**2).sum())<maxLen :
                    spatial_count=spatial_count+2;
        if frame!=0:
            for i in range(len(inds)):
                if (pre[pre['trackId']==frame2.iloc[i].to_numpy()[0]].shape[0]>0):
                    temporal_count=temporal_count+1;
        pre=frame0;
    return tracks_min,framesInfo,spatial_count,temporal_count
def create_egde_lists(tracks_min,spatial_count,temporal_count,distance):
    #create egdes array, edge1 for spatial, edge 2 for temporal
    spatial_edges=np.zeros((2, spatial_count))
    temporal_edges=np.zeros((2, temporal_count))
    maxLen=distance**2;
    count=0;
    count2=0;
    for frame in tqdm(range(tracks_min['frame'].min(),tracks_min['frame'].max()+1)):
        frame0=tracks_min[tracks_min['frame']==frame]
        frame1=frame0[["xCenter","yCenter"]]
        frame2=frame0[["trackId"]]
        inds=frame0.index;
        for i in range(len(inds)):
            for j in range(i+1,len(inds)):
                if (((frame1.iloc[i].to_numpy()-frame1.iloc[j].to_numpy())**2).sum())<maxLen :
                    spatial_edges[:,count]=[frame1.index[i],frame1.index[j]]
                    spatial_edges[:,count+1]=[frame1.index[j],frame1.index[i]];
                    count=count+2;
        if frame!=0:
            for i in range(len(inds)):
                query=pre[pre['trackId']==frame2.iloc[i].to_numpy()[0]];
                if (query.shape[0]>0):
                    temporal_edges[:,count2]=[query.index[0],frame1.index[i]]
                    count2=count2+1;
        pre=frame0;
    #with open(path+senario_num+'/new/edges_spatial.pkl', 'wb') as f2:
        #pickle.dump(spatial_edges, f2)
    #with open(path+senario_num+'/new/edges_temporal.pkl', 'wb') as f2:
        #pickle.dump(temporal_edges, f2)
    return spatial_edges,temporal_edges
def generateOutput(tracksMeta,tracks,history,horizon):
    # output trajectory in world coordinate
    InterestedTracksMeta=tracksMeta[tracksMeta['numFrames']>=(history+horizon)]
    InterestedTrackscount=InterestedTracksMeta.shape[0]
    InterestedFramescount=InterestedTracksMeta['numFrames'].sum()-(history+horizon-1)*InterestedTrackscount
    c=0;
    output=np.zeros((tracks.shape[0],2*horizon))
    output_mask=np.zeros((tracks.shape[0],1))
    for row in tqdm(InterestedTracksMeta.iterrows()):
        trackId=row[1]['trackId']
        xs=np.zeros((horizon))
        ys=np.zeros((horizon))
        c2=0;
        for frame in range(row[1]['initialFrame']+history-1,row[1]['initialFrame']+history+horizon-1):
            temp=tracks[tracks['frame']==frame]
            vehicle=temp[tracks['trackId']==trackId]
            vehicle=vehicle.iloc[0]
            xs[c2]= vehicle['xCenter']
            ys[c2]= vehicle['yCenter']
            c2=c2+1;
        for frame in range(row[1]['initialFrame']+history-1,row[1]['finalFrame']-horizon+1):
            xc=xs[0]
            yc=ys[0]
            temp=tracks[tracks['frame']==(frame+horizon)]
            vehicle=temp[tracks['trackId']==trackId]
            vehicle=vehicle.iloc[0]
            xs[:-1]=xs[1:]
            ys[:-1]=ys[1:]
            xs[-1]= vehicle['xCenter']
            ys[-1]= vehicle['yCenter']
            deltaX=xs-xc
            deltaY=(ys-yc)
            temp=tracks[tracks['frame']==(frame)]
            vehicle=temp[tracks['trackId']==trackId]
            index=vehicle.index[0]
            output[index,0:horizon]=deltaX
            output[index,horizon:]=deltaY
            output_mask[index,0]=1
            c=c+1
    return output, output_mask
def norm_denorm(nodes,output):
    max_abs_nodes= np.max(np.abs(nodes),axis=0)
    min_output= np.min(output,axis=0)
    max_output= np.max(np.abs(output),axis=0)
    return max_abs_nodes,min_output,max_output;
    
def save(framesInfo,nodes,spatial_edges,temporal_edges,output, output_mask,max_abs_nodes,min_output,max_output,path):
    with open(path+'/framesInfo.pkl', 'wb') as f2:
        pickle.dump(framesInfo, f2)
    with open(path+'/nodes.pkl', 'wb') as f2:
        pickle.dump(nodes, f2)
    with open(path+'/spatial_edges.pkl', 'wb') as f2:
        pickle.dump(spatial_edges, f2)
    with open(path+'/temporal_edges.pkl', 'wb') as f2:
        pickle.dump(temporal_edges, f2)
    with open(path+'/output.pkl', 'wb') as f2:
        pickle.dump(output, f2)
    with open(path+'/output_mask.pkl', 'wb') as f2:
        pickle.dump(output_mask, f2)
    with open(path+'/max_abs_nodes.pkl', 'wb') as f2:
        pickle.dump(max_abs_nodes, f2)
    with open(path+'/min_output.pkl', 'wb') as f2:
        pickle.dump(min_output, f2)
    with open(path+'/max_output.pkl', 'wb') as f2:
        pickle.dump(max_output, f2)
        
def main(flag):
    path=flag.data_path+flag.senario_num
    if not os.path.exists(path):
        os.mkdir(path)
    recordingMeta=pd.read_csv(path+'_recordingMeta.csv')
    tracksMeta=pd.read_csv(path+'_tracksMeta.csv')
    tracks=pd.read_csv(path+'_tracks.csv')
    frame_rate=5;
    history=flag.h_sec*frame_rate
    horizon=flag.f_sec*frame_rate;
    
    #fill the first column of framesInfo  s.t. framesInfo[frame_ind]= start_index in original_tracks
    # the ending index would be framesInfo[frame_ind+1]-1 or the end of original_tracks
    
    print("resampling to 5 samples/second");
    tracksMeta,tracks=resample(tracksMeta,tracks)
    print("Initializing framesInfo array");
    framesInfo=init_framesInfo(tracks)
    print("Creating nodes features matrix");
    nodes,tracks,tracks_full=get_nodes_features(tracks,tracksMeta)
    print("Calculating the length of edge lists and fill framesInfo");
    tracks_min,framesInfo,spatial_count,temporal_count=calculate_edge_lists_lengths(tracks_full,framesInfo,flag.distance)
    print("Generaing the edge lists");
    spatial_edges,temporal_edges=create_egde_lists(tracks_min,spatial_count,temporal_count,flag.distance)
    print("calcualting the output matrix");
    output, output_mask=generateOutput(tracksMeta,tracks,history,horizon)
    print("calculating normalization/denormalization vectors")
    max_abs_nodes,min_output,max_output=norm_denorm(nodes,output)
    print("Saving the graph")
    save(framesInfo,nodes,spatial_edges,temporal_edges,output, output_mask,max_abs_nodes,min_output,max_output,flag.data_path+flag.senario_num)
    
if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=Warning)
    flag=parser.parse_args()
    print(flag)
    main(flag)