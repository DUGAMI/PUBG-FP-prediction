import pandas as pd
import numpy as np

#change this path to your dataset root
base="E:\\PUBG-DATA\\"
data=pd.read_csv(base+"train_V2.csv")
print("data loadding finish")

#take out all matchID from dataset
matchIDList=set(data["matchId"].tolist())
matchsize=len(matchIDList)

GraphDataset=[]

for index,matchID in enumerate(matchIDList):

    print("正在转化{}[{}/{}]".format(matchID,index,matchsize))

    match = data.ix[data['matchId'] == matchID]

    groupIDList = []
    edge=[]

    #group single match data by kills
    groupByKills = match.groupby(match["kills"])

    for name, data in groupByKills:
        data = data.sort_values(by="killPlace")
        groupIdList = data["groupId"].values
        len=groupIdList.size

        for idx,groupId in enumerate(groupIdList):

            groupIDList.append(groupId)

            if idx==len-1:
                break

            edge.append([groupId,groupIdList[idx+1]])

    #the format of single sample of GraphDataset is [matchID,nodeID,edgeList]
    GraphDataset.append([matchID,set(groupIDList),edge])

np.save("dataset.npy",GraphDataset)