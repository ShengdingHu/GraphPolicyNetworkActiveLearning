from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
import argparse
import torch
from src.utils.dataloader import GraphLoader
import pickle as pkl
def selectclass(labels):
    thres=4
    numof1or0 = labels.sum(dim=0)/labels.size(0)
    idxinclude = torch.where((numof1or0>0.75+(numof1or0<0.25))==0)[0]
    # print(len(idxinclude))
    labels = labels[:,idxinclude]
    equalsset = set()
    useset =set()
    for i in range(len(idxinclude)):
        if i in equalsset:
            continue
        base = labels[:,i:i+1]
        rate = (labels*base).sum(dim=0)/labels.sum(dim=0) # labels 1 is inside base's 1
        rate2 = ((1-labels)*(1-base)).sum(dim=0)/(1-labels).sum(dim=0) # labels 0 is inside base's 0, 
                                                               #if many large rate, base doesn't have many 1 (at hierachy's bottom)
        largeloc = torch.where(rate>0.9) [0].numpy().tolist()
        largeloc2 = torch.where(rate2>0.9) [0].numpy().tolist() # if both high, two classes are equal
        

        
        
        
        # print("i:{} {}; {}".format(i,largeloc,largeloc2))
        if len(largeloc)<thres+1 and len(largeloc2)<thres:
            equals = set(largeloc).intersection(set(largeloc2))
            equalsset.update(equals)
            print("good class {} ;{}; {} ||equals: {}".format(i,largeloc,largeloc2,equals))
            # print("equal set {}".format(equalsset))
            useset.add(i)
    useset = list(useset)
    finalgoodclass = idxinclude[useset]
    # print(useset)
    # print(finalgoodclass,len(finalgoodclass))
    return finalgoodclass
    # print(labels.sum(dim=0))

def loadPPI():
    from torch_geometric.datasets import PPI
    ppi = PPI(root = "data/ppi/")
    ppitrain = ppi
    ppi += PPI(root = "data/ppi/",split="val")
    ppi += PPI(root = "data/ppi/",split="test")
    Y = []
    for ppii in ppi:
        Y.append(ppii.y)
    Y = torch.cat(Y,dim=0)
    labels = Y
    goodclass = selectclass(labels)
    # print(labels.size())
    labels = labels[:,goodclass]
    print(labels[:10,:])
    print("="*20)
    selectclass(labels) ## check
    # print(ppi.data.x[:10,:])[]
    # for i in range()

    
    for i in range(len(ppi)):
        print("ppi{}".format(i))
        ppii =ppi[i]
        N = ppii.x.size(0)
        edgelist = ppii.edge_index.transpose(0,1).numpy().tolist()
        with open("data/ppi/ppi"+str(i)+".edgelist",'w') as file:
            file.write(str(N)+"\n")
            for edge in edgelist:
                file.write(str(edge[0])+" "+str(edge[1])+"\n")
        labeli = ppii.y[:,goodclass].numpy()
        with open("data/ppi/ppi"+str(i)+".y.pkl",'wb') as file:
            pkl.dump(labeli,file)
        with open("data/ppi/ppi"+str(i)+".x.pkl",'wb') as file:
            xi = ppii.x.numpy()
            pkl.dump(xi,file)
    
    i=24
    print("ppitrain {}".format(i))
    ppii = ppitrain.data
    N = ppii.x.size(0)
    edgelist = ppii.edge_index.transpose(0,1).numpy().tolist()
    with open("data/ppi/ppi"+str(i)+".edgelist",'w') as file:
        file.write(str(N)+"\n")
        for edge in edgelist:
            file.write(str(edge[0])+" "+str(edge[1])+"\n")
    labeli = ppii.y[:,goodclass].numpy()
    with open("data/ppi/ppi"+str(i)+".y.pkl",'wb') as file:
        pkl.dump(labeli,file)
    with open("data/ppi/ppi"+str(i)+".x.pkl",'wb') as file:
        xi = ppii.x.numpy()
        pkl.dump(xi,file)
    
   



    
        





if __name__=="__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--d_name", type=str, default="ogbn-proteins")
    # args = parser.parse_args()
    # dataset = PygNodePropPredDataset(name = args.d_name) 
    # num_tasks = dataset.num_tasks
    # print(dataset.data,num_tasks)
    loadPPI()
    # data = torch.load("ogbn_proteins_pyg/processed/geometric_data_processed.pt")
    # print(data)
    # g=GraphLoader("corafull")
    # print(g.X.shape)
    # print(g.Y.shape)

    