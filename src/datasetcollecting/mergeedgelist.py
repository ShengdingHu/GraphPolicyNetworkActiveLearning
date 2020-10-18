import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",type=str)
    parser.add_argument("--mode",type=str)
    parser.add_argument("--mergeversion",type=str)
    parser.add_argument("--desversion",type=str)
    
    args = parser.parse_args()
    return args

def merge(args):
    index = 0
    all_edgelist = []
    datasets = args.datasets.split("+")
    for dataset in datasets:
        with open("data/"+dataset+"/"+dataset+".edgelist",'r') as f:
            L = f.readlines()
            num_node = int(L[0].strip())
            L = L[1:]
            edge_list = [[int(x)+index for x in e.strip().split()] for e in L]
        all_edgelist.extend(edge_list)
        index += num_node
    
    if not os.path.exists("data/merge"+args.mergeversion):
        os.mkdir("data/merge"+args.mergeversion)
    with open("data/merge"+args.mergeversion+"/merge"+args.mergeversion+".edgelist",'w') as f:
        f.write(str(index)+"\n")
        for i in range(len(all_edgelist)):
            f.write(str(all_edgelist[i][0])+" "+str(all_edgelist[i][1])+"\n")
    print("all nodes {} all edges {}".format(index,len(all_edgelist)))

def split(args):
    indexlist = []
    index = 0
    datasets = args.datasets.split("+")
    splitedvec= ['\n']*len(datasets)
    for dataset in datasets:
        with open("data/"+dataset+"/"+dataset+".edgelist",'r') as f:
            L = f.readlines()
            num_node = int(L[0].strip())
        indexlist.append(index)
        index += num_node
        
    with open("emb/merge"+args.mergeversion+"/merge.struc2vec."+args.mergeversion+args.desversion+".embs32",'r') as f:
        L=f.readlines()[1:]
        for i in range(len(L)):
            row = L[i].strip().split()
            id = int(row[0])
            vec = " ".join(row[1:])
            datasetid,offset = 0,0
            for d,j in enumerate(indexlist):
                if j<=id:
                    datasetid,offset = d,j
                else:
                    break
            true_id = id - offset
            true_row = str(true_id)+' '+vec+"\n"
            splitedvec[datasetid]+=true_row

    for id,name in enumerate(datasets):
        if not os.path.exists("emb/"+name):
            os.mkdir("emb/"+name)
        with open(os.path.join("emb",name,name+".struc2vec."+str(id)+args.mergeversion+args.desversion+".embs32"),'w') as f:
            f.write(splitedvec[id])



if __name__=="__main__":
    args = parse_args()
    if args.mode == "merge":
        merge(args)
    elif args.mode == "split":
        split(args)