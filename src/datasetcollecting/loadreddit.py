import networkx as nx
def readdata():
    from torch_geometric.datasets import Reddit
    reddit = Reddit(root = "data/reddit/")
    print(reddit.data.y.max())
    edges =  reddit.data.edge_index.transpose(0,1).numpy().tolist()
    g = nx.Graph()
    g = nx.Graph()
    g.add_edges_from(edges)
    g.add_edges_from(edges)
    
    g.subgraph([])
    
    


    


if __name__=="__main__":
    readdata()