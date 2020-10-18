from src.baselines.sampling_methods.kcenter_greedy import kCenterGreedy
import numpy as np

class CoreSetQuery(object):
    def __init__(self, batchsize,trainsetid):
        self.q = []
        self.trainsetid = trainsetid
        self.batchsize = batchsize
        self.alreadyselected = [[] for i in range(batchsize)]
        for i in range(batchsize):
            self.q.append(kCenterGreedy())
        pass

    def __call__(self, outputs):
        ret = []
        for id,row in enumerate(self.alreadyselected):
            selected = self.selectOneNode(outputs[id], row, self.q[id])
            ret.append(selected)
        ret = np.array(ret)  #.reshape(-1,1)
        ret = ret.tolist()
        self.alreadyselected = [x + ret[id] for id, x in enumerate(self.alreadyselected)]
        
        selectedtrueid = []
        for i in range(self.batchsize):
            print(ret[i])
            selectedtrueid.append(self.trainsetid[i][ret[i][0]])
            
        return selectedtrueid


    def selectOneNode(self, output, pool,q):
        selected = q.select_batch(output,pool,1)
        return selected


def unitTest():
    q = CoreSetQuery(3)

    # pool = np.array([[2,3],[2,3],[4,6]])

    for i in range(3):
        features = np.random.randn(3, 10, 5)
        selected = q(features)
        print(selected)


if __name__ == "__main__":
    unitTest()


