from collections import OrderedDict
from collections import Counter
import pickle as pkl
def readAminer():
    f = open("../dataset/AMiner-Paper.txt",'r')
    allitems = {}
    line = f.readline()
    counter = 0
    curitem = {}
    while(line):
        if line=='\n':
            allitems[curitem['id']]=curitem
        if line[0]!='#':
            line = f.readline()
            continue
        line = line.strip()
        lead = line.split()[0]
        
        try:
            if lead == "#index":
                ## handle the last one
                curitem = {"reference":[]}
                thisid = int(line.split()[-1])
                if thisid%100000==0:
                    print(thisid)
                curitem["id"] = thisid
            elif lead == "#*":
                curitem["title"] = " ".join(line.split()[1:])
            elif lead == "#@":
                curitem["author"] = " ".join(line.split()[1:]).split(";")
            elif lead == "#o":
                curitem["location"] = " ".join(line.split()[1:]).split(";")    
            elif lead == "#t":
                try:
                    curitem["year"] = int(line.split()[1]) 
                except:
                    curitem["year"] = -1     
            elif lead =="#c":
                curitem["conference"] = " ".join(line.split()[1:])
            elif lead =="#%":
                curitem["reference"].append(int(line.split()[1]))
            elif lead == "#!":
                curitem["abstract"] = " ".join(line.split()[1:])
            else:
                print("not correct line {}:{}".format(counter, line))
                raise ValueError
        except:
            print("error {}:{}".format(counter,line))
        line = f.readline()

    orderedict = OrderedDict(sorted(allitems.items(),key=lambda x:x[0]))
    with open("../dataset/AMiner-processed/ordereddict.pkl",'wb') as wfile:
        pkl.dump(orderedict,wfile)
    f.close()

def count_conference(raw):
    conferences = set()
    for i in raw:
        if i[1]["conference"] not in conferences:
            conferences.add(i[1]["conference"])
    print("conferences in total",len(conferences))
    print(list(conferences)[:100])
        
        

def process():
    print("begin open")
    with open("../dataset/AMiner-processed/ordereddict.pkl",'rb') as rfile:
        raw = pkl.load(rfile)
    print("open done")
    valuable = []
    count1 ,count2, count3 =0,0,0
    for i in raw.items():
        if "conference" not in i[1]:
            count1+=1
            continue
        if "title" not in i[1] and "abstract" not in i[1]:
            count2+=1
            continue
        if "location" not in i[1] or "-" in i[1]["location"]:
            count3+=1
            continue
        valuable.append(i)
    print(count1,count2,count3,len(valuable))
    count_conference(valuable)




if __name__=="__main__":
    # readAminer()
    process()