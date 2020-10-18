import logging
import sys


def ConfigRootLogger(name='', version=None, level="info"):
    logger = logging.getLogger(name)
    if level == "debug":
        logger.setLevel(logging.DEBUG)
    elif level == "info":
        logger.setLevel(logging.INFO)
    elif level == "warning":
        logger.setLevel(logging.WARNING)
    format = logging.Formatter("T[%(asctime)s]-V[{}]-POS[%(module)s."
                                      "%(funcName)s(line %(lineno)s)]-PID[%(process)d] %(levelname)s"
                                      ">>> %(message)s  ".format(version),"%H:%M:%S")
    stdout_format = logging.Formatter("T[%(asctime)s]-V[{}]-POS[%(module)s."
                                      "%(funcName)s(line %(lineno)s)]-PID[%(process)d] %(levelname)s"
                                      ">>> %(message)s  ".format(version),"%H:%M:%S")

    file_handler = logging.FileHandler("log.txt")
    file_handler.setFormatter(format)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(stdout_format)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


ConfigRootLogger("main","1",level="info")
logger = logging.getLogger("main")



def logargs(args,tablename="",width=120 ):
    length = 1
    L=[]
    l= "|"
    for id,arg in enumerate(vars(args)):
        name,value = arg, str(getattr(args, arg))
        nv = name+":"+value
        if length +(len(nv)+2)>width:
            L.append(l)
            l = "|"
            length = 1
        l += nv + " |"
        length += (len(nv)+2)
        if id+1 == len(vars(args)):
            L.append(l)
    printstr = niceprint(L)
    logger.info("{}:\n{}".format(tablename,printstr))

def niceprint(L,mark="-"):
    printstr = []
    printstr.append("-"*len(L[0]))
    printstr.append(L[0])
    for id in range(1,len(L)):
        printstr.append("-"*max(len(L[id-1]),len(L[id])))
        printstr.append(L[id])
    printstr.append("-"*len(L[-1]))
    printstr = "\n".join(printstr)
    return printstr

def logdicts(dic,tablename="",width=120 ):
    length = 1
    L=[]
    l= "|"
    tup = dic.items()
    for id,arg in enumerate(tup):
        name,value = arg
        nv = name+":"+str(value)
        if length +(len(nv)+2)>width:
            L.append(l)
            l = "|"
            length = 1
        l += nv + " |"
        length += (len(nv)+2)
        if id+1 == len(tup):
            L.append(l)
    printstr = niceprint(L)

    logger.info("{}:\n{}".format(tablename,printstr))




