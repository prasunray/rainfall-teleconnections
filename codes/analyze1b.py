"""
Analyze JJA precipitation data for W. Australia and SCA
"""
import numpy as np
import scipy.stats as st
from time import time
from numba import jit
fname1 = 'SCA_prec.npy'
fname2 = 'west_Aus_prec.npy'

#----- Load data and organize varianbles -----#

#load precipitation data
d1 = np.load(fname1) #SCA
d2 = np.load(fname2) #W. Australia
 
Ny = 2016-1998+1 #number of years

M,nloc1 = d1.shape #number of days, number of locations
Nd = M//Ny #number of days/yr
d1y = d1.reshape(Ny,Nd,nloc1)

M,nloc2 = d2.shape
Nd = M//Ny
d2y = d2.reshape(Ny,Nd,nloc2)
days = np.zeros((Ny,Nd),dtype=int)
for i in range(Ny):
    days[i,:] = np.arange(Nd,dtype=int)+i*365

days1d = days.reshape(M,) #index of days 0,1,...,91, 365,366,...
tau_max = 10 #delay for ES
#-----------------------------#


#----- Find EREs for each data set -----#

@jit(nopython=True)
def compute_ERE(dat,nloc,days1d):
    """
    Compute EREs above 95th percentile of wet days using precipitation data in dat
    """  
    print("Computing EREs for dataset...")
    ere_all = []
    counts = []
    x95_all = []
    #loop over locations
    for i in range(nloc):
        x = dat[:,i]
        d = days1d[x>1.0] #wet days
        x = x[x>1.0]
        x95 = np.percentile(x,95) #95th percentile
        x95_all.append(x95)
        ind = np.where(x>=x95)
        ere = d[ind[0]]
        
        #Aggregate consecutive EREs into single day
        ereN =[ere[0]]
        for i,d in enumerate(ere[1:]):
            if d==ere[i]+1:
                pass
            else:
                ereN.append(d)
        counts.append(len(ereN)) #number of EREs
        ere_all.append(ereN) 

    print("...done")
    return ere_all,counts,x95



t1 = time()

ere1_all,counts1,x951_all = compute_ERE(d1,nloc1,days1d)
ere2_all,counts2,x952_all = ere1_all,counts1,x951_all

#Check if EREs match SCA data
dshift = 151
E1 = np.load('SCA_ERE.npy',allow_pickle=True)
fail1 = []
for count,x in enumerate(zip(E1,ere1_all)):
    e1,ere1 = x
    if np.all(np.array(e1)-dshift==ere1):
        pass
    else:
        fail1.append(count)
#-----------------------------#


#----- Event synchronization -----#

#For two locations find the number of synchronized events

@jit(nopython=True)
def es_calc(e1,e2,de,tau_max=10):
    """event synchronization calculation for 
    time series e1 and e2
    """
    ind1 = np.where(np.abs(de)<tau_max) #only consider event pairs within tau_max days
    
    #compute tau_ij (dtmin)
    de1 = e1[1:]-e1[:-1]
    de2 = e2[1:]-e2[:-1]
    es = 0
    for i1,j1 in zip(ind1[0],ind1[1]): #loop over each pair of events
        dt = de[i1,j1]
        dtlist = []
        if i1>0:
            dtlist.append(de1[i1-1])
        if i1<de1.size:
            dtlist.append(de1[i1])
        if j1>0:
            dtlist.append(de2[j1-1])
        if j1<de2.size:
            dtlist.append(de2[j1])
        dtmin = min(dtlist)        
        if abs(dt)<dtmin/2:
            es+=1 #es found
    return es


i1 = 0
i2 = 0
N1 = len(ere1_all)
N2 = len(ere2_all)
es=  np.zeros((N1,N2),dtype=int)

#loop over each pair of locations
for count1,e1 in enumerate(ere1_all):
    print("count1=",count1)
    if counts1[count1]>2:
        for count2,e2 in enumerate(ere2_all):
            if counts2[count2]>2:
                de = np.subtract.outer(e1,e2)
                es12 = es_calc(np.array(e1),np.array(e2),de)
                es[count1,count2]=es12



#----- Null model ES calculation -----#

#find all (li,lj) pairs and store in pairs
pairs = set()
for c1 in counts1:
    for c2 in counts2:
        if (c2,c1) not in pairs:
            pairs.add((c1,c2))

reps = 2000
i = 0
j = 0
print("null model...")
nullDict = {}

#loop through each (li,lj) pair
for p in pairs:
    print("p=",p)
    li,lj = p
    esn = np.zeros(reps,dtype=int) 
    # x1 = np.zeros(M,dtype=int)
    # x1[:li]=1
    # x2 = np.zeros(M,dtype=int)
    # x2[:lj]=1
    for k in range(reps):

        # y1 = np.random.permutation(x1)
        # y2 = np.random.permutation(x2)
        # e1 = days1d[y1==1]
        # e2 = days1d[y2==1]
        e1 = np.sort(np.random.choice(days1d,li,replace=False)) #li random days
        e2 = np.sort(np.random.choice(days1d,lj,replace=False)) #lj random days
        de = np.subtract.outer(e1,e2)
        esn[k] = es_calc(e1,e2,de) #ES calculation
 
    nullDict[p] = np.percentile(esn,99.5) #99.5th percentile of esn values

print("done")

#use nullDict to find ES threshold values for each location pair
esnull = np.zeros_like(es)
for i in range(N1):
    for j in range(N2):
        li,lj = counts1[i],counts2[j]
        if (li,lj) in nullDict:
            esnull[i,j]=nullDict[(li,lj)]
        else:
            esnull[i,j]=nullDict[(lj,li)]

#apply threshold
links = es>esnull
links = links.astype(int)

t2 = time()
dt = t2-t1


