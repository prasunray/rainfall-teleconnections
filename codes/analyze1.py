"""
Analyze JJA precipitation data for W. Australia and SCA
"""
import numpy as np
import scipy.stats as st
from time import time
from numba import jit
fname1 = 'SCA_prec.npy'
fname2 = 'west_Aus_prec.npy'

#load precipitation data
d1 = np.load(fname1)
d2 = np.load(fname2)

Ny = 2016-1998+1

M,nloc1 = d1.shape
Nd = M//Ny
d1y = d1.reshape(Ny,Nd,nloc1)

M,nloc2 = d2.shape
Nd = M//Ny
d2y = d2.reshape(Ny,Nd,nloc2)
days = np.zeros((Ny,Nd))
for i in range(Ny):
    days[i,:] = np.arange(Nd,dtype=int)+i*365

days1d = days.reshape(M,)
tau_max = 10
#@jit(nopython=True)
def compute_ERE(d1,d2,nloc1,nloc2,days1d):
    #compute 95th percentile of wet days for each location in each dataset

    #find days above 95th percentile
    #aggregate consecutive days
    print("Computing EREs for dataset1...")
    ere1_all = []
    counts1 = []
    for i in range(nloc1):
        x = d1[:,i]
        d = days1d[x>1.0]
        x = x[x>1.0]
        x95 = np.percentile(x,95)
        ind = np.where(x>x95)
        ere1 = d[ind[0]]
        ere1N =[ere1[0]]

        for i,d in enumerate(ere1[1:]):
            if d==ere1[i]+1:
                pass
            else:
                ere1N.append(d)
        ere1N = np.array(ere1N,dtype=int)
        counts1.append(ere1N.size)
        ere1_all.append(ere1N)


    print("...done")

    print("Computing EREs for dataset2...")
    ere2_all = []
    counts2 = []
    for i in range(nloc2):
        x = d2[:,i]
        d = days1d[x>1.0]
        x = x[x>1.0]
        x95 = np.percentile(x,95)
        ind = np.where(x>x95)
        ere2 = d[ind[0]]
        ere2N = [ere2[0]]

        for i,d in enumerate(ere2[1:]):
            if d==ere2[i]+1:
                pass
            else:
                ere2N.append(d)
        ere2N = np.array(ere2N,dtype=int)
        counts2.append(ere2N.size)
        ere2_all.append(ere2N)

    print("...done")
    return ere1_all,ere2_all,counts1,counts2

t1 = time()
ere1_all,ere2_all,counts1,counts2 = compute_ERE(d1,d2,nloc1,nloc2,days1d)
t2 = time()
dt1 = t2-t1





#event synchronization

#for two locations find the number of synchronized events

i1 = 0
i2 = 0

e1 = ere1_all[i1]
e2 = ere2_all[i2]

de = np.subtract.outer(e1,e2)
ind1 = np.where(np.abs(de<=tau_max))
de1 = np.diff(e1)
de2 = np.diff(e2)


for i1,j1 in zip(ind1[0],ind1[1]):
    dt = de[i1,j1]
    



#null model

#generate random days with same number of events for each location in pair



#Find 99.5th percentile result from null model, compare to ES result
