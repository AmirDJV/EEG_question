# -*- coding: utf-8 -*-
"""
Created on Sun May 12 17:04:06 2019

@author: Amir
"""

import os
import mne
import pandas as pd
import pickle
import numpy as np
from scipy import linalg, stats
from mne import io
from mne.connectivity import spectral_connectivity
import winsound
import warnings
import re
from mne.channels import find_ch_connectivity
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import lil_matrix

def readPythonObject(path):
    #import pickle
    infile = open(path,'rb')
    objectName = pickle.load(infile)
    infile.close()
    return objectName

def makeSound(freq = 6000, # Hz
              duration = 3000): # millisecond
    #import winsound
    winsound.Beep(freq, duration)

#Load files    
datafilesPath = "D:/Amir/Dropbox/Studies BIU/Ruth Feldman/My Thesis/My Analysis/EEG/EEG/shared/EEG question data/single"
combinedBaseline = readPythonObject(datafilesPath + "/combinedBaseline.pkl")
conns = readPythonObject(datafilesPath + "/conns.pkl")
locationSettings = readPythonObject(datafilesPath + "/locationSettings.pkl")
locations = readPythonObject(datafilesPath + "/locations.pkl")
sens_loc = readPythonObject(datafilesPath + "/sens_loc.pkl")
from sklearn.externals import joblib
combinedList = joblib.load(datafilesPath + "/combinedList.pkl")
makeSound()

#My freqbands
freqbands = {'alpha': (8, 14), 'beta': (14, 30), 'gamma': (30, 60)}

#Creating meatconnectivity clusters only for the between subject channels

#For option 1
nchan = int(len(combinedBaseline.info["ch_names"]))
bin=0
idx = []
electrodes = []
for e1 in range(0, int(nchan / 2)):
    for e2 in range(int(nchan / 2), nchan):
        if e2>e1:
            idx.append(bin)
            electrodes.append((e1, e2))
        bin = bin + 1

print("my idx is:", len(idx))
print("my electrodes is:", len(electrodes))

##For option 2
#nchan = int(len(combinedBaseline.info["ch_names"]))
#bin=0
#idx = []
#electrodes = []
#for e1 in range(nchan):
#    for e2 in range(nchan):
#        if e2>e1:
#            idx.append(bin)
#            electrodes.append((e1, e2))
#        bin = bin + 1
#print("my idx is:", len(idx))
#print("my electrodes is:", len(electrodes))
#if (nchan**2 - nchan) / 2 == len(idx): 
#    print(len(idx))

#Creating 3D ndarray for the connections between the vectors (subject, freq, connections)
frequencies_labels = list(freqbands.keys())
vectconn = np.zeros([len(conns), len(frequencies_labels), len(idx)])
for f_idx, f_lab in enumerate(frequencies_labels):
    for s_idx, s_conn in enumerate(conns):
        vectconn[s_idx, f_idx, :] = s_conn[idx, f_idx]

print(vectconn.shape)

#Creating matrix for sparse matrix
##single cap connectivity
epochs = mne.read_epochs(datafilesPath + "/ICACorrection_N135_SUBJECT2_PART1_F010_epo.fif", preload=True)

##Testing to see that the names of the single/two caps are the same. 
assert [c + "-0" for c in epochs.info["ch_names"]] == [c for c in combinedBaseline.info["ch_names"][:31]], "The channels name in the single and two caps are not the same"

##Calculate connectivity for single cap
x, y = find_ch_connectivity(epochs.info, ch_type="eeg")
##Plot
sensloc = np.array([c['loc'][:3] for c in epochs.info['chs']][:31])
plt.figure(figsize=[10,10])
plt.scatter(sensloc[:, 0], sensloc[:, 1])
plt.title("single cap connectivity")
for i, txt in enumerate(epochs.info["ch_names"]):
        plt.annotate(epochs.info["ch_names"][i], (sensloc[i, 0], sensloc[i, 1]))
for e1 in range(31):
    for e2 in range(31):
        if x[e1,e2]:
            plt.plot((sensloc[e1, 0], sensloc[e2, 0]), (sensloc[e1, 1], sensloc[e2, 1]))


######Option one######
#Creating A stright from x
from scipy.sparse import csr_matrix
A = csr_matrix(x.toarray())          
  
##Plot sprase matrix
plt.figure(figsize=[10,8])
plt.spy(A)
plt.title("A sparse matrix for single cap")
plt.show()

##Plot two caps connectivity
def capTest(x):
    if x >= 31:
        out = x - 31
    else:
        out = x 
    return(out)
    
    
sensloc = np.array([c['loc'][:3] for c in combinedBaseline.info['chs']][:62])
plt.figure(figsize=[10,10])
plt.scatter(sensloc[:, 0], sensloc[:, 1])
plt.title("two caps connectivity, all between participants connections")
for i, txt in enumerate(combinedBaseline.info["ch_names"]):
        plt.annotate(combinedBaseline.info["ch_names"][i], (sensloc[i, 0], sensloc[i, 1]))
for e1 in range(62):
    for e2 in range(62):
        k1, k2 = list(map(capTest, [e1, e2]))
        if A[k1, k2] and e1 <= 30 and e2 >= 31:
            plt.plot((sensloc[e1, 0], sensloc[e2, 0]), (sensloc[e1, 1], sensloc[e2, 1]))

##Calculate metaconn
metaconn = np.zeros((vectconn.shape[-1], vectconn.shape[-1]))
for ne1, (e11,e12) in tqdm(enumerate(electrodes)):
    for ne2, (e21,e22) in enumerate(electrodes):
        #Testing to see if the channel is from cap 1 or two. 
        #if from cap2 I subtract 31 to know the location. 
        k11, k12, k21, k22 = list(map(capTest, [e11, e12, e21, e22]))
        #Calculate metaconn
        metaconn[ne1, ne2] = (A[k11, k21])
        
print(vectconn.shape[-1])
print(metaconn.shape)
makeSound()

plt.figure(figsize=[10,8])
plt.spy(lil_matrix(metaconn))
plt.title("metaconn sparse matrix")
plt.show()

#######Option two#####          
##Creating A from a multiple x
###Transforming single cap to two caps
#from scipy.sparse import csr_matrix
###Transforming single cap to two caps
#emptyTri = np.zeros(x.toarray().shape)
#leftMat = np.concatenate((emptyTri, x.toarray()), axis = 0)
#rightMat = np.concatenate((x.toarray(), emptyTri, ), axis = 0)
#fullMat = np.concatenate((leftMat, rightMat), axis = 1)
#
#from scipy.sparse import csr_matrix
#A = csr_matrix(fullMat)
#
###Plot sprase matrix
#plt.figure(figsize=[10,8])
#plt.spy(A)
#plt.title("A sparse matrix for single cap")
#plt.show()
#
###Plot two caps connectivity
#sensloc = np.array([c['loc'][:3] for c in combinedBaseline.info['chs']][:62])
#plt.figure(figsize=[10,10])
#plt.scatter(sensloc[:, 0], sensloc[:, 1])
#plt.title("two caps connectivity, all connections (intra & inter)")
#for i, txt in enumerate(combinedBaseline.info["ch_names"]):
#        plt.annotate(combinedBaseline.info["ch_names"][i], (sensloc[i, 0], sensloc[i, 1]))
#for e1 in range(62):
#    for e2 in range(62):
#        if A[e1, e2]:
#            plt.plot((sensloc[e1, 0], sensloc[e2, 0]), (sensloc[e1, 1], sensloc[e2, 1]))
#            
#            
#
###Calculate metaconn
#metaconn = np.zeros((vectconn.shape[-1], vectconn.shape[-1]))
#for ne1, (e11,e12) in tqdm(enumerate(electrodes)):
#    for ne2, (e21,e22) in enumerate(electrodes):
#        #Testing to see if the channel is from cap 1 or two. 
#        #if from cap2 I subtract 31 to know the location. 
#        k11, k12, k21, k22 = list(map(capTest, [e11, e12, e21, e22]))
#        #Calculate metaconn
#        metaconn[ne1, ne2] = ((A[k11, k22]) or (A[k12, k21]))
#        
#print(vectconn.shape[-1])
#print(metaconn.shape)
#makeSound()
#
##Plot for sparse matrix of connections
#plt.figure(figsize=[10,8])
#plt.spy(metaconn)
#plt.title("metaconn sparse matrix")
#plt.show()

#Seting groups for spliting the data
grps = np.zeros(vectconn.shape[0])
count = -1
for s_idx, c in tqdm(enumerate(combinedList)):
    for i, cSub in enumerate(c[1]):
        count +=1
#        print(cSub[i].info["subject_info"]["ID"])
        if cSub[i].info["subject_info"]["ID"][1] == "1" and cSub[i].info["subject_info"]["active"] == "m":
            grps[count] = 1
        elif cSub[i].info["subject_info"]["ID"][1] == "1" and cSub[i].info["subject_info"]["active"] == "f":
            grps[count] = 2
        elif cSub[i].info["subject_info"]["ID"][1] == "2" and cSub[i].info["subject_info"]["active"] == "m":
            grps[count] = 3
        elif cSub[i].info["subject_info"]["ID"][1] == "2" and cSub[i].info["subject_info"]["active"] == "f":
            grps[count] = 4
        elif cSub[i].info["subject_info"]["ID"][1] == "3" and cSub[i].info["subject_info"]["active"] == "m":
            grps[count] = 5
        elif cSub[i].info["subject_info"]["ID"][1] == "3" and cSub[i].info["subject_info"]["active"] == "f":
            grps[count] = 6

print("Number of male couples:", len([i for i in grps.tolist() if i == 1]))
print("Number of female couples:", len([i for i in grps.tolist() if i == 2]))
print("Number of male best friends:", len([i for i in grps.tolist() if i ==3]))
print("Number of female best friends:", len([i for i in grps.tolist() if i ==4]))
print("Number of male strangers:", len([i for i in grps.tolist() if i ==5]))
print("Number of female strangers:", len([i for i in grps.tolist() if i ==6]))

romanticCount = 0
goodFriendsCount = 0
strangeCount = 0
for s in supports:
    if s[-31:-30] == "1":
        romanticCount +=1
    if s[-31:-30] == "2":
        goodFriendsCount +=1
    if s[-31:-30] == "3":
        strangeCount +=1
print("romantic", romanticCount,
      "\ngood friends", goodFriendsCount, 
      "\nstrange", strangeCount)   

#Calculating permutation        
fobsList = []
clustersList = []
cluster_pvList = []
H0List = []
pvalue = 0.05
for f_idx, f_lab in enumerate(frequencies_labels):
    print(f_idx, f_lab)
    data1 = vectconn[np.argwhere(grps==1), f_idx, :]
    data2 = vectconn[np.argwhere(grps==2), f_idx, :]
    data3 = vectconn[np.argwhere(grps==3), f_idx, :]
    data4 = vectconn[np.argwhere(grps==4), f_idx, :]
    data5 = vectconn[np.argwhere(grps==5), f_idx, :]
    data6 = vectconn[np.argwhere(grps==6), f_idx, :]
    romantic = np.concatenate([data1, data2], axis = 0)
    goodFriends = np.concatenate([data3, data4], axis = 0)
    strangers = np.concatenate([data5, data6], axis = 0)
    Fobs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test(X=[romantic,
                                                                           goodFriends,
                                                                           strangers],
                                                                        tail=0,
                                                                        n_permutations = 1024,
                                                                        seed = 42, 
                                                                        step_down_p = pvalue,
#                                                                        connectivity = None,
                                                                        connectivity = lil_matrix(metaconn), 
                                                                        check_disjoint = False)
    fobsList.append(Fobs)
    clustersList.append(clusters)
    cluster_pvList.append(cluster_pv)
    H0List.append(H0)

makeSound()

for i, (p, c) in enumerate(zip(cluster_pvList, clustersList)):
    print("for freq", frequencies_labels[i], "I have" , len([pv for pv in p if pv < pvalue]), "pvs")
    print("for freq", frequencies_labels[i], "pv values" , [pv for pv in p if pv < pvalue])

#Creating final df as pandas
from itertools import product, compress
x, y, z = vectconn.shape
x_, y_, z_ = zip(*product(range(x), range(y), range(z)))
#df = pd.DataFrame(vectconn.flatten()).assign(group=x_, freq=y_, channel=z_)
#df.columns = ["value", "group", "freq", "channel"]
#df["ID"] = np.repeat(ID, vectconn.shape[2] * len(freqbands))
#df["group"] = np.repeat(grps, vectconn.shape[2] * len(freqbands))


#To know where the significant clusters are
cl = []
for i, p in enumerate(cluster_pvList):
    for c, pv in enumerate(p):
        if pv < pvalue:
            cl.append([i,c])

#Take the channel from each significant cluster
cPLV = []
for clusterNumber, cluster in enumerate(cl): 
    tempChannels = list(compress(electrodes, clustersList[cluster[0]][cluster[1]][0]))
    cPLV.append([cluster[0], clusterNumber, tempChannels])

for i in range(len(cPLV)):
    print(len(set(list(sum(cPLV[i][2], ())))))

#Create dataframe for each cluster
cDF = []
for clusterNumber, cluster in enumerate(cl): 
    tempdf = df[df["freq"] == cluster[0]]
    tempdf = tempdf[np.tile(clustersList[cluster[0]][cluster[1]][0],
                    vectconn.shape[0])].reset_index()
    cDF.append(tempdf)
        
for dat in cDF: 
    dat.groupby(["group"])['value'].mean().to_frame().plot(kind = "bar")


l = []
sensloc = np.array([c['loc'][:3] for c in combinedBaseline.info['chs']][:62])
plt.figure(figsize=[20,5])
for f_idx, f_lab in enumerate(frequencies_labels):
    cluster_pv = cluster_pvList[f_idx]
    clusters = clustersList[f_idx]
    assert len(cluster_pv) == len(clusters), "Number of pvalues not equal to number of clusters"
    plt.subplot(1,4,f_idx + 1)
    plt.scatter(sensloc[:, 0], sensloc[:, 1])
    #Add channels' name
    for i, txt in enumerate(combinedBaseline.info["ch_names"]):
        plt.annotate(combinedBaseline.info["ch_names"][i], (sensloc[i, 0], sensloc[i, 1]))
    #Plot significant connections
    if len(cluster_pv) > 0:
        for cl_idx, cluster in enumerate(clusters):
            if cluster_pv[cl_idx] < pvalue:
#                print(cl_idx, cluster_pv[cl_idx])
                color = np.random.rand(3)
                for ne, (e1,e2) in enumerate(electrodes):
                    if cluster[0][ne]:
                        l.append([f_lab, cl_idx, ne, (e1,e2),
                                  combinedBaseline.info["ch_names"][e1],
                                  combinedBaseline.info["ch_names"][e2]])
                        plt.plot((sensloc[e1, 0], sensloc[e2, 0]), (sensloc[e1, 1],
                                 sensloc[e2, 1]), linewidth=5, color=color)
        plt.title(f_lab)



########How I calculated conns#######

badfiles = []
conns = []
combinedList = []
combinedDataBaseline = []
combinedDataSupport = []
#freqbands = {'theta': (4, 8), 'alpha': (8, 14), 'beta': (14, 30), 'gamma': (30, 60)}
freqbands = {'alpha': (8, 14), 'beta': (14, 30), 'gamma': (30, 60)}
#baseline = baselines[0]
for baseline in tqdm(baselines):
    print(baseline)
    #Calculate baseline
    combinedBaseline = matchPartners(subject1 = baseline, filesToProcessS2 = filesToProcessS2, locationSettings = locationSettings)
    
    #Match support files
    support = [file for file in supports if file[-32:-28] == baseline[-32:-28]]
    
    combinedSupport = [matchPartners(subject1 = supportFile, filesToProcessS2 = filesToProcessS2, locationSettings = locationSettings) for supportFile in support]
    
    #Test if combined data frame was created
    ##If baseline is empty than continue to next file
    if str is type(combinedBaseline):
        badfiles.append(baseline)
        continue
    ##Testing that I have support files, if one of them is bad it will be removed
    for f, c in zip(support, combinedSupport):
        if str is type(c):
            badfiles.append(f)
            combinedSupport.remove(c)
    ##If support is empty than continue to next file
    if len(combinedSupport) == 0: 
        continue
       
    #Get active particiapns
    for f, c in zip(support, combinedSupport):    
        if getCue(f) == "F030" or getCue(f) == "F040":
            if "EEG data IDC" in f:
                active = activeParticipantIDC[activeParticipantIDC.ID == int(getSubjecctNumber(f)[1:])]["F" + getCue(f)[2]].values[0]
            if "EEG data BIU" in f:
                active = activeParticipant[activeParticipant.ID == int(getSubjecctNumber(f)[1:])]["F" + getCue(f)[2]].values[0]
        elif getCue(s1) == "F010":
            active = "base"
        #Test that active actually got "good" value
        if active not in ["female", "male", "base"]:
            badfiles.append(f)
            continue 
        #Set ID and active participant for combined data
        c.info["subject_info"] = {"ID" : getSubjecctNumber(f), "active" : active[0]} 
    
    #[i.info["subject_info"] for i in combinedSupport]
    
    #Creating list of combined data and their names
    combinedDataBaseline.append(baseline)
    combinedDataSupport.append(support)
    combinedList.append([combinedBaseline, combinedSupport])
    
    # which frequency bands to use:
    #Is defined at the start of the script
    ##freqbands = {'theta': (4, 8), 'alpha': (8, 14), 'beta': (14, 30), 'gamma': (30, 60)}
    fmin = np.array([f for f, _ in freqbands.values()])
    fmax = np.array([f for _, f in freqbands.values()])
    
    # perform the connectivity analysis:
    picks = mne.pick_types(combinedBaseline.info, eeg=True)
#    split = int(len(picks) / 2)
#    connection_pairs = mne.connectivity.seed_target_indices(picks[:split], picks[split:])
    connection_pairs = mne.connectivity.seed_target_indices(picks, picks)
    
    #Connectivity for baseline
    connB, freqsB, timesB, n_epochsB, n_tapersB = mne.connectivity.spectral_connectivity(
        combinedBaseline, method='plv', mode='multitaper', fmin=fmin, fmax=fmax, faverage=True,
        indices=connection_pairs, n_jobs=2, verbose=False)
    
    #Connectivity for support devided and loged by baseline  = log(task/baseline)
    for supCon in tqdm(combinedSupport):
        #Calculating the connectivty for current support file
        connS, freqsS, timesS, n_epochsS, n_tapersS = mne.connectivity.spectral_connectivity(
            supCon, method='plv', mode='multitaper', fmin=fmin, fmax=fmax, faverage=True,
            indices=connection_pairs, n_jobs=2, verbose=False)
        
        #Calculating the ratio, nan are transformed to zeros
        conn = np.nan_to_num(np.log(connS / connB))
        
        #Added to conns
        conns.append(conn)
        
        #Save individual arrays of each subject (there is probably a better way than to go via numpy to pandas to save this...)
        connectivity_file = '{path}/{ID}_{active}_conn.csv'.format(path=conSavePath, ID=supCon.info["subject_info"]["ID"], active=supCon.info["subject_info"]["active"])
        pd.DataFrame(
            dict(channel_1=connection_pairs[0], channel_2=connection_pairs[1],
                 **{key: conn[:, idx] for idx, key in enumerate(freqbands.keys())})
        ).to_csv(connectivity_file)
    
makeSound()












