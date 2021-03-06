# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 19:42:01 2020

@author: Amir
"""
# Loading modules and setting up paths
import os, sys
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


path = "D:/Amir/Dropbox/Studies BIU/Ruth Feldman/My Thesis/My Analysis/EEG/EEG/shared/EEG question/q2"


fileExt = ".fif"

#Getting all the files in the folder
listOfFiles = [x[0]+"/"+f for x in os.walk(path) for f in x[2] if
               f.endswith(fileExt)] 

#locations for cap1 and cap2 in the combined file
from mpl_toolkits.mplot3d import Axes3D
from copy import copy

#Creating combined object
s1 = listOfFiles[0]
s2 = listOfFiles[1]
epochsS1 = mne.read_epochs(s1, preload=True)
epochsS2 = mne.read_epochs(s2, preload=True)
epochsS1.rename_channels(dict(zip(epochsS1.info["ch_names"], [i + "-0" for i in epochsS1.info["ch_names"]])))
epochsS2.rename_channels(dict(zip(epochsS2.info["ch_names"], [i + "-1" for i in epochsS2.info["ch_names"]])))
combined = combineEpochs(epochsS1 = epochsS1, epochsS2 = epochsS2)

#Calculating locations
locations = copy(np.array([ch['loc'] for ch in combined.info['chs']]))
cap1_locations = locations[:31, :3]
print("Mean: ", np.nanmean(cap1_locations, axis=0))
print("Min: ", np.nanmin(cap1_locations, axis=0))
print("Max: ", np.nanmax(cap1_locations, axis=0))

translate = [0, 0.25, 0]
rotZ = np.pi

cap2_locations = copy(cap1_locations)
newX = cap2_locations[:, 0] * np.cos(rotZ) - cap2_locations[:, 1] * np.sin(rotZ)
newY = cap2_locations[:, 0] * np.sin(rotZ) + cap2_locations[:, 1] * np.cos(rotZ)
cap2_locations[:, 0] = newX
cap2_locations[:, 1] = newY
cap2_locations = cap2_locations + translate
print("Mean: ", np.nanmean(cap2_locations, axis=0))
print("Min: ", np.nanmin(cap2_locations, axis=0))
print("Max: ", np.nanmax(cap2_locations, axis=0))
sens_loc = np.concatenate((cap1_locations, cap2_locations), axis=0)

#testing that the new locations and the old locations are at the same length
assert len([ch['loc'] for ch in combined.info['chs']]) == len(sens_loc), "the caps locations are not in the same length"

#Changing location 
for old, new in enumerate(sens_loc):
    combined.info["chs"][old]["loc"][0:3] = new[0:3]

locationSettings = combined.info["chs"].copy()

del cap1_locations, cap2_locations, old, new, newX, newY, rotZ, translate, s1, s2

#Creating matrix for sparse matrix
##single cap connectivity
epochs = mne.read_epochs(listOfFiles[0], preload=True)

##Calculate connectivity for single cap
x, y = find_ch_connectivity(epochs.info, ch_type="eeg")

##Transforming single cap to two caps
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

#Brain areas
centerS1  = ['Cz-0', 'Fz-0', 'Pz-0']
leftTemporalS1 = ["FT9-0", "TP9-0", "T7-0"]
rightTemporalS1 = ["FT10-0", "TP10-0",  "T8-0"]

chToTake = [i[:-2] for i in centerS1 + leftTemporalS1 + rightTemporalS1]

#areas = [[i[:-2] + "-1" for i in centerS1] + centerS1, 
#         [i[:-2] + "-1" for i in leftTemporalS1] + leftTemporalS1, 
#         [i[:-2] + "-1" for i in rightTemporalS1] + rightTemporalS1]

areas = [[i[:-2] for i in centerS1], 
         [i[:-2] for i in leftTemporalS1], 
         [i[:-2] for i in rightTemporalS1]]

sensloc = np.array([c['loc'][:3] for c in combined.info['chs']][:62])
plt.figure(figsize=[10,30])
plt.scatter(sensloc[:, 0], sensloc[:, 1])
#plt.title("Two Caps Connectivity - between connections")
for i, txt in enumerate(combined.info["ch_names"]):
        plt.annotate(combined.info["ch_names"][i], (sensloc[i, 0], sensloc[i, 1]))
for e1 in range(62):
    for e2 in range(62):
        if combined.info["ch_names"][e1][:-2] in chToTake and combined.info["ch_names"][e2][:-2] in chToTake:
            k1, k2 = list(map(capTest, [e1, e2]))
            if A[k1, k2] and e1 <= 30 and e2 >= 31:
                plt.plot((sensloc[e1, 0], sensloc[e2, 0]), (sensloc[e1, 1], sensloc[e2, 1]))
    
epochs.plot_sensors(ch_type='eeg', show_names = chToTake, kind = "topomap")

#####################Creating plot for example####################
import mayavi.mlab as mlab
import moviepy.editor as mpy

con = np.zeros([62, 62])
con.size
Itook = list()
for e1 in range(62):
    for e2 in range(62):
        for area in areas:
            i = 1
            #Between
#            if combined.info["ch_names"][e1][:-2] in area and combined.info["ch_names"][e2][:-2] in area:
#                k1, k2 = list(map(capTest, [e1, e2]))
#                if e1 <= 30 and e2 >= 31:
#                    con[e1][e2] = 0.5
#                    Itook.append([combined.info["ch_names"][e1], e1, combined.info["ch_names"][e2], e2])        

            #Cap1 
            if combined.info["ch_names"][e1][:-2] in area and combined.info["ch_names"][e2][:-2] in area:
                k1, k2 = list(map(capTest, [e1, e2]))
                if e1 <= 30 and e2 <= 30 and e1 != e2:
                    con[e1][e2] = 1
#                    Itook.append([combined.info["ch_names"][e1], e1, combined.info["ch_names"][e2], e2])
            #Cap2
            if combined.info["ch_names"][e1][:-2] in area and combined.info["ch_names"][e2][:-2] in area:
                k1, k2 = list(map(capTest, [e1, e2]))
                if e1 >= 31 and e2 >= 31 and e1 != e2:
                    con[e1][e2] = 1
#                    Itook.append([combined.info["ch_names"][e1], e1, combined.info["ch_names"][e2], e2])        
    
    sum(con==1)

A = csr_matrix(con.tolist())
plt.spy(A)

#mlab.clf()
fig = mlab.figure(size=(600, 600), bgcolor=(1, 1, 1))
points = mlab.points3d(sens_loc[:, 0], sens_loc[:, 1], sens_loc[:, 2],
                  color=(0.5, 0.5, 0.5), opacity=1, scale_factor=0.005,
                  figure=fig)

mlab.view(azimuth = 180, distance = 0.7, focalpoint="auto")

#######
# Get the strongest connections
n_con = len(con)**2  # show up to 3844 connections
min_dist = 0  # exclude sensors that are less than 5cm apart
threshold = np.sort(con, axis=None)[-n_con] #sort the con by size and pick the index of n_con
ii, jj = np.where(con > 0)


# Remove close connections
con_nodes = list()
con_val = list()
for i, j in zip(ii, jj):
    if linalg.norm(sens_loc[i] - sens_loc[j]) > min_dist:
        con_nodes.append((i, j))
        con_val.append(con[i, j])


con_val = np.array(con_val)

# Show the connections as tubes between sensors

#By General - all in the same color.
vmax = np.max(con_val)
vmin = np.min(con_val)
for val, nodes in zip(con_val, con_nodes):
    x1, y1, z1 = sens_loc[nodes[0]]
    x2, y2, z2 = sens_loc[nodes[1]]
    lines = mlab.plot3d([x1, x2], [y1, y2], [z1, z2], [val, val],
                             vmin=vmin, vmax=vmax, tube_radius=0.0002,
                             colormap='blue-red')
    lines.module_manager.scalar_lut_manager.reverse_lut = True



for area, color in zip(areas, [(1,0,0), #central
                                 (0, 1, 0),  #left
                                 (0, 0, 1)]): #right 
    #subject1
    for a in area:
        x1, y1, z1 = np.array([sens_loc[combined.info["ch_names"].index(a + "-0")] for a in area]).mean(axis = 0)
    #subject1
    for a in area:
        x2, y2, z2 = np.array([sens_loc[combined.info["ch_names"].index(a + "-1")] for a in area]).mean(axis = 0)
        linesTriangle = mlab.plot3d([x1, x2], [y1, y2], [z1, z2], [val, val],
                                    vmin=vmin, vmax=vmax, tube_radius=0.002,
                                    color= color) 
        

## Add the sensor names for the connections shown
#nodes_shown = list(set([n[0] for n in con_nodes] +
#                       [n[1] for n in con_nodes]))

nodes_shown = list(range(0, 62))

chNames = []
#Changing channels name as letters -M / -F
for i, c in enumerate(combined.info["ch_names"]):
    if c[:-2] in chToTake:
        if c[-1] == "0":
            chNames.append(c[:-1] + "M")
        elif c[-1] == "1":
            chNames.append(c[:-1] + "F")

#Channels name as number -0 / -1
#picks = mne.pick_types(combinedBaseline.info, eeg=True)

#Channels name as letters -M / -F
picks = np.array(list(range(0, len(chNames))))

counterif = -1
for i, node in enumerate(nodes_shown):
    if combined.info["ch_names"][i][:-2] in chToTake:
        counterif += 1
        x, y, z = sens_loc[i]
        mlab.text3d(x, y, z, chNames[counterif],
                    scale=0.005,
                    color=(0, 0, 0))



