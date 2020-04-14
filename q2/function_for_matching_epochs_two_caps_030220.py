# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 17:47:09 2019

@author: Amir
"""

def matchPartners(subject1 = None, filesToProcessS2 = None, locationSettings = None):
    
    #Test if all vars were defined
    assert len([t for t in [subject1, filesToProcessS2, locationSettings] if t is None]) == 0, "one of the vars was not defined" 
    
    #Loading subjects
    s1 = subject1
    #Getting the matching file - subject 2
    if isinstance(filesToProcessS2, list):
        s2 = filesToProcessS2[filesToProcessS2.index(s1.replace("SUBJECT1", "SUBJECT2"))]
    else:
        s2 = filesToProcessS2
    
#    print(s1, "\n", s2)
    
    #Load files
    epochsS1 = mne.read_epochs(s1, preload=True, verbose = False)
    epochsS2 = mne.read_epochs(s2, preload=True, verbose = False)
    
    assert epochsS1.info["ch_names"] == epochsS2.info["ch_names"], "files don't have the same channels order"
    
    #Rename electrodes names
    epochsS1.rename_channels(dict(zip(epochsS1.info["ch_names"], [i + "-0" for i in epochsS1.info["ch_names"]])))
    epochsS2.rename_channels(dict(zip(epochsS2.info["ch_names"], [i + "-1" for i in epochsS2.info["ch_names"]])))
    
    #Combining the subjects to one cap
    combined = combineEpochs(epochsS1 = epochsS1, epochsS2 = epochsS2)
    
    #Adding sensors(channles) locations
    if type(combined) is not str:
        combined.info["chs"] = locationSettings.copy()
    
    return combined


def matchDescription(epochs = None, description = None):
    
    assert len(epochs.drop_log) == len(description), "drop_log and description not the same length"
    
    indexDrop = []
    goodCounter = -1
    for i in range(0, len(epochs.drop_log)):
        #Testing for good data in subject
        if any(epochs.drop_log[i]) == False:
            #Counter of good data, for indexing the epochs file
            goodCounter += 1
            #Testing to see if in the description the data is bad
            if description[i] == ["bad data"]:
                #Adding the index goodCounter to the list for droping
                indexDrop.append(goodCounter)
    
    #Droping the bad epochs
    epochs.drop(indexDrop, reason ='bad combined')
    
    epochs.drop_bad()
    return epochs

def combineEpochs(epochsS1 = None, epochsS2 = None):
    
    assert len(epochsS1.drop_log) == len(epochsS2.drop_log), "drop_log not the same length"
    
    #Creating the list of the good/bad epochs in both subjects
    description = []
    for logA, logB in zip(epochsS1.drop_log, epochsS2.drop_log): 
        if (any(logA) == True) | (any(logB) == True):
            description.append(["bad data"])
        else:
            description.append(["good data"])
    
    if len([d for d in description if d == ["good data"]]) < 5: 
        return("Not enoguh good data")
    
    
    #Matching that bad/good epochs for each particiant. It's the intersection of good epochs
    matchDescription(epochs=epochsS1, description=description)
    matchDescription(epochs=epochsS2, description=description)
    
    #Combine matched epochs as one cap, for that I rebuild the two epochs as one epoch structure from scratch
    ##concatenating the data from the caps as one
    #Test if epochs are in the same freq. If not, downsample the higher to 500fq
    if epochsS1.info["sfreq"] != epochsS2.info["sfreq"]:
        #set Sample rate
        if min(epochsS1.info["sfreq"], epochsS2.info["sfreq"]) == 500:
            newsampleRate = 250                
        else:
            for i in range(int(min(epochsS1.info["sfreq"], epochsS2.info["sfreq"])), 100, -1): 
                if epochsS1.copy().resample(i, npad='auto').to_data_frame().shape == epochsS2.copy().resample(i, npad='auto').to_data_frame().shape:
                    newsampleRate = i
                    break

        epochsS1.resample(newsampleRate, npad='auto')
        epochsS2.resample(newsampleRate, npad='auto')   

    #Concatenate epochs from subject 1 and subject2
    data = np.concatenate((epochsS1, epochsS2), axis=1)
    ##Creating an info structure
    info = mne.create_info(
            ch_names = list(epochsS1.info["ch_names"] + epochsS2.info["ch_names"]),
            ch_types = np.repeat("eeg", len(list(epochsS1.info["ch_names"] + epochsS2.info["ch_names"]))),
            sfreq = epochsS1.info["sfreq"])
    ##Creating an events structure
    events = np.zeros((data.shape[0], 3), dtype="int32")
    
    #Naming the events by the name of of the original epoch number. 
    #e.g. event == 289 is epoch 289 in the original data
    eventConter = 0
    for i, d in enumerate(description):
        if d == ['good data']:
            events[eventConter][0] = i
            events[eventConter][2] = i 
            eventConter +=1
    ##Creating event ID
    event_id = dict(zip([str(item[2]) for item in events], [item[2] for item in events]))
    ##Time of each epoch
    tmin = -0.5
    ##Building the epoch structure
    combined = mne.EpochsArray(data, info, events, tmin, event_id)
    ##Editing the channels locations
    combined.info["chs"] = epochsS1.info["chs"] + epochsS2.info["chs"]
    
    #test to see that all the good epochs are in the same length
    if len(set(map(len,[epochsS1, epochsS2, combined]))) == 1 and sum([i == ['good data'] for i in description]) == len(epochsS1):
        print("All are the same length")
    else:
        print("ERROR - They are not the same length!") 

    return combined
    
def connectivityCalc(data = None, sfreqMin = None, sfreqMax = None, indices=None, n_jobs=1, method='plv', mode='multitaper'):
    # Calculating connectivity
    fmin, fmax = int(sfreqMin), int(sfreqMax)
    sfreq = data.info['sfreq']  # the sampling frequency
    tmin = 0.0  # exclude the baseline period
    con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
        data, method=method, mode=mode, sfreq=sfreq, fmin=fmin, fmax=fmax, indices=None,
        faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=n_jobs)
    
    ch_names = data.ch_names
    idx = [ch_names.index(name) for name in ch_names]
    con = con[idx][:, idx]
    # con is a 3D array where the last dimension is size one since we averaged
    # over frequencies in a single band. Here we make it 2D
    con = con[:, :, 0]
    
    return con

def exportCon(con = None, path = None, chNames = None):
    np.savetxt(path, con, delimiter=",")
    df = pd.read_csv(path)
    df.columns = chNames
    df["rownames"] = chNames[1:]
    df.to_csv(path)


from scipy.signal import hilbert
from mne.utils import _check_combine
from mne.source_estimate import _BaseSourceEstimate
from mne.filter import next_fast_len
from astropy.stats import circcorrcoef
from time import time

def ccorr(data):
    """Compute the CCOR.
    Parameters
    ----------
    data : array-like, shape=(n_epochs, n_signals, n_times) | generator
        The data from which to compute connectivity.
        The array-like object can also be a list/generator of array,
        each with shape (n_signals, n_times), or a :class:`~mne.SourceEstimate`
        object (and ``stc.data`` will be used). If it's float data,
        the Hilbert transform will be applied; if it's complex data,
        it's assumed the Hilbert has already been applied.

    Returns
    -------
    corr : ndarray, shape ([n_epochs, ]n_nodes, n_nodes)
        The pairwise orthogonal envelope correlations.
        This matrix is symmetric. If combine is None, the array
        with have three dimensions, the first of which is ``n_epochs``.
    """
     
    #Get the real epochs number
    realEpochesNumber = list(data.event_id.values())
    
    #Get channel names
    chNames = list()
    for ch1 in data.ch_names:
        for ch2 in data.ch_names:
            chNames.append([ch1, ch2])
    
    #Calculating ccorr on the epochs data. 
    corrs = list()
    for ei, epoch_data in enumerate(data):
        if isinstance(epoch_data, _BaseSourceEstimate):
            epoch_data = epoch_data.data
        if epoch_data.ndim != 2:
            raise ValueError('Each entry in data must be 2D, got shape %s'
                             % (epoch_data.shape,))
        #get the shape of epochs
        n_nodes, n_times = epoch_data.shape
        # Get the complex envelope (allowing complex inputs allows people
        # to do raw.apply_hilbert if they want)
        if epoch_data.dtype in (np.float32, np.float64):
            n_fft = next_fast_len(n_times)
            epoch_data = hilbert(epoch_data, N=n_fft, axis=-1)[..., :n_times]

        if epoch_data.dtype not in (np.complex64, np.complex128):
            raise ValueError('data.dtype must be float or complex, got %s'
                             % (epoch_data.dtype,))
        
        #Convert to angle 
        epoch_data = np.angle(epoch_data)
        
        #The actual calculations
        corr = np.zeros((n_nodes,n_nodes))
        for n1 in range(n_nodes):
            for n2 in range(n_nodes):
                corr[n1, n2] = circcorrcoef(epoch_data[n1,:], epoch_data[n2,:])
        
        #Changing to pd.DataFrame and adding the ch names + realEpoch number 
        corr = pd.concat([pd.DataFrame(chNames), pd.DataFrame(corr.flatten())], axis = 1)
        corr = corr.assign(realEpoch = realEpochesNumber[ei])
        
        #Appending to the list of ccorr epochs
        corrs.append(corr)
              
        del corr

    corr = pd.concat(corrs)
    
    return corr