
#******************************************************************************
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import math as m
import peakutils as pu
from scipy import signal
from scipy import stats
#******************************************************************************
# Reading FHR data into the pandas DataFrame, from a csv file, with no header
# First col is index; each subsequent col is a time-series
df = pd.read_csv('C:/Users/tians/Desktop/semester 3/ESE 670/milestone2/FHRDataCol.csv', header=None)
# print some stuff
df.dtypes
df.ndim
df.shape
# Extract columns from the DataFrame, and store in a series
# Sampling freq is 4 Hz; 240 samples == 1 minute
ts = pd.Series(df[1])

tsEpoch=ts[0:2400]  # A slice of the time series (10 mins)


def baseline(timeSeries):
    baseline_values = pu.baseline(timeSeries)
    return baseline_values
 

def plt_baseline(timeSeries, baseline):
    plt.plot(timeSeries, color='mediumspringgreen', label='origin data')
    plt.plot(baseline, color='salmon', label='baseline')
    plt.title('baseline vs origin_data') 

#Asymmetric Least Squares Smoothing
def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.csc_matrix(np.diff(np.eye(L), 2))
  w = np.ones(L)
  for i in xrange(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = sparse.linalg.spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z


##detect_peaks function (online)
def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()


#find valley
revArr = []
def find_valley(timeSeries):
    for i in range(len(timeSeries)):
        revArr.append(150 - timeSeries[i])
    peak = detect_peaks(revArr, mph=10, mpd=200)
    return peak


def find_left_range_peak(origin, index):
    back_index = index
    front_index = index - 1
    while(index > 1):
        front = origin[front_index]
        back = origin[back_index]
        if(front - back > 0):
            break
        else:
            front_index = front_index - 1
            back_index = back_index - 1
    return back_index


def find_right_range_peak(origin, index):
    back_index = index + 1
    front_index = index
    while(index < len(origin) and back_index < len(origin)):
        front = origin[front_index]
        back = origin[back_index]
        if(front - back < 0):
            break
        else:
            front_index = front_index + 1
            back_index = back_index + 1
    return front_index


def find_acc_range(origin, indexes):
    acc_range = []
    for i in range(len(indexes)):
        index = indexes[i]
        left_index = find_left_range_peak(origin, index)       
        #right_index = find_right_range_peak(origin, index) 
        right_index = index
        if(right_index - left_index >= 120):
            #30seconds = 120samples
            if(origin[right_index] - origin[left_index] >= 15):
                #onset to peak greater than 15bpm
                acc_range.append(left_index)
                acc_range.append(right_index)
    return acc_range


def find_left_range_valley(origin, index):
    back_index = index
    front_index = index - 1
    while(index > 0):
        front = origin[front_index]
        back = origin[back_index]
        if(front - back < 0):
            break
        else:
            front_index = front_index - 1
            back_index = back_index - 1
    return back_index

def find_right_range_valley(origin, index):
    back_index = index + 1
    front_index = index
    while(index < len(tsEpoch)):
        front = origin[front_index]
        back = origin[back_index]
        if(front - back > 0):
            break
        else:
            front_index = front_index + 1
            back_index = back_index + 1
    return front_index


def find_dec_range(origin, indexes):
    dec_range = []
    for i in range(len(indexes)):
        index = indexes[i]
        left_index = find_left_range_valley(origin, index)
        #right_index = find_right_range_valley(origin, index)
        right_index = index
        if(right_index - left_index >= 120):
            #30seconds = 120samples
            if(origin[right_index] - origin[left_index] <= 15):
                #onset to peak less than 15bpm
                dec_range.append(left_index)
                dec_range.append(right_index)
    return dec_range


merge = []
def mergeRanges(range1, range2):
    index1 = 0
    index2 = 0
    while(index1 < len(range1) and index2 < len(range2)):
        if(range1[index1] < range2[index2]):
            merge.append(range1[index1])
            merge.append(range1[index1 + 1])
            index1 = index1 + 2
        else:
            merge.append(range2[index2])
            merge.append(range2[index2 + 1])
            index2 = index2 + 2
    while(index1 < len(range1)):
        merge.append(range1[index1])
        index1 = index1 + 1
    while(index2 < len(range2)):
        merge.append(range2[index2])
        index2 = index2 + 1
    return merge
 
    
def getRange(indexes):
    ranges = []
    for i in range(len(indexes)):
        if(indexes[i] - 50 < 0):
            ranges.append(0)
        else:
            ranges.append(indexes[i] - 50)
        ranges.append(indexes[i] + 50)
    return ranges
    

def delRange(origin, del_range):
    remain = []
    start = 0
    end = 1
    i = 0
    while i < len(origin) and start <= len(del_range)-2:
        if(i == del_range[start]):
            while(i < del_range[end]):
                i = i + 1
            start = start + 2
            end = end + 2
        else:
            remain.append(origin[i])
            i = i + 1         
    return remain

   

 
       
#step1: find peaks and valleys
peak = detect_peaks(tsEpoch, mph=140, mpd=250, show=True)
valley = detect_peaks(tsEpoch, mph=140, mpd=250, valley=True, show=True)

#step2: find the baseline
#iteratively reduce peaks and valleys, get at least 2min baseline 
for i in range(5):
    mph_val = 140
    mpd_val = 250
    if(i==0):
        peak = detect_peaks(tsEpoch, mph=mph_val, mpd=mpd_val)
        valley = detect_peaks(tsEpoch, mph=mph_val, mpd=mpd_val, valley=True)
        #acc_range = find_acc_range(peak)
        #dec_range = find_dec_range(valley) 
        acc_range = getRange(peak)
        dec_range = getRange(valley)
        merge_range = mergeRanges(acc_range, dec_range) 
        remain = delRange(tsEpoch, merge_range)
        mph_val = mph_val - 5
        mpd_val = mpd_val - 10
    else:
        peak = detect_peaks(remain, mph=mph_val, mpd=mpd_val)
        valley = detect_peaks(remain, mph=mph_val, mpd=mpd_val, valley=True)
        #acc_range = find_acc_range(peak)
        #dec_range = find_dec_range(valley) 
        acc_range = getRange(peak)
        dec_range = getRange(valley)
        merge_range = mergeRanges(acc_range, dec_range) 
        remain = delRange(remain, merge_range)
        mph_val = mph_val - 5
        mpd_val = mpd_val - 10

sum = 0        
for i in range(len(remain)):
    sum = sum + remain[i]
baseline1_val = sum/(len(remain))
baseline1 = [baseline1_val]*len(tsEpoch)
if(baseline1_val < 110):
    print("Baseline value = %s, Abnormal baseline, bradycardia" %baseline1_val)
elif(baseline1_val > 160):
    print("Baseline value = %s, Abnormal baseline, tachycardia" %baseline1_val)
else:
    print("Baseline value = %s, normal baseline" %baseline1_val)
plt_baseline(tsEpoch, baseline1)

#step3: find baseline variability
baseline_var = []
minimal_var = []   # minial baseline variability
moderate_var = []
marked_var = []
for i in range(len(remain)):
    tmp = remain[i]
    diff = np.abs(tmp - baseline1_val)
    baseline_var.append(diff)
    if(diff < 5):
        minimal_var.append(i)
    elif(diff <= 25 and diff >= 6):
        moderate_var.append(i)
    elif(diff > 25):
        marked_var.append(i)
        
print ("The number of minimal variability is %d" %len(minimal_var))
print ("The number of moderate variability is %d" %len(moderate_var))
print ("The number of marked variability is %d" %len(marked_var))

#step4: find acceleration and deceleration
smooth = baseline_als(tsEpoch, 10000, 0.01, niter=10)
peak2 = detect_peaks(smooth, mph=140, mpd=250, show=True)
valley2 = detect_peaks(smooth, mph=140, mpd=250, valley=True, show=True)

acc = find_acc_range(smooth, peak2)
dec = find_dec_range(smooth, valley2)
print ("Acceleration ranges: ")
print acc
print ("Deceleration ranges: ")
print dec
plt.figure()
plt.plot(tsEpoch)
plt.plot(acc, tsEpoch[acc], 'rs', ms=6)
plt.plot(dec, tsEpoch[dec], 'ys', ms=4)
plt.show()
#find prolong acceleration/deceleration or baseline-change
for i in range(0, len(acc), 2):
    left = acc[i]
    right = acc[i+1]
    if(right - left >= 480 and right - left < 2400):
        print("Prolong acceleration from %d to %d" %left %right)
    elif(right - left == 2400):
        print("Acceleration: Baseline change!")
    else:
        print("Normal acceleration")

for i in range(0, len(dec), 2):
    left = dec[i]
    right = dec[i+1]
    if(right - left >= 480 and right - left < 2400):
        print("Prolong deceleration from %d to %d" %left %right)
    elif(right - left == 2400):
        print("deceleration: Baseline change!")
    else:
        print("Normal deceleration")