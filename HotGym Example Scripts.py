# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 14:26:25 2020

@author: Carlton

This file goes through the HotGym example with PyHTM, showing how to:
    -Train the SP
    -Train the TM
    -Plot predictions and anomaly scores
    
A discussion about the final figure:
    Looking at the predictions vs. actual data, the TM is doing very well
    in learning the various periodicities in the data.
    
    However, the anomaly scores are a bit wild. I'm still uncertain whether
    this is due to the way in which I calculate and normalize the scores and log-likelihoods,
    or if it is because of a flaw in the encodings--such as a poorly chosen resolution or sparsity.
    
NOTE: Unlike the other programs, this one can take several hours to run, depending on the 
num_cells argument. If you select 4 it will only take a few minutes, but the performance
will not be as good due to the higher complexity of the HotGym data compared to the
contrived signals used in the simpler examples.
    
"""
import datetime as dt
import csv
from PyHTM import *
import time

def HotGym_SP_Example():
#Trains an SP on the HotGym data.
        
    #Read the data
    num_records = 4390
    file_path = 'one_hot_gym_data.csv'
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader][3:]
        data = data[:num_records]
            
    #Split it into datetimes objects and power values
    dates = [dt.datetime.strptime(row[0], "%m/%d/%Y %H:%M") for row in data]
    power = [float(row[1]) for row in data]
    
    #Make the encoders
    scalar_enc = ScalarEncoder(n = 1000, w = 50, minval = 0, maxval = 90)
    date_enc = DateEncoder(month = [8*12,20], day = [8*7,20], hour = [8*24,20], weekend = [80,40])
    
    #For convenience, wrap everything in a MultiEncoder
    enc = MultiEncoder([date_enc,scalar_enc])
    
    #Instantiate the pooler
    column_num = 1000
    active_cols = 40
    perm_inc = 0.04
    perm_dec = 0.008
    perm_thresh = 0.1
    boost = 3
    
    sp = SpatialPooler(source=enc, column_num=column_num,
                       max_active_cols=active_cols,
                       perm_increment=perm_inc,
                       perm_decrement=perm_dec,
                       boost_str=boost)
    
    start = time.time()
    #Train the pooler on the available data
    for index in range(num_records):
        if (index+1) % 100 == 0:
            end = time.time()
            print("Processed SP input {} out of 4390...".format(index+1))
            print("That took {} seconds.".format(end-start))
            start = time.time()
        sp.process_input(enc.encode([dates[index],power[index]]))
    
    return sp, enc, dates, power
    
def HotGym_TM_Example(sp,enc,dates,power):
    #Trains a TM on the HotGym data; plots predictions and anomaly scores.
    #It performs better with more cells, but at 30 cells it takes all night to run
    #the program.
    
    #Instantiate the temporal memory
    num_records = 4390
    num_cells = 20
    stimulus_thresh = 4     #Threshold to become predictive (and subsequently active)
    at = AnomalyTracker()
    tm = TemporalMemory(spatial_pooler=sp,
                        anomaly_tracker=at,
                        num_cells=num_cells,
                        stimulus_thresh=stimulus_thresh,
                        subthreshold_learning=False)
    
    #Train the temporal memory
    #You may see a RuntimeWarning about divide by 0--it won't cause any issues.
    active_SDRs = []
    pred_SDRs = []
    start = time.time()
    for index in range(num_records):
        if (index+1) % 100 == 0:
            print("Processed TM input {} out of 4390...".format(index+1))
            end = time.time()
            print("That took took {} seconds.".format(end-start))
            start = time.time()
        act, pred = tm.process_input(enc.encode([dates[index],power[index]]),sparse_output=False)
        active_SDRs.append(act)
        pred_SDRs.append(pred)
        
    #Train a translator for the power readings
    reg = Regressor(active_SDRs,power)
    
    #Translate the predicted power readings
    pred_translations = reg.translate(pred_SDRs)
        
    #Make some visualizations
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(40,10))
    
    #Plot the power readings
    ax1.plot(power)
    ax1.set_title('Power Readings')
    
    #Plot the power readings against the translated predictions of the TM
    ax2.plot(power,c='red',label='Real Data')
    ax2.plot(pred_translations,c='green',label='Predictions')
    ax2.legend()
    ax2.set_title('Comparison')
    
    #Plot the anomaly scores and log-likelihoods
    ax3.plot(at.scores,c='red',label='Scores')
    ax3.plot(at.likelihoods,c='green',label='Likelihoods')
    ax3.legend()
    ax3.set_title('Anomaly Readings')
    
    return tm, at
        
       
if __name__ == '__main__':
    sp, enc, dates, power = HotGym_SP_Example()
    tm, at = HotGym_TM_Example(sp,enc,dates,power)

    

