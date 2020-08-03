# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 21:06:27 2020

@author: Carlton

This file contains a basic walkthrough for using the PyHTM library.

-First I will show how to create an encoder and train a spatial pooler to process
simple scalar values encoded as SDRs.

-I'll show how to use the regressor object to translate from pooler SDRs into visualizable plots.

-Then I will show how to create and train a temporal memory based on the spatial pooler.
I'll show how the regressor can be trained using the cell activity and used to translate
the predictive cell SDRs into visualizable scalar values.

-Finally, I'll feed the temporal memory some simple data with anomalies in it, and demonstrate anomaly detection.

-As a bonus, we can also try letting the TM generate the sequence on its own, once it has been learned.
To do so, we take the predictions, translate them to scalar values, then re-encode them
as input SDRs. Doing this repeatedly will, hopefully, generate a continuation of the
periodic sequence that the TM originally learned.
"""
import random as rand
import numpy as np
import math
import matplotlib.pyplot as plt
import random as rand
from PyHTM import *

def Basic_SP_Example():
    #This code will guide you through the process of training a spatial pooler
    #and provide some visualizations of encoded data, SP outputs, and regressor
    #translations.
    
    #Before we get started let's get some axes objects ready for plotting.
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    plt.subplots_adjust(hspace=0.5)

    #First, instantiate a ScalarEncoder.
    encoding_length = 1000  #The SDR will be this many bits long.
    encoding_width = 40     #Only this many bits will be active
    min_val = -1             
    max_val = 1          #The input range will be between 0 and 1000.
    enc = ScalarEncoder(n=encoding_length, w=encoding_width, minval=min_val, maxval=max_val)
    
    #Let's see what a sample encoding looks like. We can use the plot_SDR() function.
    plot_SDR(enc.encode(0), ax1)
    ax1.set_title('Sample Encoding')
    
    #Note that since the encoder isn't an RDSE, the on bits are all in a sequence.
    
    #Next, instantiate a SpatialPooler
    enc_dimensions = (encoding_length,)
    col_num = 1000          #I'll arbitrarily decide to use 1000 minicolumns.
    col_active_num = 40     #The output SDRs will have 4% sparsity.
    potential_conn = 0.75   #Each minicolumn will have the potential to connect to 75% of the input space.
    perm_inc = 0.5          #Permanence learning increment size.
    perm_dec = 0.008        #Choose a much smaller decrement rate in this case.
    #Go with the default for the other options.
    # sp = SpatialPooler(input_dim = (enc.n,),
    sp = SpatialPooler(source=enc, column_num = col_num,
                       max_active_cols=col_active_num,
                       potential_percent=potential_conn,
                       perm_increment=perm_inc,
                       perm_decrement=perm_dec)
    
    #The minicolumns initially have random connections. These will become more organized when the 
    #pooler is trained.
    plot_SDR(sp.columns[0].actual_connections[0], ax2)
    ax2.set_title('Untrained Connections')
    
    #To train the spatial pooler, I need to randomly generate many input values and encode them.
    random_vals = [min_val + (max_val-min_val)*rand.random() for i in range(2000)]
    random_val_encodings = [enc.encode(val) for val in random_vals]
    sp_random_outputs = []
    
    #Now I can train the spatial pooler to recognize and represent inputs in the range [0,1000)
    for index, encoding in enumerate(random_val_encodings):
        if (index + 1) % 100 == 0:
            print("Processed {} out of 2000 SP inputs...".format(index+1))
        sp_random_outputs.append(sp.process_input(encoding))
        
    #Now that the minicolumns have been trained, let's take another look at the connections.
    #Not all of the columns have been particularly active, so for this demonstration we'll
    #select one that has been active to see what its connection space has become.
    #plot_SDR(sp.columns[np.argmin(sp.boost_factors)].actual_connections[0], ax3)
    plot_SDR(sp.columns[np.argmin(sp.boost_factors)].actual_connections[0], ax3)
    ax3.set_title('Trained Connections')
    
    #The connections now look much less random, as the minicolumn has become associated
    #with a specific range of input encodings.
    
    #To see that the SP is accurately recognizing and representing inputs, let's
    #generate a sine-wave signal, encode it, process it, and translate the processed
    #SDRs back into scalars. We'll plot the original and output values against each other
    #to see if they match.
    
    #First generate a simple sine signal and encode it
    t = np.arange(0,2*np.pi, 2*np.pi/50)
    x = np.sin(t)
    sp_x = [sp.process_input(enc.encode(val)) for val in x]
    
    #Instantiate the regressor. It will automatically learn the data. 
    #The default regressor type is sklearn's KNeighborsRegressor.
    reg = Regressor(sp_random_outputs,random_vals)
    translations = reg.translate(sp_x)
    
    #Now let's plot the original and the translation together.
    ax4.plot(x,c='red',label='Original')
    ax4.scatter(range(len(x)),translations,c='green',label='Translations')
    ax4.legend()

    return sp, enc


def Basic_TM_Example(sp,enc, generate_data = False, add_anomalies = True, N = 400, N_periods=12):
    #With the Spatial Pooler trained, let's create a temporal memory.    
    
    #I get better results with a large # of cells like 30 instead of 4,
    #but it also takes far longer to run (~hours instead of ~minutes)
    num_cells = 4  #The more temporally complex the signal, the more cells you need
    threshold = 4   #The predictive overlap threshold
    at = AnomalyTracker()   #Initialize a default anomaly tracker
    #Leave the other parameters as their default settings
    tm = TemporalMemory(spatial_pooler = sp, anomaly_tracker=at,
                        num_cells=num_cells, 
                        stimulus_thresh = threshold)
    
    #Let's train it on a sine wave again, but we'll add some anomalies this time.
    period = 50
    
    t = np.arange(0,(N_periods)*2*np.pi, 2*np.pi/period)
    x = np.sin(t)*np.sin(2*t)
    #x = np.sin(t)
    
    #If we won't be generating extra data, set N = length of t
    if not generate_data:
        N = len(t)
    
    #If we will be adding deliberate anomalies:
    if add_anomalies:    
        #Replace the last 50 data points with a different signal.
        #This should surprise the TM.
        x[-period:] = np.sin(5*t[-period:])
    
        #Add some deliberate anomalies, more surprises.
        x[period*4] = -1
        x[period*4 + 1] = 1
        x[period*5] = 1
        x[period*5 + 1] = -1
    
    encoded_x = [enc.encode(val) for val in x]
    
    #We'll record all of the active and predictive cell SDRs for future use.
    actives = []
    preds = []
    tm.reset()
    for index, encoding in enumerate(encoded_x[:N]):
        if (index+1) % 100 == 0:
            print("Processed input {} of {} TM inputs...".format(index+1,N))
        act, pred = tm.process_input(encoding)
        actives.append(act)
        preds.append(pred)
        
    #Let's use the active cell outputs to learn the translation.
    actreg = Regressor(actives[period:N],x[period:N])
    predreg = Regressor(preds[period:N-1],x[period+1:N])
    
    #We can also attempt to generate the signal based purely on the TM's own predictions.
    if generate_data:
        predvals = Forecast(tm,enc,x,N,pred,predreg)
        
    #Now we'll translate the predictions and plot them against the actual output.
    translated_preds = predreg.translate(preds)
    translated_acts = actreg.translate(actives)    
    
    #Let's get some axes objects ready for plotting.
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(30,10))
    plt.subplots_adjust(hspace=0.5)
    
    ax1.plot(t,x,c='red',label='Real Data')
    ax1.scatter(t[1:N],translated_preds[:N-1],c='green',label='Translated Predictions')
    #ax1.scatter(t[:N], translated_acts[:N], c='blue',label='Translated Activity')
    
    if generate_data:
        ax1.scatter(t[N:],predvals,c='black',label='Generated Data')
    
    ax1.legend()
    ax1.set_title('Data Comparison')
    
    #We can also plot the anomaly scores!
    ax2.plot(at.scores,c='blue',label='Score')
    ax2.plot(at.likelihoods,c='Red',label='Likelihood')
    ax2.set_title('Anomaly Data')
    ax2.legend()
    
    ax3.plot(at.prediction_num,c='blue')
    ax3.set_title('Prediction Number (#predictive / #active)')
    
    diffs = [(translated_preds[i] - translated_acts[i+1]) for i in range(N-1)]
    ax4.plot(diffs,label='Prediction Error')
    ax4.set_title('Difference between Prediction and Actual Result')
    
    return tm, at

def Forecast(tm,enc,x,N,pred,predreg):
    #This is a convenience function that takes the predictions of a TM,
    #translates them into scalars, then turns those back into input-shape SDRs
    #and feeds them into the TM again.
    #In so doing, it generates a sequence that reveals the TM's idea of what
    #the future signal will look like more than just one time-step forward.
    #Note that this is pretty finicky, and usually only works with a large
    #number of cells. If there are too few it will just get stuck producing
    #the same value repeatedly.
    preds = []
    predvals = []
    val = predreg.translate(pred)
    sdr = enc.encode(val)
    for i in range(len(x[N:])):
        if (i+1) % 100 == 0:
            print("Processed additional input {} of {}...".format(i+1,len(x)-N))
        predvals.append(val)
        act, pred = tm.process_input(sdr)
        preds.append(pred)
        val = predreg.translate(pred)
        sdr = enc.encode(val)
    return predvals

if __name__ == '__main__':
    sp, enc = Basic_SP_Example()
    tm, at = Basic_TM_Example(sp, enc)




