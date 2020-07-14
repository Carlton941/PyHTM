# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 21:06:27 2020

@author: Carlton

This file contains a basic walkthrough for using the PyHTM library.

-First I will show how to create an encoder and train a spatial pooler to process
simple scalar values encoded as SDRs.

-I'll show how to use the regressor object to translate from pooler SDRs into visualizable plots.

-Then I will show how to create an d train a temporal memory based on the spatial pooler.
I'll show how the regressor can be trained using the cell activity and used to translate
the predictive cell SDRs into visualizable scalar values.

-Finally, I'll feed the temporal memory some simple data with anomalies in it, and demonstrate anomaly detection.

-The 
"""
#from PyHTM import *
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
    encoding_length = 1000  #The SDR will be 1000 bits long.
    encoding_width = 60     #Only 60 bits will be active--this is 6% sparsity.
    min_val = 0             
    max_val = 1000          #The input range will be between 0 and 1000.
    enc = ScalarEncoder(n=encoding_length, w=encoding_width, minval=min_val, maxval=max_val)
    
    #Let's see what a sample encoding looks like. We can use the plot_SDR() function.
    plot_SDR(enc.encode(500), ax1)
    ax1.set_title('Sample Encoding')
    
    #Note that since the encoder isn't an RDSE, the on bits are all in a sequence.
    
    #Next, instantiate a SpatialPooler
    enc_dimensions = (encoding_length,)
    col_num = 1000          #I'll arbitrarily decide to use 2000 minicolumns.
    col_active_num = 40     #The output SDRs will have 4% sparsity.
    potential_conn = 0.75   #Each minicolumn will have the potential to connect to 75% of the input space.
    perm_inc = 0.5          #Permanence learning increment size.
    perm_dec = 0.008        #Choose a much smaller decrement rate in this case.
    #Go with the default for the other options.
    sp = SpatialPooler(input_dim=enc_dimensions, active_cols=col_active_num,
                       potential_percent=potential_conn,
                       perm_increment=perm_inc,
                       perm_decrement=perm_dec)
    
    #The minicolumns initially have random connections. These will become more organized when the 
    #pooler is trained.
    plot_SDR(sp.columns[0].actual_connections, ax2)
    ax2.set_title('Untrained Connections')
    
    #To train the spatial pooler, I need to randomly generate many input values and encode them.
    random_vals = [max_val*rand.random() for i in range(2000)]
    random_val_encodings = [enc.encode(val) for val in random_vals]
    
    #Now I can train the spatial pooler to recognize and represent inputs in the range [0,1000)
    for index, encoding in enumerate(random_val_encodings):
        if index % 100 == 0:
            print("Processed {} out of 2000 inputs...".format(index))
        sp.process_input(encoding)
        
    #Now that the minicolumns have been trained, let's take another look at the connections.
    #Not all of the columns have been particularly active, so for this demonstration we'll
    #select one that has been active to see what its connection space has become.
    plot_SDR(sp.columns[np.argmin(sp.boost_factors)].actual_connections, ax3)
    ax3.set_title('Trained Connections')
    
    #The connections now look much less random, as the minicolumn has become associated
    #with a specific range of input encodings.
    
    #To see that the SP is accurately recognizing and representing inputs, let's
    #generate a sine-wave signal, encode it, process it, and translate the processed
    #SDRs back into scalars. We'll plot the original and output values against each other
    #to see if they match.
    
    #First generate a simple sine signal and encode it
    t = np.arange(0,2*np.pi, 2*np.pi/50)
    x = 500 + 500*np.sin(t)
    enc_x = [enc.encode(val) for val in x]
    
    #Instantiate the regressor. It will automatically learn the data. 
    #The default regressor type is sklearn's KNeighborsRegressor.
    reg = Regressor(enc_x,x)
    translations = reg.translate(enc_x)
    
    #Now let's plot the original and the translation together.
    ax4.plot(x,c='red',label='Original')
    ax4.scatter(range(len(x)),translations,c='green',label='Translations')
    ax4.legend()

    return sp, enc


def Basic_TM_Example(sp,enc):
    #With the Spatial Pooler trained, let's create a temporal memory.
    
    #Before we get started let's get some axes objects ready for plotting.
    fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(40,10))
    plt.subplots_adjust(hspace=0.5)
    
    
    num_cells = 4   #We'll just use a few cells for this.
    threshold = 4   #The predictive overlap threshold
    at = AnomalyTracker()   #Initialize a default anomaly tracker
    #Leave the other parameters to default
    tm = TemporalMemory(spatial_pooler = sp, anomaly_tracker=at, num_cells=num_cells, stimulus_thresh = threshold)
    
    #Let's train it on a sine wave again, but we'll add some anomalies this time.
    t = np.arange(0,(25)*2*np.pi, 2*np.pi/40)
    x = 500 + np.sin(t)*500
    
    #Replace the last 200 data points with a region with 3X the frequency
    x[-200:] = 500 + np.sin(3*t[-200:])*500

    #Add some random jumps
    x[400] = 0
    x[401] = 1000
    x[500] = 750
    x[501] = 250
    
    #Let's visualize the raw sine wave data with the anomalies.
    ax1.plot(x)
    ax1.set_title('Raw Anomalous Data')
    
    encoded_x = [enc.encode(val) for val in x]
    
    #We'll record all of the active and predictive cell SDRs for future use.
    actives = []
    preds = []
    
    for index, encoding in enumerate(encoded_x):
        if index % 100 == 0:
            print("Processed input {} of 1000...".format(index))
        act, pred = tm.process_input(encoding)
        actives.append(act)
        preds.append(pred)
        
    #Let's use the active cell outputs to learn the translation.
    reg = Regressor(actives,x)
    
    #Now we'll translate the predictions and plot them against the actual output.
    translation = reg.translate(preds)
    ax2.plot(x,c='red',label='Real Data')
    ax2.plot(translation,c='green',label='Translations')
    ax2.legend()
    
    #We can also plot the anomaly scores!
    ax3.plot(at.scores)
    ax3.set_title('Anomaly Scores')
    
    return tm, at


if __name__ == '__main__':
    sp, enc = Basic_SP_Example()
    tm, at = Basic_TM_Example(sp, enc)




