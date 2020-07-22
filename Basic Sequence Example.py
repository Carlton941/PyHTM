# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 12:28:46 2020

This is a very basic sequence example with no noise, showing that the TM
can learn simple patterns like A B C D and A B D C.

In this script we encode four scalar values and train an SP to represent them.

(Theoretically it's not even necessary to train an SP, we could just send the
 encodings directly to the TM. But I digress.)

Then we show four sequences of inputs to the TM.
1) A B C D
2) A B C D
3) A B D C
4) A B D C

Then we plot the anomaly scores and the number of predictions made at each step.

The comments at the end help to interpret the plots.

@author: Carlton
"""

import random as rand
import numpy as np
import math
import matplotlib.pyplot as plt
import random as rand
from PyHTM import *


#Define the encoder
enc = ScalarEncoder(n=1000,w=40,minval=0,maxval=1)

#Define the SP
sp = SpatialPooler(source = enc, boost_str=1)
# sp = SpatialPooler(input_dim = (enc.n,), boost_str=1)

#Train the SP to recognize 4 different values
A = enc.encode(0.1)
B = enc.encode(0.3)
C = enc.encode(0.9)
D = enc.encode(0.5)
inputs = [A, B, C, D]
for i, encoding in enumerate(inputs*30):
    sp.process_input(encoding)
    
#Instantiate the TM
at = AnomalyTracker()
tm = TemporalMemory(spatial_pooler = sp, anomaly_tracker = at, num_cells=4,stimulus_thresh=4)

#Run the A B C D sequence once, then reset.
#Resetting tells the TM that the sequence is ended.
tm.process_input(A)
tm.process_input(B)
tm.process_input(C)
tm.process_input(D)
tm.reset()

#The TM should now have learned the pattern A B C D
#Let's run it again to see if it learned. (We will make some figures later)
tm.process_input(A)
tm.process_input(B)
tm.process_input(C)
tm.process_input(D)
tm.reset()

#Now let's throw a wrench into the mix by introducing the new pattern A B D C
tm.process_input(A)
tm.process_input(B)
tm.process_input(D)
tm.process_input(C)
tm.reset()

#Now that the system has learned TWO patterns, let's run one of the patterns
#one last time, and then we'll plot some data
tm.process_input(A)
tm.process_input(B)
tm.process_input(D)
tm.process_input(C)
tm.reset()

#Now we'll plot some results.
fig, (ax1,ax2) = plt.subplots(2,1)
plt.subplots_adjust(hspace=0.5)

#First, let's plot the anomaly scores.
#This measures the ratio of *correctly* predictive cells to total active cells.
#If it's 0, the input was a complete surprise (or the TM is still learning the sequence)
#If it's 1, the input was completely expected.
#It can also be in between 0 and 1, but for this simple pattern it shouldn't be.
ax1.scatter(range(16),at.scores,c='red')
ax1.set_title('Anomaly Data')

#Let's also plot the ratio of total predictive cells to total active cells
#This ratio indicates how many *different* predictions the system was making
#before it saw the current input.
ax2.scatter(range(16),at.prediction_num,c='green')
ax2.set_title('Prediction Numbers')

#First, let's examine the anomaly scores.
#Note that every fourth input, starting with the first one, is anomalous.
#That's because we reset at the end of every sequence, so the system is never
#predicting an A.

### The first 4 points
#Now notice that the entire first sequence is anomalous. That's because
#the system has never before seen the pattern A B C D. 

### The next 4 points
#But after the pattern completes the first time, when it sees A B C D for the 
#second time it knows what to expect after A.

### The next 4 points
#Then it sees A for the third time, and expects a B, which is correct.
#However, it then sees D C instead of C D, which is anomalous.

### The last 4 points
#Finally, it sees A B D C a second time, and this time it recognizes the sequence.

#Now let's look at the prediction numbers. This indicates how many different
#values it is predicting. Note again that the system never knows how to predict A.

### The first 4 points
#The system hasn't learned anything so it makes no predictions.

### The next 4 points
#After A, it makes exactly one prediction for B, C and D in order.

### The next 4 points
#After A, it again predicts B and C. But when it sees D instead of C, it makes
#no prediction for the last value because it has never seen anything follow after D.

### The last 4 points
#After seeing the sequence A B D C for the first time in the last set,
#it now makes two predictions after B. It might be C, it might be D!
#Once it has confirmed that D follows after B this time, it remembers
#from last time that C should follow B so it makes one prediction.
    


