# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:31:49 2020

@author: Carlton

This file demonstrates how to train a SpatialPooler with connections to another SP
as well as its dedicated encoder.
The connected pooler learns to activate even without proximal activity from its encoder.
In this example, The first pooler learns to represent simple scalar values and the second 
pooler learns translate between x and sin(x) for x in [0,pi]

In this script we will:

-Define two different encoders

-Instantiate the two Spatial Poolers and train them simultaneously.
    
    -Note that during training, the second spatial pooler will have its
    'process_input' method called twice--once for the proximal input from its
    encoder, and once for the input from its sibling pooler.
    
-Train two regressors, one for each pooler.

-Run a new dataset through a encoder/pooler pair, and send the results
to the *other* pooler.

-Plot the translated output of the second pooler vs. the ideal output.
"""
from PyHTM import *
import random as rand
import matplotlib.pyplot as plt

### Instantiate the two encoders

#The first encoder will be a simple scalar encoder.
enc1 = ScalarEncoder(n=1000, w=40, minval=0, maxval=np.pi)

#The second will be an RDSEncoder.
enc2 = RDSEncoder(n=800, w=32, start = 0, res=0.001)

### Instantiate and train the SpatialPoolers

#First instantiate a basic pooler connected to enc1
sp1 = SpatialPooler(source=enc1, column_num = 1200)

#Now make a second pooler connected to enc2
sp2 = SpatialPooler(source=enc2, column_num=1000)

#Connect sp2 to sp1
sp2.connect(sp1)

### Train the poolers in tandem

#Generate many values between 0 and pi
data = [np.pi*rand.random() for i in range(3000)]

#Now train the poolers
sp1_training_outputs = []
sp2_training_outputs = []
for i in range(3000):
    if (i+1) % 100 == 0 and i > 0:
        print("Trained on {} out of 3000 inputs...".format(i+1))
    #Process sp1's input normally
    sp1_in = enc1.encode(data[i])
    sp1_out = sp1.process_input(sp1_in)
    
    #Collect both enc2's SDR and sp1's output SDR for processing by sp2.
    sp2_in = [enc2.encode(np.sin(data[i])), sp1_out]
    sp2_out = sp2.process_multiple_inputs(sp2_in)
    
    sp1_training_outputs.append(sp1_out)
    sp2_training_outputs.append(sp2_out)
    
### Train regressors

#Only use the last 1000 inputs, after the SPs have settled down.
print("Training regressor on SP1...")
reg1 = Regressor(sp1_training_outputs[-1000:],data[-1000:])
print("Training regressor on SP2...")
reg2 = Regressor(sp2_training_outputs[-1000:],np.sin(data[-1000:]))

### Run another dataset for testing
#The dataflow will look like X -> enc1 -> sp1 -> sp2 -> reg2

test_data = [np.pi*i/1000 for i in range(1000)]
sp1_testing_outputs = []
sp2_testing_outputs = []
sp2_testing_outputs_comp = []
true_y = np.sin(test_data)
for i, x in enumerate(test_data):
    if (i+1) % 100 == 0:
        print("Tested {} inputs out of 1000...".format(i+1))
    #For this part we'll turn off , learning and duty cycle updating
    #Generate SDRs from sp1 corresponding to enc1(x)
    sp1_out = sp1.process_input(enc1.encode(x), boosting=False, sp_learning=False)
    
    #Generate SDRs from sp2 corresponding to only the sp2 output
    sp2_out = sp2.process_input(sp1_out,boosting = False, sp_learning=False, input_ID = sp1.ID)
    
    #Generate SDRs from sp2 corresponding to the proximal connections to enc2(sin(x))
    sp2_testing_outputs_comp.append(sp2.process_input(enc2.encode(np.sin(x)), sp_learning=False, boosting=False, new_cycle=False))
    
    sp1_testing_outputs.append(sp1_out)
    sp2_testing_outputs.append(sp2_out)
    
#Now let's plot the data again, just like before.
fig, (ax1, ax2, ax3) = plt.subplots(3,1)
plt.subplots_adjust(hspace=1.0)

ax1.plot(test_data,c='red',label='Testing Data')
ax1.scatter(range(1000),reg1.translate(sp1_testing_outputs), c='blue',label='Translations')
ax1.set_title('SP1 Data')
ax1.legend()

ax2.plot(np.sin(test_data),c='red',label='Testing Data')
ax2.scatter(range(1000),reg2.translate(sp2_testing_outputs_comp), c='blue',label='Translated Data')
ax2.set_title('SP2 Encoder-Input Results')
#ax2.legend()

ax3.plot(np.sin(test_data),c='red',label='Testing Data')
ax3.scatter(range(1000),reg2.translate(sp2_testing_outputs),c='blue',label='Translated Data')
ax3.set_title('SP2 SP-Input Results')
#ax3.legend()
    
    






