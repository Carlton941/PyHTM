# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:31:49 2020

@author: Carlton

This file demonstrates how to train a pair of Spatial Poolers with apical connections
so that the second pooler learns to activate even without proximal activity.
In this example, The first pooler learns to represent simple scalar values and the second 
pooler learns translate between x and sin(x) for x in [0,pi]

In this script we will:

-Define two different encoders

-Instantiate the two Spatial Poolers and train them simultaneously.
    
    -Note that during training, the second spatial pooler will have its
    'process_input' method called twice--once for the proximal input from its
    encoder, and once for the apical input from its sibling pooler.
    
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

#First instantiate a basic pooler with no apical connections
sp1 = SpatialPooler(input_dim=(enc1.n,), column_num = 1200, max_active_cols=50)

#Now make a pooler with apical connections to sp1.
ap_threshold = 20   #Any column with this many or more active apical connections will activate, regardless of proximal inputs.
sp2 = SpatialPooler(input_dim=(enc2.n,), column_num=1000, apical_input_dim=(1200,), apical_stim_thresh = ap_threshold)

### Train the poolers in tandem

#Generate many values between 0 and pi
data = [np.pi*rand.random() for i in range(3000)]

#Now train the poolers
sp1_training_outputs = []
sp2_training_outputs = []
for i in range(3000):
    if i % 100 == 0 and i > 0:
        print("Trained on {} out of 3000 inputs...".format(i))
    #Process sp1's input normally
    sp1_training_outputs.append(sp1.process_input(enc1.encode(data[i])))
    
    #Process sp2's proximal input normally, but don't update the duty cycle
    sp2.process_input(enc2.encode(np.sin(data[i])), apical=False, update_duty_cycle = False)
    
    #Process sp2's apical input and update the duty cycle
    sp2_training_outputs.append(sp2.process_input(sp1_training_outputs[i], apical=True, update_duty_cycle=True))
    
### Train regressors

#Only use the last 1000 inputs, after the SPs have settled down.
reg1 = Regressor(sp1_training_outputs[-1000:],data[-1000:])
reg2 = Regressor(sp2_training_outputs[-1000:],np.sin(data[-1000:]))

### Run another dataset for testing
#The dataflow will look like X -> enc1 -> sp1 -> sp2 -> reg2

test_data = [np.pi*i/1000 for i in range(1000)]
sp1_testing_outputs = []
sp2_testing_outputs = []
sp2_testing_outputs_comp = []
true_y = np.sin(test_data)
for i, x in enumerate(test_data):
    if i % 100 == 0 and i > 0:
        print("Tested {} inputs out of 1000...".format(i))
    #For this part we'll turn off , learning and duty cycle updating
    #Generate SDRs from sp1 corresponding to enc1(x)
    sp1_testing_outputs.append(sp1.process_input(enc1.encode(x), boosting=False, sp_learning=False))
    
    #Generate SDRs from sp2 corresponding to the apical connections to sp1
    sp2_testing_outputs.append(sp2.process_input(sp1_testing_outputs[-1], reset=True, apical=True, boosting=False, update_duty_cycle=False, sp_learning=False))
    
    #Generate SDRs from sp2 corresponding to the proximal connections to enc2(sin(x))
    sp2_testing_outputs_comp.append(sp2.process_input(enc2.encode(np.sin(x)),apical=False,sp_learning=False,boosting=False))
    
#Now let's plot the data again, just like before.
fig, (ax1, ax2, ax3) = plt.subplots(3,1)
plt.subplots_adjust(hspace=1.0)

ax1.plot(test_data,c='red',label='Testing Data')
ax1.scatter(range(1000),reg1.translate(sp1_testing_outputs), c='blue',label='Translations')
ax1.set_title('SP1 Data')
ax1.legend()

ax2.plot(np.sin(test_data),c='red',label='Testing Data')
ax2.scatter(range(1000),reg2.translate(sp2_testing_outputs_comp), c='blue',label='Translated Data')
ax2.set_title('SP2 Proximal Results')
#ax2.legend()

ax3.plot(np.sin(test_data),c='red',label='Testing Data')
ax3.scatter(range(1000),reg2.translate(sp2_testing_outputs),c='blue',label='Translated Data')
ax3.set_title('SP2 Apical Results')
#ax3.legend()
    
    






