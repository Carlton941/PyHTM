# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:19:40 2020

@author: Carlton

In this script I will test the ability of my spatial poolers to associate
images with other inputs from different sources.

I will first train an SP with two encoders, and see if it can reconstruct
inputs from one encoder based only on a corresponding input from the other.
The idea of this test is that the SP should learn a representation that is 
invariant to the input domain.

Then I will generalize this test by training one SP on each encoder, and a 
third SP on the output from both SPs. This is essentially the same test, but 
demonstrates a broader applicability if it works.

In the first function:
    
-I will use connect() to join an SP to two encoders:
    an image binarizer and a ten-class scalar encoder.

-After training simultaneously with both encoders, I will provide input from
    the class encoder and use a Reconstructor to turn that output into an image.
    
In the second function:
    
-Do the same thing, but generalize a bit by putting two more poolers between the
'master' pooler and the two encoders.

In the third function:
    
-Take the 'master' pooler from the first function and wrap a TM around it.

-See if the TM can learn sequences invariant to input type.

"""

from PyHTM import *
from keras.datasets import mnist
import random as rand

(images1, labels1), (images2, labels2) = mnist.load_data()

#Define the image binarizer and scalar class encoder
ib = ImageBinarizer(images1[0])
se = ScalarEncoder(n=400, w=40, minval=0, maxval=9)

def Test1():
    #Train one SP on two encoders
    sp = SpatialPooler(source = ib, boost_str = 1)
    sp.connect(se)
    
    ib_encs = []
    se_encs = []
    sp_outs = []
    
    for i in range(10000):
        if (i+1) % 100 == 0:
            print("Processed {} out of 10000 inputs...".format(i+1))
        #Train the sp on both encodings.
        ib_enc = ib.encode(images1[i])
        se_enc = se.encode(labels1[i])
        
        ib_encs.append(ib_enc)
        se_encs.append(se_enc)
        
        sp_out = sp.process_multiple_inputs([ib_enc,se_enc])
        sp_outs.append(sp_out)
        
    #Train a reconstructor on the image data
    ib_rec = Reconstructor(sp_outs[5000:10000],images1[5000:10000])
    
    #Train a reconstructor on the scalar encodings (rather than simply the numerical values)
    se_rec = Reconstructor(sp_outs[5000:10000],se_encs[5000:10000])
    
    ### Now we'll test a few single-inputs.
    
    fig, axes = plt.subplots(2,2)
    ((ax1, ax2), (ax3, ax4)) = axes
    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(wspace=0.5)
    
    #Give the SP one scalar encoding
    sp_out1 = sp.process_input(se.encode(labels1[10001]),boosting=False,sp_learning=False, input_source_num=1)
    sp_out2 = sp.process_input(se.encode(labels1[10002]),boosting=False,sp_learning=False, input_source_num=1)

    #Try to turn this scalar value into an image
    img1 = ib_rec.translate(sp_out1)
    img2 = ib_rec.translate(sp_out2)
    
    ax1.imshow(img1)
    ax1.set_title("Image of {} reconstructed\nfrom sccalar input.".format(labels1[10001]))
    ax2.imshow(img2)
    ax2.set_title("Image of {} reconstructed\nfrom scalar input.".format(labels1[10002]))
    
    #Give the SP one image encoding
    sp_out3 = sp.process_input(ib.encode(images1[10001]),boosting=False,sp_learning=False)
    sp_out4 = sp.process_input(ib.encode(images1[10002]),boosting=False,sp_learning=False)
    
    img3 = se_rec.translate(sp_out3)
    img4 = se_rec.translate(sp_out4)
    
    plot_SDR(img3, ax3)
    plot_SDR(img4, ax4)
    ax3.set_title("Scalar {} encoding reconstructed\nfrom image input.".format(labels1[10001]))
    ax4.set_title("Scalar {} encoding reconstructed\nfrom image input".format(labels1[10002]))
    
    for ax in axes.reshape(-1,):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
    return sp
        
def Test2():
    #Train three SPs on two encoders.
    #Practically, this is redundant and you only need Test1.
    
    #Make the first two single-input poolers
    ib_sp = SpatialPooler(source=ib, boost_str=1, ID='IB_SP')
    se_sp = SpatialPooler(source=se, boost_str=1, ID='SE_SP')
    
    #Make the second-layer pooler that reads both of the other poolers.
    sp = SpatialPooler(boost_str=1)
    sp.connect(ib_sp)
    sp.connect(se_sp)
    
    ib_encs = []
    se_encs = []
    
    ib_sp_outs = []
    se_sp_outs = []
    
    sp_outs = []
    
    #Train all three together
    for i in range(10000):
        if (i+1) % 100 == 0:
            print("Processed {} out of 10000 inputs...".format(i+1))
        #Encode the two different inputs
        ib_enc = ib.encode(images1[i])
        se_enc = se.encode(labels1[i])
        
        #Process the two inputs through the single-poolers
        ib_sp_out = ib_sp.process_input(ib_enc)
        se_sp_out = se_sp.process_input(se_enc)
        
        #Process the single-pooler outputs through the master pooler
        sp_in = [ib_sp_out,se_sp_out]
        sp_out = sp.process_multiple_inputs(sp_in)
        
        ib_encs.append(ib_enc)
        se_encs.append(se_enc)
        ib_sp_outs.append(ib_sp_out)
        se_sp_outs.append(se_sp_out)
        sp_outs.append(sp_out)
        
    img_rec = Reconstructor(sp_outs[5000:10000],images1[5000:10000])
    classifier = Classifier(sp_outs[5000:10000],labels1[5000:10000])
    
    ###Now, the moment of truth. We will try feeding one or the other of the base SPs and see
    #if the master SP can generate the other's output.
    
    #Generate a couple of new binarized images
    ib_enc1 = ib.encode(images1[10003])
    ib_enc2 = ib.encode(images1[10005])
    
    #Process the images through the lower SP
    ib_sp_out1 = ib_sp.process_input(ib_enc1,boosting=False,sp_learning=False)
    ib_sp_out2 = ib_sp.process_input(ib_enc2,boosting=False,sp_learning=False)
    
    #Run just the one SP input through the master SP
    sp_out1 = sp.process_input(ib_sp_out1,boosting=False,sp_learning=False,input_source_num=0)
    sp_out2 = sp.process_input(ib_sp_out2,boosting=False,sp_learning=False,input_source_num=0)
    
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    
    ax1.imshow(img_rec.translate(sp_out1))
    ax2.imshow(img_rec.translate(sp_out2))
    ax1.set_title('This should be {}'.format(labels1[10003]));
    ax2.set_title('This should be {}'.format(labels1[10005]));
    
    return ib_sp, se_sp, sp

def Test3(sp):
    #This test trains a TM wrapped around the SP and gives
    #it a couple of sequences. 
    
    #First let's make a collection of images sorted by digit.
    imgs = [[] for i in range(10)]
    for index, img in enumerate(images1[:100]):
        imgs[labels1[index]].append(img)
        
    #Define a pair of sequences
    seq1 = [0, 2, 4, 6, 8]
    seq2 = [0,1,2,3,4,5]
    # seq2 = seq1
    
    #Train the TM on the sequences, showing both inputs
    at = AnomalyTracker()
    tm = TemporalMemory(spatial_pooler = sp, anomaly_tracker = at, num_cells=4, stimulus_thresh=4)
    
    #Store all the data in case we want to use it
    sequences = []
    actives = []
    preds = []
    
    #Also specifically track the images and corresponding active-cell outputs 
    #so we can train a reconstructor
    training_imgs = []
    training_actives = []
    
    for iter in range(200):
        #Reset the tm.
        tm.reset()
        
        #Pick a sequence
        if rand.random() <= 0.5:
            seq = seq1
            sequences.append(1)
        else:
            seq = seq2
            sequences.append(2)
            
        #Process each input in the sequence
        for digit in seq:
            #Randomly select one of the images of the appropriate class
            image = imgs[digit][rand.randint(0,len(imgs[digit])-1)]
            ib_enc = ib.encode(image)
            se_enc = se.encode(digit)
            
            #Append the training image data
            training_imgs.append(image)
            
            #Pass the ib_sp output to the TM
            active, pred = tm.process_input([ib_enc, se_enc], multi_inputs = True, static_sp = True)
            actives.append(active)
            preds.append(pred)
            
            #append the active cell data for training
            training_actives.append(active)
                
    #Train a reconstructor
    cell_rec = Reconstructor(training_actives, training_imgs)
    
    ### Now, the big test!
    
    #Let's feed it a sequence using only scalar data and watch the 
    #predicted image
    tm.reset()
    seq = seq1
    predictions = []
    for digit in seq:
        se_enc = se.encode(digit)
        active, pred = tm.process_input(se_enc,input_source_num=1, multi_inputs = False, static_sp = True, tm_learning=False)
        predictions.append(pred)
        
    #Make a list of reconstructions
    rec_imgs = cell_rec.translate(predictions)
    
    #Plot the images that the TM predicted based on the scalar inputs
    #For example, if the input is sequence 1, then when 2 is given it should
    #predict a 4, and when 4 is given it should predict a 6 etc.
    fig, axes = plt.subplots(1,len(seq)-1)
    for index, ax in enumerate(axes):
        ax.imshow(rec_imgs[index])
        ax.set_title('Input: {}'.format(seq[index]));
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
    return tm, at
        
if __name__ == '__main__':
    sp = Test1()
    tm, at = Test3(sp)
    pass
