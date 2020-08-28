# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:37:04 2020

@author: Carlton

Experimental vector-to-SDR encoder for GloVE vector objects.
Requires a dictionary that maps from word strings to GloVe vectors.
You can get the necessary data at the glove reference link below. The files are too large
to upload to github individually.
"""
import numpy as np
class GloVeEncoder():
    #This object encodes GloVe word-vectors as SDRs. GloVe reference: (https://nlp.stanford.edu/projects/glove/)
    #You can use pickle to load the GloVeDict.txt dictionary file which contains
    #400,000 word-vectors indexed to their associated word.
    def __init__(self, d, b = 20, w = 40, ID = 'GE1'):
        #Constructor method.
        #filepath -> path to the file containing the GloVe word vectors.
        #b -> Half of the number of bits assigned to each vector dimension.
            #Each half corresponds to -/+ values , respectively.
        #Number of on-bits for the encoding.
        
        #Assign the basic variables
        self.d = d
        self.b = b
        self.w = w
        self.n = 2*b*d
        self.ID = ID
        self.output_dim = (self.n,)
        
    def test_similarity(self, key1, key2, dictionary, weighting = 'exp', beta = 3):
        #Tests the % similarity of two encodings corresponding to two words.
        sdr1 = self.encode(key1, dictionary, weighting=weighting, beta=beta)
        sdr2 = self.encode(key2, dictionary, weighting=weighting, beta=beta)
        #Normalize by the sum of the second sdr, and multiply by 100 to transform to %.
        return round(100/np.sum(sdr2)*overlap(sdr1,sdr2),1)
    
    def encode_difference(self, key1, key2, dictionary, weighting = 'exp', beta = 3, enforce_w = False):
        #Encodes the vector difference between two word normalized word vectors.
        
        #Get the word vectors
        vec1 = np.array(dictionary[key1])
        vec2 = np.array(dictionary[key2])
        
        #Normalize the vectors
        # vec1 /= np.linalg.norm(vec1)
        # vec2 /= np.linalg.norm(vec2)
        
        #Call the encode() method
        return self.encode(dictionary=dictionary, vector=vec2-vec1, weighting=weighting, beta=beta,enforce_w=enforce_w)
        
    def encode(self,  dictionary, key = None, vector = None, weighting = 'exp', enforce_w = False, beta = 3):
        #Encodes a GloVe word-vector as an SDR. Can either take the vector or the dictionary and key.
        #Each dimension is assigned 2*self.b bits, half for positive and half for negative.
        #Then some of the bits for each dimension are turned on, according to the relative
        #magnitude of the value in that position, until self.w bits have been activated.
        
        #Take the input as a numpy array
        if vector is None:
            vector = np.array(dictionary[key]).reshape((-1,))
        elif type(vector) is not np.ndarray:
            vector = np.array(vector)
            
        #Normalize the vector
        # vector = vector / np.linalg.norm(vector)
            
        
        #Get the total weight of the vector by summing either the absolute values
        #or the squares of the elements.
        if weighting == 'abs':
            weights = np.abs(vector)
            total = np.sum(weights)
        elif weighting == 'squares':
            weights = np.square(vector)
            total = 1
        elif weighting == 'exp':
            weights = np.abs(vector)**beta
            total = np.sum(weights)
        else:
            print('Invalid weighting argument. Defaulting to squares.')
            weights = np.square(vector)
            total = 1
            
        #Set the number of active bits for each zone by multipling the weight by w/total.
        num_active_bits = np.round(self.w/total*weights).astype('int')
                    
        #Make sure there isn't a single zone with more than self.b active bits.
        sorted_bits = np.argsort(num_active_bits)
        overflow = 0
        for index in range(self.d - 1, -1, -1):
            bits = num_active_bits[sorted_bits[index]]
            if bits > self.b:
                #Increment the overflow, and scale back the active bits to self.b
                overflow += bits - self.b
                num_active_bits[sorted_bits[index]] = self.b
            elif bits < self.b:
                #Use up some of the overflow, and scale up the active bits
                num_active_bits[sorted_bits[index]] += overflow
                overflow -= min(self.b, bits + overflow) - bits
        
        #Make sure the rounding process didn't produce an incorrect bit total.
        #We can accomplish this by scanning through all the zones with active bits,
        #starting at the highest one, and adding/subtracting one bit to/from each.
        if enforce_w:
            increment_index = len(sorted_bits)
        
            #While the total number of active bits is less than w:
            while np.sum(num_active_bits) != self.w:
                increment_index -= 1
                
                #Start the cycle over again if the index passed 0
                if increment_index < 0:
                    increment_index = len(sorted_bits) - 1
                    
                #Skip this index if the maximum bits are already contained here.
                if num_active_bits[sorted_bits[increment_index]] == self.b:
                    continue
                
                #Update the bit counter for this index
                num_active_bits[sorted_bits[increment_index]] += np.sign(self.w-np.sum(num_active_bits))        
                        
        #Now we'll define the output SDR and fill in the active bits for each zone.
        SDR = np.zeros(self.n,)
        for index, bits in enumerate(num_active_bits):
            if bits > self.b:
                print("Error! Bits = {} > {}".format(bits,self.b))
            if vector[index] < 0:
                #Activate the bits at index*(2*b)
                start = index*2*self.b
                end = index*2*self.b + bits
                SDR[start:end] = 1
                                
            else:
                #Activate the bits at index*(2*b) + b
                start = (1 + index*2)*self.b
                end = (1 + index*2)*self.b + bits
                SDR[start:end] = 1          
                
        # if np.sum(SDR) != self.w:
        #     print("Error compiling SDR: Total active bits = {} and not {}.".format(np.sum(SDR),self.w))
                
        return SDR
            
def overlap(sdr1, sdr2):
    #Returns the overlap score of two SDRs with matching shapes.
    return np.sum(sdr1*sdr2)
