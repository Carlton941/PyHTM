# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:51:40 2020

@author: Carlton94 (Andrew Stephan)

This library implements the basic functionalities of HTM theory in object-oriented
Python 3. The following objects are defined:
    -ScalarEncoders, which encode a scalar value as an SDR using sequential bits.
    
    -SeedEncoders, which are similar to RDSE but use seed-based RNG to avoid maintaining a list of indices.
    
    -RDSEncoders, which encode scalar values with non-sequential bits using a sliding window over a stored list of randomly-generated indices.
    
    -DateEncoders, which encode a datetime object as an SDR.
    
    -MultiEncoders, which contain multiple encoders.
    
    -MiniColumns, which maintain several numpy arrays containing information about
        synaptic permanences and true connections to an input space.
        
    -SpatialPoolers, which curate lists of miniColumns by feeding them SDR
        inputs from encoders and training their permanences.
    
    -Segments, which are the basic element of Temporal Memory. Llike MiniColumns,
        they maintain numpy arrays containing synaptic permanence and connection info.
        
    -Cells, which curate lists of Segments and can compare a given input SDR
        with the connections in their Segments and report the highest overlap score.
        
    -TemporalMemory, which curates a list of Cells and a SpatialPooler. Encoded
        SDR inputs can be sent to the SP for processing, then passed to the Cells
        to determine cell activity and prediction as well as learning updates.   
        
        
This is a work-in-progress, so many advanced functionalities (such as RDSE) are not implemented yet.

    
"""
from scipy import sparse
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import random as rand
import warnings

class ScalarEncoder():
    #This is a simple 1D scalar value encoder.
    
    def __init__(self, n, w, minval, maxval, wrap = False):
        #Constructor method. Needs basic information about the encoding width w,
        #the encoding length n, the min and maximum allowed values, and whether 
        #the encoding should wrap-around.
        
        if w > n:
            raise ValueError('w cannot exceed n!')
        if minval > maxval:
            raise ValueError('minval cannot exceed maxval!')
        
        self.boost_factor = 1
        self.w = w      #Number of on-bits, I.E. encoding width
        self.minval = minval
        self.maxval = maxval
        self.n = n      #Total length of bit array
        self.wrap = wrap
        if self.wrap:
            #There are n bins.
            self.num_bins = n
        else:
            #There are n - (w - 1) bins
            self.num_bins = n - w + 1
            
        #Record the numeric width of each bin, I.E. the resolution.
        #An input may change by up to this value without changing any bits in
        #the resulting encoding.
        self.bin_length = (maxval - minval)/self.num_bins
        
        #Record the max overlap distance. This is the maximum value by which
        #two inputs can differ and still register at least one bit of overlap.
        self.max_overlap_distance = self.w*self.bin_length
    
    def encode(self, val):
        #Encodes a single numeric value. Assumes min/max cutoff.
        
        #Cut off the input val at minval or maxval.
        x = max(self.minval,min(self.maxval, val))
        
        #Define a zeros array
        arr = np.zeros((self.n,))

        #If the encoder is periodic:
        if self.wrap:
            #Find which bin x belongs in.
            bin_index = math.floor((x - self.minval)/self.bin_length)
            
            #Check to see if the encoding will need to wrap around
            end_index = bin_index + self.w
            if end_index > self.n:
                wrap_remainder = end_index - self.n
                arr[bin_index:] = 1
                arr[:wrap_remainder] = 1
                
            else:
                arr[bin_index:bin_index + self.w] = 1
                    
        #If the encoder is nonperiodic:
        else:
            #Find which bin x belongs in.
            bin_index = math.floor((x - self.minval)/self.bin_length)
            arr[bin_index:bin_index + self.w] = 1
        
        return arr
    
class SeedEncoder():
    #Experimental version of the RDSEncoder that doesn't keep a running list of indices.
    #Instead uses random seeds for repeatability.
    #Note that, unlike the RDSE, this encoder has a minimum input value and cuts off anything below that.
    #Note that this encoder does *not* check to ensure that, within each bin,
    #no indices are duplicated. This is unlikely to be an issue if n and w are sufficiently large.
    #If there are duplicates, no errors will be thrown--the SDR will simply be a bit more sparse.
    
    def __init__(self, n, w, minval = 0, res = 1):
        #Constructor method.
        #n->encoding length
        #w->encoding width
        #res->the amount of input space covered by one encoding--i.e. encoding resolution
        self.n = n
        self.w = w
        self.minval = minval
        self.res = res
        
    def encode(self, val, seed = 0, printout = False):
        #Encodes a value as an SDR.
        
        #Cut off the input val at the minimum
        val = max(val,self.minval)

        #Set the seed.
        rand.seed(seed)
        
        #Calculate how many bins to the right
        bin_shift = int(np.floor(val - self.minval)/self.res)
        
        #Unless bin_shift is 0, we will need to dump a few numbers to advance the RNG state.
        for i in range(bin_shift):
            rand.randint(0,self.n-1)
            bin_shift -= 1
        
        index_list = [rand.randint(0,self.n-1) for i in range(self.w)]
        
        #Generate an array with 1's in all of the index_list locations
        out = np.zeros((self.n,))
        for index in index_list:
            out[index] = 1
            
        #For debugging purposes:
        if printout:
            print("Encoding val: {}. Indices = {}".format(val,index_list))
            
        #Return the array
        return out
       
class RDSEncoder():
    #Random Distributed Scalar Encoder. Similar to the Scalar Encoder,
    #but instead of sequential bits this encoder uses bits randomly distributed
    #around the space. This greatly increases the capacity of a single encoder.
    #This object does not support wrap-around encodings.
    
    def __init__(self, n, w, start = 0, res = 1):
        #Constructor method.
        #n->encoding length
        #w->encoding width
        #res->the amount of input space covered by one encoding--i.e. encoding resolution
        self.n = n
        self.w = w
        self.start = start
        self.res = res
        self.num_bins = 1
        self.start_index = 0 #The index in i_list corresponding to the start bin.
        self.bin_range = [start,start+res] #The range covered by the current bins. Bins are right-exclusive: [start, start + res)
        
        #Initialize the index list with enough unique indices to encode the starting value
        self.i_list = []
        for i in range(self.w):
            new_i = rand.randint(0,self.n-1)
            while new_i in self.i_list:
                new_i = rand.randint(0,self.n-1)
            self.i_list.append(new_i)
                
    def encode(self,val):
        #Encode a value in an SDR. Also handles expansion of the index-list.

        #First, check to see if we've seen a value this far from start before
        if (val >= self.bin_range[0]) and (val < self.bin_range[1]):
            #Determine which bin this input belongs to.
            bin_i = int(self.start_index + np.floor((val - self.start)/self.res))
            
        #If this value is smaller (Or more negative) than any hitherto-seen value, add more bins to the left
        while val < self.bin_range[0]:
            self.add_bin('left')
            bin_i = 0

        #If this value is larger than any hitherto-seen value, add more bins to the right
        while val >= self.bin_range[1]:
            self.add_bin('right')
            #Account for 0-based indexing with -1
            bin_i = self.num_bins - 1

        #Go through the indices in i_list matching the appropriate bin
        out = np.zeros((self.n,))
        for index in self.i_list[bin_i:bin_i + self.w]:
            out[index] = 1
        
        return out
    
    def add_bin(self,where):
        #Adds a new bin by extending the index list either at the end or the beginning.
        #Ensures that, within each bin, no indices are repeated.
        
        #If adding to the left end of the list:
        if where == 'left':
            #Generate a new index that's not shared by the currently left-most bin.
            new_i = rand.randint(0,self.n-1)
            while new_i in self.i_list[:self.w]:
                new_i = rand.randint(0,self.n-1)
                
            #Insert the new index into the left end of the index list
            self.i_list.insert(0,new_i)
            
            #Update the bin range list
            self.bin_range[0] -= self.res
            
            #Update the start index tracker
            self.start_index += 1
            
        #Alternatively, if adding to the right end:
        elif where == 'right':
            #Generate a new index that's not shared by the currently right-most bin.
            new_i = rand.randint(0,self.n-1)
            while new_i in self.i_list[-self.w:]:
                new_i = rand.randint(0,self.n-1)
                
            #Insert the new index to the right end of the list
            self.i_list.append(new_i)
            
            #Update the bin range list
            self.bin_range[1] += self.res
            
        #If there's an error with the 'where' arg:
        else:
            raise ValueError('Argument "where" must equal either "left" or "right"!')
        
        #Update the bin counter
        self.num_bins += 1
                
class DateEncoder():
    #Encodes a date with year, month, weekday, hour and weekend data.
    #Uses a ScalarEncoder for each individual component of the encoding.
    #Each component is optional, and each component that will be used
    #is specified by setting <component> = [n, w] in the call to __init__.
    
    def __init__(self, year = None, month = None, day = None, hour = None, weekend = None):
        #Constructor method. Needs information about which timestamp components to include
        #as well as their length and width.
        
        self.year_enc = None
        self.month_enc = None
        self.day_enc = None
        self.hour_enc = None
        self.weekend_enc = None
        self.n = 0
        self.w = 0
        
        if year:
            self.year_enc = ScalarEncoder(n = year[0], w = year[1], wrap=False)
            self.n += year[0]
            self.w += year[1]
        if month:
            self.month_enc = ScalarEncoder(month[0], month[1], 1, 13, wrap=True)
            self.n += month[0]
            self.w += month[1]
        if day:
            self.day_enc = ScalarEncoder(day[0], day[1], 0, 7, wrap=True)
            self.n += day[0]
            self.w += day[1]
        if hour:
            self.hour_enc = ScalarEncoder(hour[0], hour[1], 0, 24, wrap=True)
            self.n += hour[0]
            self.w += hour[1]
        if weekend:
            self.weekend_enc = ScalarEncoder(weekend[0], weekend[1], 0, 1)
            self.n += weekend[0]
            self.w += weekend[1]

    def encode(self, date):
        #Takes a Datetime date as input, encodes each component separately and concatenates the arrays.
        
        weekend = 0
        if date.weekday() > 4:
            weekend = 1
        
        arr_list = []
        if self.year_enc:
            arr_list.append(self.year_enc.encode(date.year))
        if self.month_enc:
            arr_list.append(self.month_enc.encode(date.month))
        if self.day_enc:
            arr_list.append(self.day_enc.encode(date.weekday()))
        if self.hour_enc:
            arr_list.append(self.hour_enc.encode(date.hour))
        if self.weekend_enc:
            arr_list.append(self.weekend_enc.encode(weekend))
        
        return np.concatenate(arr_list)
    
class MultiEncoder():
    #This object is used to conveniently combine any number of different encoder objects
    #and produces a net encoding containing the concatenated output of each encoder.
    
    def __init__(self, encoders):
        #Constructor method.
        #encoders -> a list of other encoder objects, of any type.
        
        self.encoders = encoders
        self.n = sum([enc.n for enc in encoders])
        self.w = sum([enc.w for enc in encoders])
            
    def encode(self,inputs):
        #Takes in a list of inputs equal in length to the number of stored encoders.
        #Returns their respective encodings, concatenated in order of index.
        if len(inputs) != len(self.encoders):
            raise ValueError('Number of inputs does not match number of encoders!')
        return np.concatenate([self.encoders[i].encode(inputs[i]) for i in range(len(inputs))])            
       
class miniColumn():
    #This object maintains a list of permanences and connections.
    #The Spatial Pooler object contains a list of these miniColumn objects.
    
    def __init__(self, input_dim = (1000,), potential_percent = 0.5, perm_decrement = 0.008, perm_increment = 0.05, perm_thresh = 0.1, duty_cycle_period = 1000, apical_input_dim = None):
        #Constructor method.
        #input_dim -> Expected dimensions of the input space.
        #potential_percent -> The fraction of bits in the input space to which
        #this miniColumn *may* grow connections.
        #perm_decrement -> Amount by which the permanence to an inactive bit will decrease.
        #perm_increment -> Amount by which the permanence to an active bit will increase.
        #perm_thresh -> Threshold over which a connected synapse will form.
        #duty_cycle_period -> Number of recent inputs used to compute the duty cycle.
        
        #Record the numeric parameters
        self.input_dim = input_dim
        self.potential_percent = potential_percent
        self.perm_decrement = perm_decrement
        self.perm_increment = perm_increment
        self.perm_thresh = perm_thresh
        self.duty_cycle_period = duty_cycle_period
        self.apical_input_dim = apical_input_dim
        
        #Initialize the column connections.
        #The potential connections will be stored in an array of size input_dim with 1's to indicate possible connections.
        self.potential_connections = np.random.choice([1,0], size=input_dim, p=[potential_percent, 1-potential_percent])
        
        #The permanences will be initialized using a normal distribution centered on perm_thresh, multiplied pointwise by potential_connections.
        self.perms = np.random.normal(loc=perm_thresh, scale=perm_increment, size=input_dim)*self.potential_connections
        
        #The actual connections will be stored in an array like potential_connections
        self.actual_connections = (self.perms >= perm_thresh)
        
        #If apical inputs exist, add similar arrays to account for them as well
        if apical_input_dim is not None:
            self.potential_apical_connections = np.random.choice([1,0], size=apical_input_dim, p = [potential_percent, 1-potential_percent])
            self.apical_perms = np.random.normal(loc=perm_thresh, scale=perm_increment, size=apical_input_dim)*self.potential_apical_connections
            self.actual_apical_connections = (self.apical_perms >= perm_thresh)
        
        #Initialize the duty cycle tracker
        self.duty_tracker = [0]*int(duty_cycle_period)
        
    def get_overlap_score(self, arr, apical = False):
        #Returns the overlap score between the actual connections and an array of active synapses.
        if not apical:
            return np.sum(arr*self.actual_connections)
        else:
            return np.sum(arr*self.actual_apical_connections)
    
    def update_perms(self, arr, apical = False):
        #Increments the permanence values for active synapses.
        #Decrements the permanence values for inactive synapses.
        #Updates the actual_connections array.
        if not apical:
            self.perms[(arr > 0)] += self.perm_increment
            self.perms[(arr < 1)] -= self.perm_decrement
            self.actual_connections = self.potential_connections*(self.perms >= self.perm_thresh)
        else:
            self.apical_perms[(arr > 0)] += self.perm_increment
            self.apical_perms[(arr < 1)] -= self.perm_decrement
            self.actual_apical_connections = self.potential_apical_connections*(self.apical_perms >= self.perm_thresh)
    
    def low_duty_cycle_inc(self, apical = False):
        #Increments all permanence values to promote an increased duty cycle.
        if not apical:
            self.perms += self.perm_increment
            self.actual_connections = self.potential_connections*(self.perms >= self.perm_thresh)
        else:
            self.apical_perms += self.perm_increment
            self.actual_apical_connections = self.potential_apical_connections*(self.apical_perms >= self.perm_thresh)
        
    def duty_cycle_update(self,activity):
        #Updates the duty cycle tracker.
        self.duty_tracker.insert(0,activity) #Insert the newest activity number to the head of the list
        self.duty_tracker.pop(int(self.duty_cycle_period)) #Pop off the oldest activity number 
        
    def get_duty_cycle(self):
        #Returns the number of times within the last duty cycle period that this minicolumn activated.
        return np.sum(self.duty_tracker)
    
class SpatialPooler():
    #The SpatialPooler object curates a list of minicolumn, providing inputs and gathering outputs.
    #The pooler can maintain connections to external proximal inputs as well as apical connections to
    #the minicolumns of another pooler. In this case, 
    
    def __init__(self, input_dim = (1000,), column_num = 1000, potential_percent = 0.85, max_active_cols = 40, stimulus_thresh = 0, perm_decrement = 0.005, perm_increment = 0.04, perm_thresh = 0.1, min_duty_cycle = 0.001, duty_cycle_period = 100, boost_str = 3, apical_input_dim = None, apical_stim_thresh = 10):
        #Constructor method.
        #input_dim -> Expected dimensions of the input space.
        #column_num -> Number of minicolumn to be used.
        #potential_percent -> The fraction of bits in the input space to which
        #this miniColumn *may* grow connections.
        #max_active_cols -> Number of allowed minicolumn activations per input processed
        #stimulus_thresh -> Threshold of overlap for a minicolumn to be eligible for action.
        #perm_decrement -> Amount by which the permanence to an inactive bit will decrease.
        #perm_increment -> Amount by which the permanence to an active bit will increase.
        #perm_thresh -> Threshold over which a connected synapse will form.
        #min_duty_cycle -> A minicolumn with duty cycle below this value will be encouraged to be more active.
        #duty_cycle_period -> Number of recent inputs used to compute the duty cycle.
        #boost_str -> Strength of the boosting effect used to enhance the overlap score of low-duty-cycle minicolumn.
        #apical_input_dim -> shape of apical inputs from a neighboring SP. Default of None indicates no inputs.
        #apical_stim_thresh -> Stimulus threshold to activate a minicolumn based on apical connections, or possibly a combination of apical and proximal.
        
        #Record the numeric parameters
        self.input_dim = input_dim
        self.column_num = column_num
        self.potential_percent = potential_percent
        self.max_active_cols = max_active_cols
        self.stimulus_thresh = stimulus_thresh
        self.perm_decrement = perm_decrement
        self.perm_increment = perm_increment
        self.perm_thresh = perm_thresh
        self.min_duty_cycle = min_duty_cycle
        self.duty_cycle_period = duty_cycle_period
        self.boost_str = boost_str
        self.apical_input_dim = apical_input_dim
        self.apical_stim_thresh = apical_stim_thresh
        
        #Initialize column activity tracker
        self.active_cols = np.zeros((column_num,))
        
        #Track the boost factors
        self.boost_factors = np.ones((column_num,))
                
        #Count how many inputs have been processed. Used for duty cycle tracking.
        self.input_cycles = 0
        
        ##Initialize the columns
        self.columns = [miniColumn(input_dim = input_dim, potential_percent = potential_percent, perm_decrement = perm_decrement, perm_increment = perm_increment, perm_thresh = perm_thresh, duty_cycle_period = duty_cycle_period, apical_input_dim = apical_input_dim) for i in range(column_num)]
        
        #Collect all of the column connections in a single numpy array
        self.all_connections = np.array([col.actual_connections for col in self.columns]).reshape(self.column_num,-1)

        #If apical connections exist, make all of these same arrays for apical inputs
        if self.apical_input_dim:
            self.apical_boost_factors = np.ones(self.apical_input_dim)
            self.all_apical_connections = np.array([col.actual_apical_connections for col in self.columns]).reshape(self.column_num,-1)

    def compute_overlap(self, arr, apical = False):
        #Reads in an encoded SDR, which is an array of 0s and 1s of shape input_dim that 
        #indicate inactive/active input bits respectively. Returns an output of shape 
        #(column_num,1) containing the overlap score of each minicolumn.
        
        #Get the raw overlap scores of each minicolumn.
        overlap_scores = np.zeros((self.column_num,))
        for i in range(self.column_num):
            overlap_scores[i] = self.columns[i].get_overlap_score(arr, apical)
            
        return overlap_scores
    
    def compute_overlap_par(self, arr, apical = False):
        #Just like compute overlap, but uses the all_connections array.
        #Testing to see if this is faster.
        try:
            if not apical:
                return np.dot(self.all_connections,arr)
            else:
                return np.dot(self.all_apical_connections,arr)
        except ValueError:
            print("Exception! apical is {}".format(apical))
        
    def update_boost_factors(self):
        #Recalculates the boost factors of each minicolumn if a duty cycle period has passed.
        if self.input_cycles >= self.duty_cycle_period:
            #Update the boost factors
            for i in range(self.column_num):
                self.boost_factors[i] = np.exp(-self.boost_str*(self.columns[i].get_duty_cycle()/self.duty_cycle_period - self.max_active_cols/self.column_num))
    
    def get_active_columns(self, overlaps, apical = False, reset = False):
        #Takes a set of overlap scores of shape (column_num, 1) and returns a 
        #binary array indicating inactive/active minicolumns.
        #Can be given either pre- or post-boost overlap scores.
        #Note that if many minicolumn share an overlap score, the number of active
        #minicolumns may exceed the active_cols value.

        #Reset the active cols to start, if this is a proximal input process
        if (not apical) or (reset):
            self.active_cols = np.zeros(self.active_cols.shape)
        self.active_cols[np.argpartition(overlaps,-self.max_active_cols)[-self.max_active_cols:]] = 1
        
        #Find the highest non-activated overlap score.
        # active_overlap_list = find_N_highest(overlaps, int(self.max_active_cols))
        # active_overlap_index = int(self.max_active_cols) - 1
        # min_active_overlap = active_overlap_list[active_overlap_index]
        
        # #Assign an activation score of 1 to all minicolumn with large enough overlap scores.
        # if not apical:
        #     self.active_cols = np.zeros(self.active_cols.shape)
        #     self.active_cols[(overlaps >= self.stimulus_thresh) & (overlaps >= min_active_overlap)] = 1
        # else:
        #     self.active_cols[(overlaps >= self.apical_stim_thresh) & (overlaps >= min_active_overlap)] = 1
            
        # # This while loop will ensure that, in the case of ties, there are still not too many active columns.        
        # # We'll be lenient and add a 20% buffer.
        # while (np.sum(self.active_cols) > self.max_active_cols*1.2) and (active_overlap_index > 0):
        #     # Decrement the active overlap index once each loop. In other words:
        #     # If last loop looked at the Nth highest, this time we'll look at the 
        #     # N-1th highest, and so on until the 1st highest overlap score.
        #     active_overlap_index -= 1
        #     min_active_overlap = active_overlap_list[active_overlap_index]
        #     if active_overlap_index == 0:
        #         print(np.sum(self.active_cols))
            
        #     # Repeat the above process of activation assignment based on overlap score
        #     if not apical:
        #         self.active_cols = np.zeros(self.active_cols.shape)
        #         self.active_cols[(overlaps >= self.stimulus_thresh) & (overlaps >= min_active_overlap)] = 1
        #     else:
        #         self.active_cols[(overlaps >= self.apical_stim_thresh) & (overlaps >= min_active_overlap)] = 1
            
        return self.active_cols
    
    def duty_cycle_update(self, apical = False):
        #Updates the duty cycle data of every column.
        for i in range(self.column_num):
            self.columns[i].duty_cycle_update(self.active_cols[i])
            
    def permanence_update(self, arr, apical=False):
        #Update the permanence of active minicolumns.
        #Can be used for either proximal or apical connections.
        for i in range(self.column_num):
            if self.active_cols[i] > 0:
                #Update the permanences of active columns
                self.columns[i].update_perms(arr,apical)
                
                #Update the all_connections array
                if not apical:
                    self.all_connections[i,:] = self.columns[i].actual_connections
                else:
                    self.all_apical_connections[i,:] = self.columns[i].actual_apical_connections
        
    def low_duty_cycle_inc(self):
        #Calls low_duty_cycle_inc() for each column with a duty cycle below the
        #minimum to encourage more activity.
        for i in range(self.column_num):
            if self.columns[i].get_duty_cycle()/self.duty_cycle_period <= self.min_duty_cycle:
                #Update the permanences
                self.columns[i].low_duty_cycle_inc(apical=False)
                #Update the all_connections array
                self.all_connections[i,:] = self.columns[i].actual_connections
                
                #If apical connections exist, update them too.
                if self.apical_input_dim is not None:
                    self.columns[i].low_duty_cycle_inc(apical=True)
                    self.all_apical_connections[i,:] = self.columns[i].actual_apical_connections
        
    def process_input(self, arr, boosting = True, apical = False, update_duty_cycle = True, sp_learning = True, reset = False):
        #Takes an encoded input SDR and goes through all of the steps needed to
        #process it, I.E. determining which minicolumns become active and performing
        #the learning updates. Returns an SDR of active minicolumns.

        #Increment the input counter. This is for duty cycle tracking purposes.
        #Since there may be multiple inputs to process per single cycle,
        #check the boolean update value first so that we do not update
        #more than once per cycle.
        if update_duty_cycle:
            self.input_cycles += 1     
        
        #Get the minicolumn overlap scores.
        pre_overlap_scores = self.compute_overlap_par(arr, apical)
        
        #Boost the overlap scores if boosting is being used.
        post_overlap_scores = pre_overlap_scores
        if boosting:
            self.update_boost_factors()
            post_overlap_scores = pre_overlap_scores*self.boost_factors
            
        #Determine the active minicolumns based on the net overlap scores
        self.get_active_columns(post_overlap_scores, apical, reset);
        
        #Update the duty cycle tracker for each minicolumn
        if update_duty_cycle:
            self.duty_cycle_update()
            
        #Update the permanences for each minicolumn, both proximal and apical
        if sp_learning:
            self.permanence_update(arr,apical)
        
       #Periodically increment the permanences of minicolumns with low duty cycles to promote more activity.
        if (self.input_cycles % self.duty_cycle_period == 0) and update_duty_cycle:
            self.low_duty_cycle_inc()
            
        #Return an SDR of the active columns.
        return self.active_cols
    
    def save_SP(self, path = '', name = 'Spatial_Pooler', string = ''):
        #Saves a copy of the SP to a file.
        filename = path + name + string + ".txt"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        
 
class Segment():
    #This is the most basic unit of the Temporal Memory algorithm.
    #Analogous to the miniColumn object, the Segment object stores an array of 
    #synaptic permanences and contains the methods necessary to update them
    #and grow new synapses. These synapses connect to Cells rather than to the
    #input space as the minicolumn connections do.
    
    def __init__(self, last_active_cells, own_index = 0, perm_thresh = 0.2, max_synapse_per_segment = 255, perm_increment = 0.1, perm_decrement = 0.1, incorrect_pred_dec = 0.0, initial_perm = 0.11, max_new_synapse = 20):
        #Constructor method.
        #last_active_cells -> An SDR of the cells that were active in the previous cycle. Segments are
        #grown specifically to connect to a given group of cells.
        #own_index -> The index corresponding to this segment in its parent Cell.
        #perm_thresh -> The permanence threshold for growing a new connection.
        #max_synapse_per_segment -> The maximum number of allowed connections
        #perm_increment -> Amount by which the permanence to an active cell increases in learning.
        #perm_decrement -> Amount by which the permanence to an inactive cell decreases in learning.
        #incorrect_pred_dec -> Amount by which the permanences of a segment which made an 
        #incorrect prediction are decreased.
        #initial_perm -> All permanences in this segment begin at this value.
        #max_new_synapse -> The maximum number of new connections a segment can grow in one step.
        
        self.max_synapse_per_segment = max_synapse_per_segment
        self.perm_increment = perm_increment
        self.perm_decrement = perm_decrement
        self.incorrect_pred_dec = incorrect_pred_dec
        self.initial_perm = initial_perm
        self.max_new_synapse = max_new_synapse
        self.perm_thresh = perm_thresh
        
        #own_index will be used later for removing references to the segment's owner cell
        #Assuming the cell shouldn't have a synapse connecting to itself.
        self.own_index = own_index
        
        #Initialize the synapse permanences with the last_active_cells argument
        self.synapse_perms = initial_perm * last_active_cells
        self.actual_connections = (self.synapse_perms >= self.perm_thresh)
        
        #Check to make sure the initial connection number does not exceed self.max_synapse_per_segment
        total = np.sum(self.actual_connections)
        if total > self.max_synapse_per_segment:
            diff = int(total - self.max_synapse_per_segment)
            sorted_indices = np.argsort(self.actual_connections)
            self.actual_connections[sorted_indices[-diff:]] = 0
            
    def dec_perms(self):
        #Decrements the permanences of every synapse to punish an incorrect prediction.
        self.synapse_perms -= self.incorrect_pred_dec
        
        #Update the actual connections
        self.actual_connections = (self.synapse_perms >= self.perm_thresh)

    def inc_perms(self, previous_active_cells):
        #Increments the permanences of every synapse according to the previous activity of the other cells
        self.synapse_perms[previous_active_cells > 0] += self.perm_increment
        self.synapse_perms[previous_active_cells < 1] -= self.perm_decrement
        
        #Record the updated connections list, including both new and old synapses
        new_connections_list = (self.synapse_perms >= self.perm_thresh)
        
        ##Check to ensure that no more than max_new_synapse have been added by this update
        #Calculate the total number of added connections
        diff_arr = (new_connections_list > 0)*(self.actual_connections < 1)
        diff = int(np.sum(diff_arr)) - self.max_new_synapse
        
        #If the number of added connections exceeds self.max_new_synapse, delete the necessary number of them
        if diff > 0:
            sorted_indices = np.argsort(diff_arr)
            new_connections_list[sorted_indices[-diff:]] = 0
            diff_arr[sorted_indices[-diff:]] = 0
        
        ##Check to ensure added connections does not cause the total synapses to exceed self.max_synapse_per_segment
        total = int(np.sum(new_connections_list))
        if total > self.max_synapse_per_segment:
            diff = total - self.max_synapse_per_segment
            sorted_indices = np.argsort(diff_arr)
            new_connections_list[sorted_indices[-diff:]] = 0
            
        #Record the new connections
        self.actual_connections = new_connections_list
            
    def get_segment_overlap_score(self, active_cells):
        #Returns the overlap score this segment has with the given cell activity.
        return np.sum(active_cells*self.actual_connections)
         
class Cell():
    #The Cell object is the first level of abstraction in the Temporal Memory algorithm.
    #Each Cell curates a unique list of Segment objects with connections to the other Cells.
    def __init__(self, own_index, perm_thresh = 0.2, max_synapse_per_segment = 255, perm_increment = 0.1, perm_decrement = 0.1, incorrect_pred_dec = 0.0, initial_perm = 0.21, min_learning_thresh = 10, max_new_synapse = 20, max_segments = 20):
        #Constructor method.
        #own_index -> the index of this cell in its parent TemporalMemory object's list of Cells.
        #perm_thresh -> The permanence threshold for growing a new connection.
        #max_synapse_per_segment -> The maximum number of allowed connections
        #perm_increment -> Amount by which the permanence to an active cell increases in learning.
        #perm_decrement -> Amount by which the permanence to an inactive cell decreases in learning.
        #incorrect_pred_dec -> Amount by which the permanences of a segment which made an 
        #incorrect prediction are decreased.
        #initial_perm -> All permanences in this segment begin at this value.
        #min_learning_thresh -> Minimum overlap score which is sufficient for learning but not activation. ##NOTE: This functionality is not implemented yet.
        #max_new_synapse -> The maximum number of new connections a segment can grow in one step.
        #max_segments -> The maximum number of segments each Cell can have.
        
        
        #Record all of the basic parameters.
        self.max_synapse_per_segment = max_synapse_per_segment
        self.perm_increment = perm_increment
        self.perm_decrement = perm_decrement
        self.incorrect_pred_dec = incorrect_pred_dec
        self.initial_perm = initial_perm
        self.min_learning_thresh = min_learning_thresh
        self.max_new_synapse = max_new_synapse
        self.max_segments = max_segments
        self.perm_thresh = perm_thresh
        self.own_index = own_index
        self.TooManySegmentsErrorCount = 0
        
        #Initialize the segments and connections list
        self.segments = []
        self.all_connections = []
    
        #Initialize the variables that record the index and overlap strength
        #of the most overlapping segment at each step, used for later learning.
        self.predictive_overlap = -1
        self.predictive_segment = -1    

    def add_segment(self, active_cells):
        #Grows a new segment for this cell with connections to the given active cells.
        if len(self.segments) < self.max_segments:
            self.segments.append(Segment(last_active_cells = active_cells, perm_thresh = self.perm_thresh, max_synapse_per_segment = self.max_synapse_per_segment, perm_increment = self.perm_increment, perm_decrement = self.perm_decrement, incorrect_pred_dec = self.incorrect_pred_dec, initial_perm = self.initial_perm, max_new_synapse = self.max_new_synapse))
            #Record the index of the newest segment for learning in the next stage
            self.predictive_segment = len(self.segments) - 1
            
            #Update the all_connections array
            self.all_connections = np.array([segment.actual_connections for segment in self.segments]).reshape((len(self.segments),-1))
            
        else:
            #raise ValueError('Cell already has the maximum number of segments!')
            self.TooManySegmentsErrorCount += 1
    
    def get_max_overlap_score(self,active_cells):
        #Returns the largest overlap score and the segment index for the given activity among all the segments this cell has.
        #If the cell has no segments, returns -1.
        ##NOTE: I tried vectorizing this process by using a N+1-dimensional array consisting of 
        #the connections of all the cell's segments and then broadcasting active_cells to it.
        #It's about 20% slower than the explicit for-loop in this version of the method.
        if len(self.segments) == 0:
            return -1
        
        max_val = -1
        max_index = 0
        for i, segment in enumerate(self.segments):
            val = segment.get_segment_overlap_score(active_cells)
            if val >= max_val:
                max_val = val
                max_index = i
        #Also records the index of the largest-overlap segment, in case it is selected for learning
        self.predictive_segment = max_index
        self.predictive_overlap = max_val
        return max_val
    
    def get_max_overlap_score_par(self,active_cells):
        #Just like get_max_overlap_score, but uses the all_connections array.
        #Should be faster.
        if len(self.segments) == 0:
            return -1
        
        scores = np.dot(self.all_connections,active_cells)
        self.predictive_segment = np.argmax(scores)
        self.predictive_overlap = scores[self.predictive_segment]
        return self.predictive_overlap
    
    def update_perms(self, previous_active_cells = 0,correct_prediction = False):
        #Updates the permanences of the synapses in a given segment of this cell according to the given activity.
        if correct_prediction:
            #If this cell made a correct prediction, increment its predictive segment permanences
            if self.predictive_segment == -1:
                return
            
            try:
                #Note that if correct_prediction is true, previous_active_cells should be a numpy array
                #rather than the default value of 0.
                self.segments[self.predictive_segment].inc_perms(previous_active_cells) 
            except IndexError:
                print('Predictive index {} outside maximum segment index {}!'.format(self.predictive_segment,len(self.segments)))
        else:
            #If this cell made an incorrect prediction, decrement its predictive segment permanences
            self.segments[self.predictive_segment].dec_perms()
 
class AnomalyTracker():
    #This object handles the tracking of anomaly scores for a TemporalMemory's predictions.
    def __init__(self, threshold = 0.9, estimation_samples = 100, max_window = 10000, decision_type = 'score_threshold'):
        #Constructor method.
        #threshold -> Score or likelihood threshold over which a result is deemed anomalous.
        #estimation_samples -> period after which the mean and deviation of recorded scores is updated.
        #max_window -> Total number of recent samples used for mean and dev calculation.
        #decision_type -> 'score_threshold' or 'likelihood_threshold', variable used to decide if a result is anomalous.
        
        self.estimation_samples = estimation_samples
        self.max_window = max_window
        self.scores = []
        self.likelihoods = []
        self.input_counter = 0
        self.threshold = threshold
        self.mean = 0
        self.dev = 0
        self.decision_type = decision_type
        
    def compute_anomaly_score(self, pred, active):
        #Computes and stores the anomaly score of one input given a set of predictive and active cells.
        #The anomaly score is simply 1 - (overlap between pred and active) / (# of active cells)
        #If the activity was predicted with 100% accuracy, the score will be 0.
        #If it was a complete surprise, the score will be 1.
        self.scores.append(1 - np.sum(pred*active)/np.sum(active))
        
    def compute_anomaly_likelihood(self):
        #Estimates the anomaly likelihood of the most recent anomaly score and records it.
        #The anomaly likelihood is based on an estimate of the Gaussian distribution of 
        #anomaly scores, using a sliding window of max_window size.
        #If the anomaly score is less than the mean, the likelihood is assumed to be zero.
        #Otherwise, the likelihood is 1 - (odds of getting that score, assuming normally distributed scores)
        if self.dev == 0:
            self.likelihoods.append(0)
            return
        
        if self.scores[-1] < self.mean:
            val = 0
        else:
            #Take 1 - (probability of getting a score this high, assuming normal distribution)
            val = 1 - 1/2*np.exp(-1/2*((self.scores[-1] - self.mean)/self.dev)**2)
        #Check to see if val is an empty object (which occurs if self.dev = 0)
        #In the unlikely event that the anomaly scores have all been zero, a divide by zero warning
        #will be produced. Replace the resulting nan value with a likelihood of 0.
        if (not val) or math.isnan(val):
            val = 0
            
        #Append the new likelihood value to the list
        self.likelihoods.append(val)
        
    
    def update_parameter_estimates(self):
        #Updates the stored estimates of mean and standard deviation in anomaly scores.
        #Uses a sliding window of the most recent data points, unless there are fewer than max_window
        #scores recorded, in which case it just uses all of them.
        
        if len(self.scores) <= self.max_window:
            self.mean = np.mean(self.scores)
            self.dev = np.std(self.scores)
        else:
            self.mean = np.mean(self.scores[-self.max_window:])
            self.dev = np.std(self.scores[-self.max_window:])
        
    def process_input(self, pred, active):
        #Takes a set of predictive and active cells and computes the anomaly score and likelihood.
        #Also updates the Gaussian distribution estimates if necessary.
        #Returns a binary decision based on the threshold and decision method specified during instantiation of the object.
        
        #First update the input counter
        self.input_counter += 1
        
        #If counter is a multiple of estimation_samples, update the Gaussian distribution estimates
        if self.input_counter % self.estimation_samples == 0:
            self.update_parameter_estimates()
        
        #Record the newest anomaly score
        self.compute_anomaly_score(pred,active)
        
        #Record the newest anomaly likelihood, if it's not too early
        if self.input_counter > self.estimation_samples:
            self.compute_anomaly_likelihood()
        else:
            self.likelihoods.append(0.0)
        
        #If the anomaly decision is based on a simple binary threshold
        if self.decision_type == 'score_threshold':
            return self.scores[-1] >= self.threshold
        #If it's based on the # of standard deviations above the mean
        elif self.decision_type == 'probability_threshold':
            return self.likelihoods[-1] >= self.threshold
        else:
            raise ValueError('Parameter "decision_type" not set to a valid option!')
            
class TemporalMemory():
    #The TemporalMemory object curates a list of Cell objects,
    #which themselves contain Segments, and also has a reference to a dedicated Spatial Pooler.
    def __init__(self, spatial_pooler, anomaly_tracker = None, num_cells = 16, stimulus_thresh = 13, initial_perm = 0.55, perm_thresh = 0.5, min_learning_thresh = 10, max_new_synapse = 20, perm_increment = 0.1, perm_decrement = 0.1, incorrect_pred_dec = 0.0, max_segments = 128, max_synapse_per_segment = 40, subthreshold_learning = False):
        #Constructor method.
        #spatial_pooler -> The dedicated spatial pooler used to determine what minicolumns
        #become active in response to an input SDR.
        #anomaly_tracker -> The dedicated anomaly tracker object.
        #num_cells -> The number of cells per minicolumn.
        #stimulus_thresh -> The overlap threshold over which a cell becomes predictive.
        #initial_perm -> The permanence all segment synapses begin with.
        #perm_thresh -> The permanence threshold over which a synapse becomes a true connection.
        #min_learning_thresh -> Minimum overlap score which is sufficient for learning but not activation. ##NOTE: This functionality is not implemented yet.
        #max_new_synapse -> The maximum number of new connections a segment can grow in one step.
        #perm_increment -> Amount by which the permanence to an active cell increases in learning.
        #perm_decrement -> Amount by which the permanence to an inactive cell decreases in learning.
        #incorrect_pred_dec -> Amount by which the permanences of a segment which made an 
        #incorrect prediction are decreased.
        #max_segments -> The maximum number of segments each Cell can have.
        #max_synapse_per_segment -> The maximum number of allowed connections
        #subthreshold_learning -> Whether or not a cell can learn if its overlap was too low to become
        #active but exceeded min_learning_thresh. ##NOTE: This functionality is not implemented yet.

        #Record all the basic parameters.
        self.column_num = spatial_pooler.column_num
        self.num_cells = num_cells
        self.stimulus_thresh = stimulus_thresh
        self.initial_perm = initial_perm
        self.perm_thresh = perm_thresh
        self.min_learning_thresh = min_learning_thresh
        self.max_new_synapse = max_new_synapse
        self.perm_increment = perm_increment
        self.perm_decrement = perm_decrement
        self.incorrect_pred_dec = incorrect_pred_dec
        self.max_segments = max_segments
        self.max_synapse_per_segment = max_synapse_per_segment
        self.active_cols = np.zeros((self.column_num,))
        self.subthreshold_learning = subthreshold_learning
        
        if subthreshold_learning:
            self.subthreshold_cells = np.zeros((self.num_cells*self.column_num,))
        
        #Create a list of cells that is num_cells*column_num long.
        self.cells = [Cell(own_index = i, max_synapse_per_segment = self.max_synapse_per_segment, perm_increment = self.perm_increment, perm_decrement = self.perm_decrement, incorrect_pred_dec = self.incorrect_pred_dec, initial_perm = self.initial_perm, perm_thresh = self.perm_thresh, min_learning_thresh = self.min_learning_thresh, max_new_synapse = self.max_new_synapse) for i in range(num_cells*self.column_num)]
        
        #Initialize a tracker for last time's active cells
        self.last_active_cells = np.zeros((num_cells*self.column_num,))
        
        #Initialize a tracker for the currently active cells
        self.active_cells = self.last_active_cells
        
        #Initialize a tracker for the predictive cells
        self.predictive_cells = np.zeros((num_cells*self.column_num,))
    
        #Record the spatial pooler
        self.spatial_pooler = spatial_pooler
        
        #Record the anomaly tracker, if any
        self.anomaly_tracker = anomaly_tracker

    def get_active_columns(self, SDR):
        #Takes an SDR input and calls the spatial pooler to determine which columns become active.
        #This method does not let the SP use boosting or make learning updates.
        return self.spatial_pooler.get_active_columns(self.spatial_pooler.compute_overlap_par(SDR))

    def find_active_cells(self, learning = False):
        #Takes in a list of active columns and updates the active cells.
        #Note: I tried using array assignments to instantly assign all of the predictive cells in active columns
        #to active cells, but it was actually about 20% slower than the explicit for-loop code below. I don't know why, and it wasn't because of 
        #the weirdness of rank 1 numpy arrays.
        
        #Reset the active cell array
        self.active_cells = np.zeros(self.active_cells.shape) 
        for col_index, col_bit in enumerate(self.active_cols):
            #For each active column:
                if col_bit:
                    #For each corresponding cell:
                        predictive_exists = False
                        for cell_index in range(col_index*self.num_cells, (col_index+1)*self.num_cells):
                            if self.predictive_cells[cell_index]:
                                #If any of the cells is predictive, make it active.
                                #If multiple cells are predictive, they all become active.
                                self.active_cells[cell_index] = 1
                                predictive_exists = True
                            
                        #If none of the cells in this column were predictive, burst the column
                        #This means select one of the cells randomly and make it active.
                        if (not predictive_exists):
                            cell_index = rand.randint((col_index)*self.num_cells, (col_index+1)*self.num_cells - 1)
                            self.active_cells[cell_index] = 1
                            
                            #In case this cell will not grow a new segment, indicate no predictions
                            self.cells[cell_index].predictive_segment = -1
                            
                            #If there is learning to be done, add a segment to the cell.
                            if (learning) and (np.sum(self.last_active_cells) > 0):
                                #No learning takes place if there were no previously-active cells,
                                #which happens if this is the first input to the temporal memory.
                                self.cells[cell_index].add_segment(self.last_active_cells)
                                #The new segment takes in the list of previously active cells and forms synapses to them.
                                
        return self.active_cells

    def find_predictive_cells(self, all_cells_allowed = True, num_allowed = -1):
        #Calculates which cells become predictive now based on current cell activity.
        #The all_cells_allowed variable determines if each column can be represented by
        #all of its cells, or some preset number of them. Setting all_cells_allowed to True
        #is highly recommended if speed is an issue.
        #This method also checks which cells don't quite meet the prediction threshold but
        #are still eligible to learn. These cells will be updated during this cycle, although
        #they wouldn't have become active until *next* cycle if they were predictive.
        #This doesn't introduce an error in the algorithm since the cell activity
        #method doesn't actually check overlap scores, but rather predictive state.
        if all_cells_allowed:
            num_predictive_allowed = self.num_cells
        else:
            num_predictive_allowed = num_allowed
            
        #Reset the predictive cell array and learning array
        self.predictive_cells  = np.zeros(self.predictive_cells.shape)
        
        if self.subthreshold_learning:
            self.subthreshold_cells = np.zeros(self.subthreshold_cells.shape)

        #Loop over each column index to check if any of its cells become predictive
        for i in range(self.column_num):
            
            #If 'all_cells_allowed' is true, then all the cells in a column can be predictive
            #There is no need to sort them by overlap score, just compare each to the threshold.
            #This allows a tremendous acceleration of the program. 
            if all_cells_allowed:
                #Check each cell corresponding to this column and, if it exceeds the stimulus threshold,
                #record it in the 'cells' variable.
                for cell in self.cells[self.num_cells*(i):self.num_cells*(i+1)]:
                    
                    #Get the overlap score
                    score = cell.get_max_overlap_score_par(self.active_cells)
                    
                    #Check it against the stimulus threshold and learning threshold
                    if score >= self.stimulus_thresh:
                        self.predictive_cells[cell.own_index] = 1
                    elif self.subthreshold_learning and (score >= self.min_learning_thresh):
                        self.subthreshold_cells[cell.own_index] = 1
                        
            #If only a certain number of cells are allowed to activate per column:
            else:
                #Get a list of tuples containing the cell and the cell's max overlap score.
                max_overlaps = [(cell, cell.get_max_overlap_score_par(self.active_cells)) for cell in self.cells[self.num_cells*(i):self.num_cells*(i+1)]]
                
                #Record the subthreshold learning cells:
                if self.subthreshold_learning:
                    for pair in max_overlaps:
                        if pair[1] >= self.min_learning_thresh and pair[1] < self.stimulus_thresh:
                            self.subthreshold_cells[pair[0].own_index] = 1
                    
                #Sort the tuples by overlap score
                max_overlaps.sort(reverse=True, key=(lambda x: x[1]))
                
                for j in range(num_predictive_allowed):
                #Check the top few (number specified by num_predictive_allowed) and make them predictive
                #if their overlaps exceed the threshold.
                    if max_overlaps[j][1] >= self.stimulus_thresh:
                        #If the biggest few overlap scores among the cells exceeds the stimulus threshold,
                        #make those cells predictive. Otherwise none of them will be. 
                        idx = max_overlaps[j][0].own_index
                        self.predictive_cells[idx] = 1
                
        return self.predictive_cells
    
    def process_input(self, SDR, tm_learning = True, sp_learning = False, all_cells_allowed = True, num_allowed = -1, sparse_output = False):
        #Reads in an SDR, allows the SP to process it, determines which Cells
        #become active and predictive, then performs all of the learning updates.
        #Returns SDRs of the active and predictive cells as a tuple.
        if sp_learning:
            self.active_cols = self.spatial_pooler.process_input(SDR)
        else:
            self.active_cols = self.get_active_columns(SDR)
        
        #Cycle the last_active_cells and active_cells
        self.last_active_cells = self.active_cells
        
        #Get the new active cells based on the active columns and currently predictive cells
        self.find_active_cells(tm_learning);
        
        #If this TM has an anomaly tracker, update its scores
        #This needs to be done after the new active cells are computed but before the old predictive cells
        #are overwritten.
        if self.anomaly_tracker:
            self.anomaly_tracker.process_input(self.predictive_cells, self.active_cells)
        
        #Update the permanences of the active cells' highest-overlap segments.
        if tm_learning:
            for i in range(self.num_cells*self.column_num):
                #If the cell is active, train it on last iteration's cell activity
                if (self.active_cells[i] > 0):
                    self.cells[i].update_perms(self.last_active_cells,correct_prediction = True)
                #If the cell is not active but was predictive, lower its permanences
                elif self.predictive_cells[i] > 0:
                        self.cells[i].update_perms(correct_prediction = False)
        
        #Get the new predictive cells based on the new active cells
        self.find_predictive_cells();
        
        #Now update the cells that were below the predictive threshold but exceeded learning threshold
        #on the current cell activity
        if tm_learning and self.subthreshold_learning:
            for i in range(self.num_cells*self.column_num):
                if self.subthreshold_cells[i] > 0:
                    #Note that we 'reward' the cells as if they had correctly predicted activity
                    #even though they won't become active next cycle since they failed to exceed the 
                    #prediction threshold (stimulus_thresh)
                    self.cells[i].update_perms(self.active_cells,correct_prediction = True)
        
        if sparse_output:
            return sparse.csr_matrix(self.active_cells), sparse.csr_matrix(self.predictive_cells)
        else:
            return self.active_cells, self.predictive_cells
    
    def reset(self):
        #Resets all column and cell activity data.
        self.last_active_cells = np.zeros_like(self.last_active_cells)
        self.active_cells = self.last_active_cells
        self.predictive_cells = self.last_active_cells
        self.active_cols = np.zeros_like(self.active_cols)
            
    def plot_cells(self,type='predictive'):
        #Visualize the cells as a grid of black and white squares.
        if type == 'predictive':
            cells = self.predictive_cells
        elif type == 'active':
            cells = self.active_cells
        else:
            raise ValueError('Invalid type!')
            return
        
        cells_list = []
        for num in range(self.num_cells):
            cells_list.append(np.array([cells[self.num_cells*i + num] for i in range(self.column_num)]))
        
        fig, axes_list = plt.subplots(1,self.num_cells)
        
        if self.num_cells == 1:
            plot_SDR(cells_list[0], axesObject = axes_list)
        else:
            for index in range(self.num_cells):
                plot_SDR(cells_list[index], axesObject=axes_list[index])
            
    def save_cells(self, string = '', path = ''):
        #Save the Cell objects individually. Used when the TM object is too 
        #large to be saved as a whole.
        for index, cell in enumerate(self.cells):
            print("\n Saving cell number " + str(index) + " ...")
            filename = path + "Cell" + str(index) + string + ".txt"
            with open(filename, 'wb') as f:
                pickle.dump(cell, f)
                
    def load_cells(self, num, string = '', path = ''):
        #Load Cell objects from files.
        for i in range(num):
            print("\n Loading cell number " + str(i) + " ...")
            filename = path + "Cell" + str(i) + string + ".txt"
            with open(filename, 'rb') as f:
                self.cells[i] = pickle.load(f)
                
class Regressor():
    #This object contains an SKlearn regressor and learns to translate from
    #SDRs to scalar values.
    def __init__(self,SDR_list,value_list,regressor_type = 'KNN', report_score = True, *args, **kwargs):
        #Constructor method. 
        #SDR_list -> A list of 1-D SDRs, either column/row vectors or rank-1 numpy matrices.
        #value_list -> A list of values corresponding to the SDRs, used for training.
        #regressor_type -> Type of regressor to be used, passed as a string
        #report_score -> Whether or not to report the training score.
        #*args, **kwargs -> Extra arguments passed to the regressor constructor.
        
        #Reshape the inputs.
        SDRs = [np.array(SDR).reshape(-1,) for SDR in SDR_list]
        if regressor_type == 'KNN':
            from sklearn.neighbors import KNeighborsRegressor as KNN
            #Instantiate and fit the regressor
            self.reg = KNN(*args, **kwargs).fit(SDRs,value_list)
        elif regressor_type == 'SVM':
            from sklearn.svm import SVR
            #Instantiate and fit the regressor
            self.reg = SVR(*args, **kwargs).fit(SDRs,value_list)
        elif regressor_type == 'linear':
            from sklearn.linear_model import LinearRegression as LR
            #Instantiate and fit the regressor
            self.reg = LR(*args, **kwargs).fit(SDRs,value_list)
        elif regressor_type == 'tree':
            from sklearn.tree import DecisionTreeRegressor as DTR
            #Instantiate and fit the regressor
            self.reg = DTR(*args,**kwargs).fit(SDRs,value_list)
        elif regressor_type == 'forest':
            from sklearn.ensemble import RandomForestRegressor as RFR
            #Instantiate and fit the regressor
            self.reg = RFR(*args,**kwargs).fit(SDRs,value_list)
        else:
            print('Input "regressor_type" not set to a valid value! Defaulting to KNN...')
            from sklearn.neighbors import KNeighborsRegressor as KNN
            #Instantiate and fit the regressor
            self.reg = KNN(*args, **kwargs).fit(SDRs,value_list)
        
        #Record and report the score of the regressor.
        self.score = -1
        if report_score:
            self.score = self.reg.score(SDRs,value_list)
            print("R2 score on training SDRs: " + str(self.score))
            
    def translate(self,x):
        #Takes a single SDR or a list of SDRs and translates into a scalar value using the regressor.
        
        #If x is a list, assume it is a list of SDRs.
        if type(x) == list:
            return [self.reg.predict(np.array(SDR).reshape(1,-1))[0] for SDR in x]
        #Otherwise, assume it is a single SDR.
        else:
            return self.reg.predict(np.array(x).reshape(1,-1))[0]
                    
def transform_2D(arr):
    #Convenience function used to transform a 1D SDR into 2D for visualization purposes.
    
    #Create a new zeros array of size (new_len, new_len)
    #where new_len is ceiling(sqrt(len(arr))).
    new_len = int(np.ceil(np.sqrt(len(arr))))
    new_arr = np.zeros((new_len,new_len))
    
    #Fill in the new array row by row until you run out of values from arr
    start_index = 0
    row_index = 0
    while start_index < len(arr) - 1:
        end_index = min(start_index + new_len,len(arr))
        new_arr[row_index,:end_index - start_index] = arr[start_index:end_index]
        start_index += new_len
        row_index += 1

    #Return the new 2D padded array.
    return new_arr
           
def plot_SDR(in_arr, axesObject = None):
    #Convenience function used to visualize SDRs.
    
    #Check if the figure is 1D. If it is, create a new padded array of size (new_len, new_len)
    #where new_len is ceiling(sqrt(len(arr)))
    if len(in_arr.shape) == 2:
        if in_arr.shape[1] == 1:
            arr = in_arr.flatten()
            arr = transform_2D(arr)
        else:
            arr = in_arr.copy()
    else:
        arr = transform_2D(in_arr)
  
    #Plot the array.
    if axesObject:
        axesObject.imshow(arr, cmap='Greys')
    else:
        plt.close()
        plt.figure()
        plt.imshow(arr, cmap='Greys')
        
def overlap(sdr1, sdr2):
    #Returns the overlap score of two SDRs with matching shapes.
    return np.sum(sdr1*sdr2)

def find_N_highest(arr,N):
    #Finds the nth highest value in a 1D array.
    #Should be faster than sorting if N << len(arr)
    Nlist = []
    for n in range(N):
        val = -1
        #Loop through the array
        for x in arr:
            #Find the largest value in the array that's not in Nlist
            if x > val and x not in Nlist:
                val = x
        #Append the largest value found in this iteration to Nlist
        Nlist.append(val)
    #The resulting list is the N highest values in arr, sorted in descending order
    return Nlist
        