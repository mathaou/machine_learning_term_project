#!/bin/python3

# import hand_classifier as hand

import numpy as np
# import pickle
import os.path
import random

class PokerHandANN():
    # remember the .6:.2:.2 split for testing/training/validation
    # need to compile all the lines into one file and then go from there
    inputs = np.array([])
    targets = np.array([])
    testing = []
    validation = []

    nhidden = 10
    beta = 1
    momentum = .9
    outtype = 'linear'
    w = .25
    num_iterations = 1000

    pickle_file_path = 'pkr_hnd.pkl'

    def __init__(self, hand, mqtt):

        # files_exist = os.path.exists(self.pickle_file_path)
        with open("poker-hand.data") as file:
            data = file.readlines()
            data = data[0:30000]
            # define splits
            testing_validation_split = int(len(data) * .2)

            if mqtt is not None:
                print("Sending status...")
                mqtt.publish("hand/client", "Initializing network...")

            print("Splitting testing set...")
            # divy up testing
            self.testing = data[:-testing_validation_split]
            data = data[:-testing_validation_split or None]

            print("Splitting validation set...")
            # divy up validation
            self.validation = data[:-testing_validation_split]

            print("Assigning inputs and targets for the remaining 60% of data...")
            # assigning inputs and targets with whats left (60%)
            (self.inputs, self.targets) = self.createInputsAndTargets(data[0:len(data) - testing_validation_split])

            print("Assigning inputs and targets for validation set...")
            # create validation
            (validation_input, validation_target) = self.createInputsAndTargets(self.validation)

            self.nin = np.shape(self.inputs)[1]
            self.nout = np.shape(self.targets)[1]
            self.ndata = np.shape(self.inputs)[0]
        
            # Initialise network
            self.weights1 = (np.random.rand(self.nin+1,self.nhidden)-0.5)*2/np.sqrt(self.nin)
            self.weights2 = (np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)
            
            # print("Splitting validation set...")
            # # divy up validation
            # self.validation = data[:-testing_validation_split]

            # print("Assigning inputs and targets for the remaining 60% of data...")
            # # assigning inputs and targets with whats left (60%)
            # (self.inputs, self.targets) = self.createInputsAndTargets(data[0:len(data) - testing_validation_split])

            # print("Assigning inputs and targets for validation set...")
            # # create validation
            # (validation_input, validation_target) = self.createInputsAndTargets(self.validation)

            # possibly make random for repeated trainings
            # data = random.sample(data, len(data))
            print("Beginning training...")
            self.mlptrain(self.inputs, self.targets, self.w, self.num_iterations)

            print("Validation...")
            self.confmat(validation_input, validation_target)
        
        print("Creating testing inputs and targets")
        (testing_input, testing_targets) = self.createInputsAndTargets(self.testing)
        
        print("Recall...")
        self.confmat(testing_input, testing_targets)

    def createInputsAndTargets(self, input):
        inputs = []
        targets = []
        print("Beginning input traversal...")
        v = 0
        for line in input:
            if v % 1000 == 0:
                print("Iteration {0}/{1} complete...".format(v, len(input)))
            v += 1
            i = line.split(",")
            temp_input = np.array(i[:-1]).astype(np.float)
            if len(inputs) == 0:
                inputs = np.concatenate((temp_input, inputs), axis=0)
            else:
                inputs = np.vstack((temp_input, inputs))
            if len(targets) == 0:
                targets = np.concatenate((np.array(i[-1:]).astype(np.float), targets), axis=0)
            else:
                targets = np.vstack((np.array(i[-1:]).astype(np.float), targets))
        return (inputs, targets)
            
    def earlystopping(self,inputs,targets,valid,validtargets,eta,niterations=100):
    
        valid = np.concatenate((valid,-np.ones((np.shape(valid)[0],1))),axis=1)
        
        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000
        
        count = 0
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1)>0.001)):
            count+=1
            print(count)
            self.mlptrain(inputs,targets,eta,niterations)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.mlpfwd(valid)
            new_val_error = 0.5*np.sum((validtargets-validout)**2)
            
        print("Stopped", new_val_error,old_val_error1, old_val_error2)
        return new_val_error
        
    def mlptrain(self,inputs,targets,eta,niterations):
        """ Train the thing """    
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((self.ndata,1))),axis=1)
        # change = range(self.ndata)
    
        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))
            
        for n in range(niterations):
    
            self.outputs = self.mlpfwd(inputs)

            error = 0.5*np.sum((self.outputs-targets)**2) 
            if (np.mod(n,100)==0):
                print ("Iteration: ",n, " Error: ",error)    

            # Different types of output neurons
            if self.outtype == 'linear':
                deltao = (self.outputs-targets)/self.ndata
            elif self.outtype == 'logistic':
                deltao = self.beta*(self.outputs-targets)*self.outputs*(1.0-self.outputs) # 4.8
            elif self.outtype == 'softmax':
                deltao = (self.outputs-targets)*(self.outputs*(-self.outputs)+self.outputs)/self.ndata 
            else:
                print ("error")
            
            deltah = self.hidden*self.beta*(1.0-self.hidden)*(np.dot(deltao,np.transpose(self.weights2))) # 4.9
                      
            updatew1 = eta*(np.dot(np.transpose(inputs),deltah[:,:-1])) + self.momentum*updatew1 # 4.11
            updatew2 = eta*(np.dot(np.transpose(self.hidden),deltao)) + self.momentum*updatew2 # 4.10
            self.weights1 -= updatew1
            self.weights2 -= updatew2
                
            # Randomise order of inputs (not necessary for matrix-based calculation)
            #np.random.shuffle(change)
            #inputs = inputs[change,:]
            #targets = targets[change,:]
            
    def mlpfwd(self,inputs):
        """ Run the network forward """

        self.hidden = np.dot(inputs,self.weights1) # 4.4
        self.hidden = 1.0/(1.0+np.exp(-self.beta*self.hidden)) # 4.5
        self.hidden = np.concatenate((self.hidden,-np.ones((np.shape(inputs)[0],1))),axis=1) # bias node

        outputs = np.dot(self.hidden, self.weights2) # 4.6

        # Different types of output neurons
        if self.outtype == 'linear':
            return outputs
        elif self.outtype == 'logistic':
            return 1.0/(1.0+np.exp(-self.beta*outputs)) # 4.7
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(outputs),axis=1)*np.ones((1,np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs))/normalisers)
        else:
            print ("error")

    def confmat(self,inputs,targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs = self.mlpfwd(inputs)
        
        nclasses = np.shape(targets)[1]

        if nclasses==1:
            nclasses = 2
            outputs = np.where(outputs>0.5,1,0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)

        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

        print ("Confusion matrix is:")
        print (cm)
        print ("Percentage Correct: ",np.trace(cm)/np.sum(cm)*100)

# PokerHandANN()