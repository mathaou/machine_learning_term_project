#!/bin/python3

# import hand_classifier as hand
# import pandas as pd
import paho.mqtt.client as MQTT
import time
import json
import pickle

import traceback
import re
import numpy as np
import pylab as pl
import os.path
import random
import sys

class PokerHandANN(object):
    # remember the .6:.2:.2 split for testing/training/validation
    # need to compile all the lines into one file and then go from there
    inputs = np.array([])
    targets = np.array([])
    testing = []
    validation = []
    errors = []
    valid_errors = []

    nhidden = 22
    beta = 1
    momentum = .9
    outtype = 'softmax'
    w = .3
    num_iterations = 200

    def __init__(self, nhidden):
        # np.set_printoptions(threshold=sys.maxsize)
        # TODO balance training self.data
        with open("poker-hand-test.data") as file:
            self.errors = []
            self.valid_errors = []
            self.nhidden = nhidden
            self.data = file.readlines()
            random.shuffle(self.data)
            self.num_data = (41 * self.nhidden + (self.nhidden + 1) * 10)
            self.data = self.data[:self.num_data]
            # define splits
            self.testing_validation_split = int(len(self.data) * .2)

            print("Splitting testing set...")
            # divy up testing
            self.testing = self.data[:-self.testing_validation_split]
            self.data = self.data[:-self.testing_validation_split or None]

            print("Splitting validation set...")
            # divy up validation
            self.validation = self.data[:-self.testing_validation_split]

            print("Assigning inputs and targets for the remaining 60% of self.data...")
            # assigning inputs and targets with whats left (60%)
            (self.inputs, self.targets) = self.createInputsAndTargets(self.data[0:len(self.data) - self.testing_validation_split])

            print("Assigning inputs and targets for validation set...")
            # create validation
            (self.validation_input, self.validation_target) = self.createInputsAndTargets(self.validation)

            self.nin = np.shape(self.inputs)[1]
            self.nout = np.shape(self.targets)[1]
            self.ndata = np.shape(self.inputs)[0]

            # print(self.inputs)
        
            # Initialise network
            self.weights1 = (np.random.rand(self.nin+1,self.nhidden)-0.5)*2/np.sqrt(self.nin)
            self.weights2 = (np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)
            
            print("Beginning training...")
            # self.mlptrain(self.inputs, self.targets, self.w, self.num_iterations)

            self.earlystopping(self.inputs, self.targets, self.validation_input, self.validation_target, self.w, self.num_iterations)

        # pl.plot(*zip(*self.valid_errors), 'b-', label='Validation Set Error')

    # This function creates the inputs and targets for us based on .csv string
    def createInputsAndTargets(self, input):
        inputs = []
        targets = []
        print("Beginning input traversal...")
        v = 0
        # for each line of input provided
        for line in input:
            # split on comma
            l = line.split(",")
            
            # target values are encoded to a list and appended to the containing list
            # print(l)
            targets.extend(list(map(lambda x: encode_long(int(x[0]), 10), l[-1:])))

            # the remainder of the list is converted to integers to make it easier in rest of encoding
            l = list(map(lambda x: int(x) - 1, l[:-1]))
            temp_input = []
            # print(l)

            # every thousand iterations
            if v % 10 == 0:
                print("Iteration {0}/{1} complete...".format(v, len(input)))
            v += 1

            # traverse list in groups of two to get suit/rank pairs
            for i in range(0, len(l), 2):
                # temporary list is extended rather than appended in suit/rank format
                temp_input.extend(encode(l[i], 4))   
                temp_input.extend(encode(l[i+1], 13))
            # print(temp_input)
            # then appended to containing list
            inputs.append(temp_input)
            temp_input = []

        # this counts how many of each type of class appears in targets
        dic = {}
        for elem in targets:
            for i in range(10):
                if elem[i] == 1:
                    key = str(i)
                    dic[key] = dic.get(key, 0) + 1

        print(dic)
        # print(inputs[0])

        # converted to numpy arrays with high level of precision needed to succeed with exponentiation of softmax and logistic
        return (np.array(inputs, dtype=np.float64), np.array(targets, dtype=np.float64))

    # textbook says the (input - mean)/(max-min)
    def normalize(self, arr):
        max = np.max(arr)
        min = np.min(arr)
        mean = np.mean(arr)
        return np.vectorize(lambda x: (x - mean)/(max-min))(arr)
            
    def earlystopping(self,inputs,targets,valid,validtargets,eta,niterations=100):
        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000
        
        count = 0
        self.outputs = np.array([])
        self.inputs = inputs
        self.targets = targets
        valid = np.concatenate((valid,-np.ones((np.shape(valid)[0],1))),axis=1)
        new_data = np.array_split(self.data, 10)
        # TODO Implement randomization
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1)>0.001)):
            print("Total iterations of {0}: {1}".format(niterations, count))
            count+=1
            self.mlptrain(self.inputs, self.targets, eta, niterations)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.mlpfwd(valid)
            new_val_error = 0.5*np.sum((validtargets-validout)**2)

            # # comment this
            # val = random.randint(0, 9)
            # # print("Validation input: {0}".format(new_data[val]))
            # (valid, validtargets) = self.createInputsAndTargets(new_data[val])
            # temp = new_data[val]
            # new_data.pop(val)
            # # print("INPUTS: {0}".format(new_data))
            # t = np.concatenate(new_data).ravel()
            # # print(t)
            # (self.inputs, self.targets) = self.createInputsAndTargets(t)
            # valid = np.concatenate((valid,-np.ones((np.shape(valid)[0],1))),axis=1)
            # # print("INPUTS: {0}".format(self.inputs))
            # new_data.append(temp)
            # random.shuffle(new_data)

            # other way

            random.shuffle(self.data)
            valid = self.data[:-self.testing_validation_split]
            (self.inputs, self.targets) = self.createInputsAndTargets(self.data[:len(self.data) - self.testing_validation_split])
            (valid, validtargets) = self.createInputsAndTargets(valid)
            valid = np.concatenate((valid,-np.ones((np.shape(valid)[0],1))),axis=1)

            self.errors.append((count, new_val_error))
            
        print("Stopped", new_val_error,old_val_error1, old_val_error2)
        return new_val_error
        
    def mlptrain(self,inputs,targets,eta,niterations):
        """ Train the thing """    
        # Add the inputs that match the bias node
        self.ndata = len(inputs)
        inputs = np.concatenate((inputs,-np.ones((self.ndata,1))),axis=1)

        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))

        for n in range(niterations):
            
            self.outputs = self.mlpfwd(inputs)

            error = 0.5*np.sum((self.outputs-targets)**2) 
            if (np.mod(n,10)==0):
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
        
        # np.set_printoptions(threshold=sys.maxsize)
        print(outputs)

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
        return np.trace(cm)/np.sum(cm)*100

pickle_file_path = 'pkr_hand_86.pkl'

"""
0, 8 // 8 heart
1, 10 // 10 spade
1, 12 // queen spade
3, 11 // jack club
2, 4 // 4 diamond

"""

def encode_long(x, n):
    out = [0] * n
    try:
        out[x] = 1
    except:
        pass
        # print(x)
    print(out)
    return out

def encode(n, x):
    # print(n)
    temp = "{0}".format('{0:04b}'.format(n))
    # print(temp)
    ret = []
    [ret.extend(list(x)) for x in temp]
    # print(ret)
    return ret

def output_result(arr):
    arr = np.concatenate((arr,-np.ones((np.shape(arr)[0],1))),axis=1)
    obj_file = get_pickle()
    fwd = obj_file.mlpfwd(arr)
    results = fwd[0]
    res = {}
    resultsSort = results.argsort()[-5:][::-1]
    for i in resultsSort:
        res["{0}".format(int(i))] = float(results[i])
        
    return res

def get_pickle():
    f = open(pickle_file_path, "rb")
    obj_file = pickle.load(f)
    f.close()
    return obj_file

def calc_final_error(o):
    print("Creating testing inputs and targets")
    (o.testing_input, o.testing_targets) = o.createInputsAndTargets(o.testing)
    
    print("Recall...")
    return o.confmat(o.testing_input, o.testing_targets)

def get_output(ti):
    with open("poker-hand-test.data") as file:
        lines = file.readlines()

        found = False
        for line in lines:
            for i in range(10):
                temp = "{0},{1}".format(ti, i)
                if temp in line:
                    ti = temp
                    found = True
                    break
        if not found:
            ti = "{0},{1}".format(ti, 0)  
        file.close()

    t = get_pickle().createInputsAndTargets([ti])

    test_input = t[0]
    test_target = t[1]

    print(test_input)
    print(test_target)

    output = {}

    output["result"] = []
    o = output_result(test_input)
    for key in o:
        output['result'].append({"classifier": key, "error": o[key]})
    output["expected"] = int(list(np.where(test_target[0] == 1)[0])[0])

    print(output)

    dic = {
        0: "Nothing in hand; not a recognized poker hand",
        1: "One pair; one pair of equal ranks within five cards",
        2: "Two pairs; two pairs of equal ranks within five cards",
        3: "Three of a kind; three equal ranks within five cards",
        4: "Straight; five cards, sequentially ranked with no gaps",
        5: "Flush; five cards with the same suit",
        6: "Full house; pair + different rank three of a kind",
        7: "Four of a kind; four equal ranks within five cards",
        8: "Straight flush; straight + flush",
        9: "Royal flush; {Ace, King, Queen, Jack, Ten} + flush"
    }

    final = ""

    return json.dumps(output)

def run():
    nhidden = 22
    arr = []
    for i in range(10):
        temp = None
        temp = PokerHandANN(nhidden)

        err = calc_final_error(temp)
        if(err > 80):
            arr.append(temp.valid_errors)
            file_handler = open("{0}_{1}_{2}.pkl".format(pickle_file_path, nhidden, err), "wb")
            pickle.dump(temp, file_handler)
            file_handler.close()

    f, axes = pl.subplots(len(arr[0]), 1)

    for x in range(len(arr)):
        for i in range(len(arr[x])):
            axes[i].plot(arr[x][i], label='Validation Set Error')

    pl.legend(loc='upper right')

    pl.xlabel('Number of Iterations of {0}'.format(50))
    pl.ylabel('% Error')

    pl.show()

class MQTTBroker():

    """Used as placeholders for all the devices."""

    mqtt = None

    server_in = "hand/server"
    client_out = "hand/client"

    """Location of mqtt broker."""
    broker_url = "localhost"
    broker_port = 1883

    subscription_list = [server_in]

    """Initialization of the server."""
    def __init__(self):

        self.mqtt = MQTT.Client("server")

        self.link_handlers()

        self.mqtt.connect(self.broker_url, self.broker_port)

        self.subcribe_to_list()

        self.handle_loop()

    """Handles all of the linking for mqtt method handlers"""
    def link_handlers(self):
        self.mqtt.on_connect = self.on_connect
        self.mqtt.on_disconnect = self.on_disconnect
        self.mqtt.on_log = self.on_log
        self.mqtt.on_message = self.on_message

    def handle_loop(self):
        while True:
            self.mqtt.loop(.1, 64)

    """On message handler gets called anytime self.mqtt recieves a subscription"""
    def on_message(self, client, userdata, msg):
        payload = str(msg.payload.decode("utf-8"))
        print("{}: {}".format(msg.topic, payload))

        """Any data destined for host from client node"""
        if(msg.topic == self.server_in):
            self.mqtt.publish(self.client_out, str(get_output(payload)))

    """On connect handler gets called upon a connection request"""
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.publish(self.client_out, b'Connect')
            print("Established connection...")
        else:
            print("No connection established, returned error code {}...".format(rc))

    """On disconnect handler gets called upon a disconnect request"""
    def on_disconnect(self, client, userdata, flags, rc = 0):
        print("Disconnected with result code {}".format(rc))

    """Logs any error messages, kind of annoyting because it doesn't provide any information about WHERE the error came from but prevents outright crash"""
    def on_log(self, client, userdata, level, buf):
        print("LOG: {}".format(buf))

    """Helper method to just subscribe to any topic inside of a list"""
    def subcribe_to_list(self):
        for x in self.subscription_list:
            self.mqtt.subscribe(x)

MQTTBroker()

# run()
