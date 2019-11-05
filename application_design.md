# Machine Learning Application Design

## Poker Hand Classification

1. Variable initialization and constants

    nhidden = 52
    beta = 1
    momentum = .9
    outtype = 'softmax'
    w = .1
    num_iterations = 7500
    randomize = False
    normalize = False # most likely not needed

2. Read in data and create testing, training, and validation sets.

    data = file.readlines()
    random.shuffle(data)
    data = data[0:10000]
    
    testing_validation_split = int(len(data) * .2)

    self.testing = data[:-testing_validation_split]
    data = data[:-testing_validation_split or None]
    self.validation = data[:-testing_validation_split]
    (self.inputs, self.targets) = self.createInputsAndTargets(data[0:len(data) - testing_validation_split])
    (self.validation_input, self.validation_target) = self.createInputsAndTargets(self.validation)

3. Initialize Weights

    self.weights1 = (np.random.rand(self.nin+1,self.nhidden)-0.5)*2/np.sqrt(self.nin)
    self.weights2 = (np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)

4. Train with early stopping, num_iterations each time

    self.earlystopping(self.inputs, self.targets, self.validation_input, self.validation_target, self.w, self.num_iterations)

