##CNN from scratch 
## there is an assumption that each example is always 3D tensor (RGB or greyscale)

#### Features :
'''
   -- Different etas for each layer
   -- Different pooling functions
   -- b2b convolutional layers (no pooling in between)
'''

#### Todos :
'''
  -- Add Striding
  
  
   '''


### if the activation is softmax, then so is the cost

import random
import numpy as np


class sigmoid():
  @staticmethod
  def func(z):
      return 1.0 / (1.0 + np.exp(-z))

  @staticmethod
  def prime(z):
      return sigmoid.func(z) * (1.0 - sigmoid.func(z))



class tanh():
  @staticmethod
  def func(z):
      return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

  @staticmethod
  def prime(z):
      return 1 - np.square(tanh.func(z))


class ReLu():
  @staticmethod
  def func(z):
      return z if z > 0 else 0

  @staticmethod
  def prime(z):
      return 1 if z > 0 else 0


class softmax():

  @staticmethod
  def func(z):

    # for k in range(np.shape[-1]):
    #   theVector = np.sum(np.exp(z[:,k]))
    #   activations[:,k] = np.exp(z[:,k]) / theVector
    exp = np.exp(z)
    theVector = np.sum(exp,axis=0)
    activations = exp / theVector

    return activations



class loglikedlihood():
    
    @staticmethod
    def fn(a, y):
        theKey =  np.array([0,1,2,3,4,5,6,7,8,9]).reshape(10,1)
        a = -1 * np.log(a)
        return   a  * mat.mult(np.transpose(y) , theKey)
    
    @staticmethod      
    def delta(z , a, y):

        return (a-y)




class QuadraticCost():

    @staticmethod
    def fn(a, y):
 
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):

        return (a-y) * sigmoid.prime(z)


class CrossEntropyCost():

    @staticmethod
    def fn(a, y):

        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z , a, y):

        return (a-y)






class ConvPoolLayer():
  ## assumption that the kernel is less than the input in size
   def __init__(self, PrevlayerSize, Kernels, PoolingBool=0, PoolingMaps=None, activationFunction=ReLu): 
          ''' PrevlayerSize is the size of the previous layer,
              Kernels is a list of all the kernels in the layer,
              if the layer contains Pooling or not and if it does
              PoolingMaps is a list of tuples for each map of pooling size  and the pooling function which
              has to return a scalar np.max() 
               
               '''

    for k in Kernels:
      if k[-1] != PrevlayerSize[-1]:
        raise Exception("Kernel " + str(Kernels.index(k)) + " is not the right size")    


    self.PrevlayerSize = PrevlayerSize  
    self.PoolingBool = PoolingBool
    self.PoolingMaps = PoolingMaps
    self.kernelsSize = Kernels
    self.Kernels = [] ##tuple of (bias, weights) for each feature map
    self.activationFunc = activationFunction

    ## Will store the weighted input for the last forward pass call
    self.weightedInputs = None
    self.errors = None

    ### Init some Kernels
    for k in Kernels:
      self.Kernels.append((np.random.randn() * 0.001, np.random.randn(k[0] , k[1] , k[2]) * 0.001 ))


   def poolingMap(self, batchActivations):
        poolSize = self.poolingMaps
        listOfMapsByPool = [] ## In this case a map is a 4D tensor of a kernel map for the given batch
        # sizeOfPrev = batchActivations.shape[:-1]


        ## iterating over each kernel
        
        for pool in poolSize:
          poolIndex = poolSize.index(pool)
          listOfMapsByImage = []
          poolingFunction = pool[1]
          poolX = pool[0][0] 
          poolY = pool[0][1]   ### 
          ###this is potentially 

          for inputCount in range(batchActivations.shape[-1]):
              input = batchActivations[:, :, poolIndex - 1 , inputCount]
              map  = np.zeros(poolX, poolY, 1,1)

              ## This is with assumption that no overlapping is done in pooling
              for x in range(math.floor(input.shape[0] / poolX ) ):
              for y in range( math.floor( input.shape[1] / poolY ) ):
                  map[x][y][0][0]  =  poolingFunction(input[2 * x : 2 * x + poolX,
                                                            2 * y : 2 * y + poolY])             
                  
                  
              listOfMapsByImage.append(map)


        ### This may fail
        listOfMapsByKernel.append(np.concat(listOfMapsByImage, axis= 3))
        
        return np.concat(listOfMapsByKernel, axis =2) 


  def feedForward(self, inputActivation): ## input Activation is guranteed to be 4D tensor
    Kernels = self.Kernels
    kernelsSize = self.kernelsSize
    listOfMapsByKernel = [] ## In this case a map is a 4D tensor of a kernel map for the given batch
    sizeOfPrev = inputActivation.shape[:-1]
    weightedInputsByKernel = []


    ## iterating over each kernel
    
    for k in kernelsSize:
      kernel = Kernels[KernelsSize.index(k)][1]
      bias = Kernels[KernelsSize.index(k)][0]
      k = k[:-1]
      listOfMapsByImage = []
      weightedInputsByImage = []

      for inputCount in range(inputActivation.shape[-1]):
        input = inputActivation[:,:,:,inputCount]
        map  = np.zeros(sizeOfPrev[0] - k[0] + 1, sizeOfPrev[1] - k[1] + 1, 1,1)
        weightedInputs


        for x in range(sizeOfPrev[0] - k[0] + 1):
          for y in range(sizeOfPrev[1] - k[1] + 1):

            z =  np.correlate(input[x:x+k[0],y:y+k[1],:].flatten() , 
                            kernel.flatten()) + bias
            map[x][y][0][0] = self.activationFunc.func(z)
            
            
              
        listOfMapsByImage.append(map)

        ### This may fail
      listOfMapsByKernel.append(np.concat(listOfMapsByImage, axis= 3))
    
    batchActivations = np.concat(listOfMapsByKernel, axis =2) 
    if self.PoolingBool == 0
      return batchActivations
    else:
      return self.poolingMap(batchActivations)


  def backprop(self, eta, batchSize, batchLabels, inputActivation, deeperDeltas, deeperWeights):
      ### BP1 does not apply to Conv Pool Layer 


    ### BP 2 
    if self.PoolingBool == 1 :

      Deltas = np.sum( , axis=1)

      
    


    else:


      inputActivation = inputActivation.reshape(self.PrevlayerSize,inputActivation.shape[-1])
      




class FullyConnected ():


    def __init__(self, PrevlayerSize, size, activationFunction=ReLu, costFunction=None):  
      ''' PrevlayerSize is a tuple of the previous layer, 
          size of the layer    '''
        self.PrevlayerSize = 1
        for k in PrevlayerSize[:-1]:
          self.PrevlayerSize *= k
          self.weights =  np.random.randn(size, PrevlayerSize])  * 0.0001
          self.biases =   np.random.randn(size,)  * 0.0001
          self.activationFunc = activationFunction


        ###Forbackprop
        self.activations = None
        self.weightedInputs = None
        self.errors = None
        self.costFunction = costFunction

        
    def feedForward(self, inputActivation): 

      # if len(inputActivation.shape) > 2 :
      #     ## then we have a 4D tensor from a ConvLayer

      inputActivation = inputActivation.reshape(self.PrevlayerSize,inputActivation.shapes)
      z = np.matmul(self.weights, inputActivation) + self.biases
      self.weightedInputs = z
      self.activations = (self.activationFunc.func(z))
      
      return self.activations

      #   for inputIndex in range(inputActivation.shape[-1]):
      #     input = inputActivation[:,:,:,inputIndex]
      #     z = np.matmul(self.weights, input.flatten()) + self.biases
      #     activationl.append(self.activation(z))
      #   return np.concat(activationl, axis=len(inputActivation.shape) - 1)
      # else :
      #   z = np.matmul(self.weights, inputActivation) + self.biases
      #   return (self.activation(z))


   def backprop(self, eta, batchSize, batchLabels, inputActivation, deeperDeltas, deeperWeights):
      inputActivation = inputActivation.reshape(self.PrevlayerSize,inputActivation.shape[-1])
      NablaW = None
      Deltas = None
      NablaB = Deltas 



      #### BP1 / BP2 ######## 
      ## BP2
      if self.CostFunction == None:
          
        Deltas =   (np.matmult(numpy.transpose(deeperWeights) , deeperDeltas))   *  self.activationFunc.prime(self.weightedInputs)
          

      else :
        ##BP1  : this is for all cost and all activations incl. softmax

        Deltas = self.CostFunction.delta( self.weightedInputs , self.activation , batchLabels)
      
    ##BP3/4    
        NablaW = np.rot90(inputActivation[:,:,np.newaxis], k = 1, axes=(0,2))  * np.repeat( Deltas[:,:,np.newaxis], self.PrevlayerSize , axis =2)
        NablaB = Deltas
        
        
        self.weights = self.weights -   ( eta * np.sum(NablaW, axis=2)  / batchSize )
        self.biases = self.biases -   ( eta * np.sum(NablaB, axis=1)   / batchSize )
        
        return self.weights, self.biases
        
    

  



class Network(object):

    ## the first layer is an input layer

    def __init__(self, layers, etas=3.0, epochs=1, bacthSize=10):  ## layers is a list of layers with the first being a tuple (i.e. the dimension of the input in np)
        ## etas is a list of etas for each layer otherwise just a float
        ## Sets up the layers of the network
        self.numlayers = len(layers)
        self.layers = layers

        if type(etas) == list:
          if lent(etas) != len(layers) - 1:
            raise Exception("Not enough etas")

        self.etas = etas
        self.m = bacthSize
        self.n = 1

        ## backprop to find gradient of C_x
    def backprop(self, batch):
        nq = self.numlayers - 1  ## nq  layers w/o the input

        ## Feedforward to find all the activation
        batchImages = np.concat([x.reshape(self.layers[0][0], 
                                          self.layers[0][1],
                                         self.layers[0][2], 1) for x, y in batch], axis =-1) ## 4Dtensor 
        batchLabels = np.hstack([y for x, y in batch])
        batchActivations = [batchImages]
        batchlistofZs = []
        ### for each layer, we feed forward

        for layerIndex in range(nq): 
          layer = self.layers[layerIndex + 1] 
          batchActivations.append(layer.feedForward(deeperLayerWeights))


12222222sdfdsfds


        ##########  ***** Back Propogatioon *******######### 
        ## The last layer
        #  eta, batchSize, batchLabels, inputActivation, deeperDeltas, deeperWeights 

        deeperWeights, deeperBiases = self.layers[-1].backprop(etas[-1], self.m, batchLabels=batchLabels,  
                                  batchActivations[-2], deeperDeltas=None, deeperWeights=None)
                                 
                                 prevLayerActivation, 
    deeperLayerWeights, deeperLayerbiases, deeperLayerErrors)
        

        ##Every other layer
        for layerIndex in range(2, nq + 1): 
          self.layers[layerIndex * -1].backprop()


        



    ## the update rule execution for a given mini batch
    def updateParameters(self, batch):
        self.backprop(batch)
 

    def SGD(self, batchSize, eta, epochs,
            trainingSet):  ## the epoch could later be made into a float
        leng = len(trainingSet)

        if leng == 0:
            return
        self.eta = eta
        self.m = batchSize
        m = self.m

        for T in range(0, epochs):
            ## shuffle up the training example and divy them up into batches of the size batchSize
            random.shuffle(trainingSet)
            setOfBatches = [trainingSet[q:q + m] for q in range(0, leng, m)]

            for Batch in setOfBatches:

                self.updateParameters(Batch)




    ## returns the ANN's answer to the input image x
    def feedForward(self, batch):
      nq = self.numlayers - 1
     
      batchImages = np.concat([x.reshape(self.layers[0][0], 
                                          self.layers[0][1],
                                         self.layers[0][2], 1) for x, y in batch], axis =-1)
      

      batchLabels = [y for x, y in batch]
      batchActivations = [batchImages]


      for layerIndex in range(nq): 
        layer = self.layers[layerIndex + 1] 
        batchActivations.append(layer.feedForward(batchActivations[layerIndex]))

      return batchActivations[-1], batchLabels

    def eval(self, testImages, batchSize): ## measure accuracy
      accuracy = 0
      for interval in range(0, len(testImages), batchSize):
        x,y = self.feedForward(testImages[k:k+batchSize])
        answers =  [np.argmax(x[:,q])  for q in range(batchSize):]
        answers = zip(answers, y)
        accuracy += sum(int(x == y) for (x, y) in answers)
      return accuracy


