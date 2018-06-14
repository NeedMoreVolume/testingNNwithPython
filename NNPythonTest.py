#Daniel Pivalizza
#Testing Neural Networks with custom market data matricies.

#We will send them through the network to see if it can figure out if it should buy on the next day,
#   or sell based on my input for when to buy and sell.
#We will only feed 4 daily sets of data. -1=sell, 0=do nothing, and 1=buy. Dataset is a matrix that represents format
#   [open,close,volume], where 1 = increase, 0 = no change, and -1 = decrease. The data used is from 6/10/2018-6/14/2018
#   from Yahoo on the BTCUSD market.
#We will run a few different alpha values, since we are testing this neural network to see how it handles
#   daily market data and learning the market.

#import numpy for linear algebra functions
import numpy as np

#A parameter to modify the weights in the network to see if we can help speed the process up is utilized.
weightScalars = [0.005,0.02,0.25,1,2,5]

#number of nodes to utilize in the network and initialize for synapses to operate appropriately, can use
#	less for this example, but 32 makes it quicker.
size = 32

# Calculate the sigmoid function
def sigmoidFunction(x):
    output = 1/(1+np.exp(-x))
    return output

# Calculate the derivative of the output of the sigmoid function.
def sigmoidDerivitaveOutput(output):
    return output*(1-output)

dataset = np.array([[-1,-1,1],[-1,0,-1],[0,-1,1],[-1,-1,1]])
desiredOutput = np.array([[-1],[1],[0],[1]])
for weightScalar in weightScalars:
    print ("\nTraining With Alpha:" + str(weightScalar))
    np.random.seed(1)

    # Randomly initialize a synapse to hold our dataset, number of inputs in each matrix and the arbitrary
    #   node size declared under the weightScalars, and another to hold the desiredOutput to train the network to.
    s0 = 2*np.random.random((3,size)) - 1
    s1 = 2*np.random.random((size,1)) - 1

    for j in xrange(60000):

        # Feed the dataset into the datalayer, and through our two synapses to train the network.
        dataLayer = dataset
        resultLayer1 = sigmoidFunction(np.dot(dataLayer,s0))
        resultLayer2 = sigmoidFunction(np.dot(resultLayer1,s1))

        # Calculate the error from the result and the target
        resultLayer2_err = resultLayer2 - desiredOutput

        if (j% 10000) == 0:
            print ("Calculated Error after "+str(j)+" iterations:" + str(np.mean(np.abs(resultLayer2_err))))

        # Calculate the change based on the error and the derivative of the resulting 2nd layer,
        #    such that if the error is small, the change needed is also small
        resultLayer2_delta = resultLayer2_err * sigmoidDerivitaveOutput(resultLayer2)

        # Calculate the error of the 1st layer based on the change of the 2nd layer, and the output synapse.
        resultLayer1_err = resultLayer2_delta.dot(s1.T)
        
        # Calculate the change based on the error and the derivative of the resulting 1nd layer,
        #    such that if the error is small, the change needed is also small
        resultLayer1_delta = resultLayer1_err * sigmoidDerivitaveOutput(resultLayer1)

        #Scale the change with our arbitrary weights to attempt to speed up the process and converge faster.
        s1 -= weightScalar * (resultLayer1.T.dot(resultLayer2_delta))
        s0 -= weightScalar * (dataset.T.dot(resultLayer1_delta))

    print("Final result layer")
    print(resultLayer2)
