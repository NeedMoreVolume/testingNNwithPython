# testingNNwithPython

Please ensure that numpy is already installed on the machine before running or playing with the test provided.

The dataset is hardcoded since this was just a test of an idea that a neural network may be able to figure out how to predict lows and when to buy in the market based on daily open, close, and volume statistics.

The results of the test yield uncertainty as the dataset is small and nearly always yielding smaller numbers as it progresses. The results used to train are not uniquely derrived from the given dataset alone, aiding in the difficulty of the network in being certain.

Future tests are planned to include other algorythms to be utilized for training on the dataset, and a significantly larger dataset.

You can simply just run the script as is, or load your own arrays into the np.array and reconfigure the matricies that are created to operate correctly and not have a value error or an overflow error from the sigmoid function.