from numpy import exp, array, random, dot
from math import atan2,degrees


class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 5* random.random((3, 1)) - 3

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        
     
        return 1 / (1 + exp(-x))
    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print "Random starting synaptic weights: "
    print neural_network.synaptic_weights

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[512,500,-79.4482],[331,690,-40.6525],[531,382,-106.201],[712,581,-32.9838],[823,615,-34.1145],[569,668,-13.3925],[494,319,156.7426],[760,266,-152.241],[695,286,-141.546]])
    training_set_outputs = array([[2,2,2,1,1,1,3,3,3]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print "New synaptic weights after training: "
    print neural_network.synaptic_weights

    # Test the neural network with a new situation.
    print "Considering new situation 1: "
    print neural_network.think(array([675,736,-45.46962702]))

    print "Considering new situation 1: "
    print neural_network.think(array([577,678 ,-34.00440531]))

    print "Considering new situation 1: "
    print neural_network.think(array([569,668,-13.39249775]))

    print "Considering new situation 2: "
    print neural_network.think(array([517,274,-161.565 ]))

    print "Considering new situation 2: "
    print neural_network.think(array([515,226,-129.806 ]))

    print "Considering new situation 2: "
    print neural_network.think(array([546,306, -116.5650512]))
    


    print "Considering new situation 3: "
    print neural_network.think(array([715, 698, -45.7008]))

    print "Considering new situation 3: "
    print neural_network.think(array([399,308,161.2270269]))


    print "Considering new situation 3: "
    print neural_network.think(array([556,188,153.0584212]))

 
 


