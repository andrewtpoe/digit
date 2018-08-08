const { compose, curry, last, sum, zip, zipWith } = require('ramda');

const { reduceWithIndex } = require('../utils');

/**
 * Evaluates the index of the highest activation. This is used as the predicted value.
 *
 * @param {array.<Number>} outputActivations the activations of the neurons in the output layer.
 * @returns A prediction for this run of the network.
 */
function evaluateOutputActivations(outputActivations) {
  const prediction = reduceWithIndex(
    (acc, activation, index) => {
      if (activation >= acc.activation) {
        return { activation: activation, value: index };
      }
      return acc;
    },
    { activation: 0, value: 0 },
    outputActivations,
  );
  return prediction.value;
}

/**
 * Generates a neuron's final activation
 *
 * This function implements the sigmoid formula:
 *
 * σ(z) ≡ 1 / (1 + e^−z)
 *
 * Reference:
 * http://neuralnetworksanddeeplearning.com/chap1.html#sigmoid_neurons
 *
 * @param {number} weightedInput
 * @returns
 */
function sigmoid(weightedInput) {
  return 1.0 / (1.0 + Math.exp(-weightedInput));
}

/**
 * TODO: add docs
 *
 * @param {*} neuronWeights
 * @param {*} previousLayerActivations
 * @param {*} bias
 */
function generateNeuronActivation(
  neuronWeights,
  previousLayerActivations,
  [bias],
) {
  return sigmoid(
    sum(zipWith((w, a) => w * a, neuronWeights, previousLayerActivations)) +
      bias,
  );
}

/**
 * Generates activations for the next layer in the network.
 *
 * This function implements this formula:
 *
 * (a,l,j) = σ( ∑,k((w,l,j,k) (a,l−1,k)) + (b,l,j))
 *
 * where:
 * - a represents the activation of a neuron
 * - w represents the weight of the connection going into the neuron
 * - b represents the bias of a neuron
 *
 * The syntax (w,l,j,k) is a method of specifying which neuron or connection is being specified.
 * - l represents the current layer
 *   - l-1 represents the current layer - 1, or the previous layer
 * - j represents the neuron in the current layer
 * - k represents the neuron in the previous layer (l-1)
 *
 * For example, given the following values for a network with layers [2, 1]:
 * - the activations for layer 0 are [0.1, 1.0]
 * - the weights between layers 0 and 1 are [[1, 2]]
 * - the bias for layer 1 is [[.9]]
 *
 * - (a, 1, 0) = sigmoid(((1 * 0.1) + (1.0 * 2)) + 0.9)
 * - (a, 1, 0) = sigmoid((0.1 + 2) + 0.9)
 * - (a, 1, 0) = sigmoid(2.1 + 0.9)
 * - (a, 1, 0) = sigmoid(3.0)
 *
 * Thus the output layer's activations would be [sigmoid(3.0)]. If there were multiple
 * neurons in layer 1, this would be repeated for each neuron in the layer.
 *
 * Reference:
 * http://neuralnetworksanddeeplearning.com/chap2.html#warm_up_a_fast_matrix-based_approach_to_computing_the_output_from_a_neural_network
 *
 * @param {*} layerInputWeights
 * @param {*} previousLayerActivations
 * @param {*} biases
 * @returns
 */
function generateLayerActivations(
  layerInputWeights,
  previousLayerActivations,
  biases,
) {
  return reduceWithIndex(
    (accumulator, bias, neuronIndex) => {
      const neuronActivation = generateNeuronActivation(
        neuronIndex,
        layerInputWeights[neuronIndex],
        previousLayerActivations,
        bias,
      );
      return accumulator.concat([neuronActivation]);
    },
    [],
    biases,
  );
}

/**
 * Feeds an array of input neuron activations forward through the network to determine the
 * output neuron activations.
 *
 * @param {object} model The model being used to generate the predictions
 * @property {array.<array>} model.biases A nested array of arrays representing the biases
 *           for each neuron.
 * @property {array.<array>} model.weights A nested array of arrays representing the weights
 *           for each connection between each layer of neurons.
 * @param {array.<number>} inputActivations A 784 element long array of pixel inputActivations
 * @returns An array of arrays each containing numbers representing the pixel activations for
 *          each layer in the network.
 */
const feedForward = curry(({ biases, weights }, inputActivations) => {
  const layers = zip(weights, biases);

  return reduceWithIndex(
    (accumulator, [layerInputWeights, layerBiases], index) => {
      const previousLayerActivations = accumulator[index];

      return accumulator.concat([
        generateLayerActivations(
          layerInputWeights,
          previousLayerActivations,
          layerBiases,
        ),
      ]);
    },
    [inputActivations],
    layers,
  );
});

/**
 * TODO: add docs
 *
 * @param {*} model
 * @param {*} inputActivations
 * @returns
 */
function predictValue(model, inputActivations) {
  return compose(
    evaluateOutputActivations,
    last,
    feedForward(model),
  )(inputActivations);
}

/**
 * TODO: add docs
 *
 * @param {*} model
 * @param {*} image
 * @returns
 */
const evaluateImage = curry((model, image) => {
  const [inputActivations, actualValue] = image;
  const predictedValue = predictValue(model, inputActivations);

  return {
    accurate: actualValue === predictedValue,
    actualValue,
    predictedValue,
  };
});

module.exports = { evaluateImage };
