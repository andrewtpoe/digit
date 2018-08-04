const { compose, filter, isEmpty, slice } = require('ramda');

const { BATCH_SIZE, EPOCHS, LAYERS, LEARNING_RATE } = require('../constants');

const { reduceWithIndex } = require('../ramda-utils');

/**
 * Loads the saved model from the file system.
 *
 * @returns {object} either the saved model or an empty object
 */
function loadModel() {
  try {
    return require('../model.json');
  } catch (error) {
    return {};
  }
}

/**
 * Evaluates a layer for the expected number of elements.
 *
 * @param {array.<array>} matrixLayer a single layer of the matrix representing a layer
 *        in either the weight or biase matrix.
 * @param {number} layerLength the number of elements expected in a layer.
 * @returns A boolean.
 */
function layerInvalid(matrixLayer, layerLength) {
  return matrixLayer.length !== layerLength;
}

/**
 * Evaluates the matrixLayer for valid elements.
 *
 * @param {function} neuronLengthFn Evaluates a neuron for the number of elements.
 * @param {array.<array>} matrixLayer a single layer of the matrix representing a layer
 *        in either the weight or biase matrix.
 * @param {array.<number>} layers An array of numbers indicating how many neurons
 *        should be in each layer.
 * @param {number} layerNum the current layer of the matrix being processed.
 * @returns An array of boolean flags.
 */
function neuronsInvalid(neuronLengthFn, matrixLayer, layers, layerNum) {
  return reduceWithIndex(
    (accumulator, neuron) => {
      if (neuron.length !== neuronLengthFn(layers, layerNum)) {
        return accumulator.concat(true);
      }
      return accumulator;
    },
    [],
    matrixLayer,
  );
}

/**
 * Evaluates if a matrix (biases or weights) is valid.
 *
 * @param {function} neuronLengthFn Evaluates a neuron for the number of elements.
 * @param {array.<number>} layers An array of numbers indicating how many neurons
 *        should be in each layer.
 * @param {array.<array>} matrix An array of arrays, each representing some relevant
 *        value for the network.
 * @returns {boolean}
 */
function isMatrixValid(neuronLengthFn, layers, matrix) {
  if (matrix.length !== layers.length - 1) {
    return false;
  }

  return compose(
    isEmpty,
    filter((v) => v),
    // returns an array of booleans. a true value represents an invalid layer or neuron.
    reduceWithIndex((accumulator, matrixLayer, layerNum) => {
      return accumulator
        .concat(layerInvalid(matrixLayer, layers[layerNum + 1]))
        .concat(neuronsInvalid(neuronLengthFn, matrixLayer, layers, layerNum));
    }, []),
  )(matrix);
}

/**
 * Builds a default matrix to represent either the weights or biases.
 *
 * @param {function} neuronLengthFn Evaluates a neuron for the number of elements.
 * @param {array.<number>} layers An array of numbers indicating how many neurons
 *        should be in each layer.
 * @returns A deeply nested array of arrays initialized with default values.
 */
function buildDefaultMatrix(neuronLengthFn, layers) {
  return reduceWithIndex(
    (accumulator, layer, index) => {
      const newLayer = new Array(layers[index + 1]).fill(
        // V1 of this neural network just initializes with 0 values. This is extremely basic,
        // and may eventually be changed to provide better initializations.
        new Array(neuronLengthFn(layers, index)).fill(0),
      );
      return accumulator.concat([newLayer]);
    },
    [],
    // Both the weights and biases matrixes start with layer index 1 of the network.
    // The first layer is specific to the input vector.
    slice(1, Infinity, layers),
  );
}

/**
 * Evaluates the provided matrix for validitiy and builds a default if needed.
 *
 * @param {function} neuronLengthFn Evaluates a neuron for the number of elements.
 * @param {array.<number>} layers An array of numbers indicating how many neurons
 *        should be in each layer.
 * @param {array.<array>} [matrix=[]] An array of arrays, each representing some relevant
 *        value for the network.
 * @returns Either the valid provided matrix, or a default matrix.
 */
function buildMatrix(neuronLengthFn, layers, matrix = []) {
  return isMatrixValid(neuronLengthFn, layers, matrix)
    ? matrix
    : buildDefaultMatrix(neuronLengthFn, layers);
}

/**
 * Validates the biases matrix based on the layers, returns either the provided biases
 * matrix or a valid default matrix. Each layer in a biases matrix contains the biase
 * values for each neuron in the respective layer.
 *
 * Example:
 * layers = [784, 30, 10];
 * biases = [
 *   The first layer of neurons in this biases matrix will be 30 elements long,
 *   one for each neuron in the second layer of the network.
 *   [
 *     Each element in this layer will contain the biase value for a single neuron.
 *     [ ... ],
 *     ...
 *   ],
 *   This array would have 10 elements, each containing a single value.
 *   [
 *     [ ... ],
 *     ...
 *   ],
 * ]
 *
 * @param {array.<number>} layers An array of numbers indicating how many neurons
 *        should be in each layer.
 * @param {array} biases A nested array of arrays representing the biases for each
 *        neuron.
 * @returns A valid biases matrix
 */
function buildBiases(layers, biases) {
  // This determines how many values are required for each neuron in each layer of a biases matrix.
  const neuronLengthFn = () => 1;
  return buildMatrix(neuronLengthFn, layers, biases);
}

/**
 * Validates the weights matrix based on the layers, returns either the provided weights matrix or a
 * valid default matrix. Each layer in a weights matrix contains the weight values
 * for the connection between a neuron in the current layer and every neuron in the previous layer.
 *
 * Example:
 * layers = [784, 30, 10];
 * weights = [
 *   The first layer of neurons in this weight matrix will be 30 elements long,
 *   one for each neuron in the second layer of the network.
 *   [
 *     Each element in this layer will contain weight values for the connection between every
 *     neuron in layer 1 of the network. In this example, this array would have 784 values.
 *     [ ... ],
 *     ...
 *   ],
 *   This array would have 10 elements, each containing 30 values.
 *   [
 *     [ ... ],
 *     ...
 *   ],
 * ]
 *
 * @param {array.<number>} layers An array of numbers indicating how many neurons
 *        should be in each layer.
 * @param {array} weights A nested array of arrays representing the weights for each
 *        connection between each layer of neurons.
 * @returns A valid weight matrix
 */
function buildWeights(layers, weights) {
  // This determines how many values are required for each neuron in each layer of a weights matrix.
  const neuronLengthFn = (layersArr, index) => layersArr[index];
  return buildMatrix(neuronLengthFn, layers, weights);
}

/**
 * Loads the model.
 *
 * @returns Either the persisted model, or a valid default model.
 */
function getModel() {
  console.log('Loading the model.');

  const {
    batchSize = BATCH_SIZE,
    biases,
    epochs = EPOCHS,
    layers = LAYERS,
    learningRate = LEARNING_RATE,
    weights,
  } = loadModel();

  return {
    batchSize,
    biases: buildBiases(layers, biases),
    epochs,
    layers,
    learningRate,
    weights: buildWeights(layers, weights),
  };
}

module.exports = { buildBiases, buildWeights, getModel };
