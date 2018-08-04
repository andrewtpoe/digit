const { compose, isEmpty, not, slice } = require('ramda');

const { BATCH_SIZE, EPOCHS, LAYERS, LEARNING_RATE } = require('../constants');

const { mapWithIndex, reduceWithIndex } = require('../ramda-utils');

const BIASES = 'biases';
const WEIGHTS = 'weights';

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
 * Evaluates if a matrix (biases or weights) is valid.
 *
 * @param {function} layersLengthFn Evaluates a layer in the layers array for the
 *        correct number of elements.
 * @param {array.<number>} layers An array of numbers indicating how many neurons
 *        should be in each layer.
 * @param {array.<array>} matrix An array of arrays, each as long as the number in
 *        the same index of the layers array
 * @returns {boolean}
 */
function isMatrixValid(layerLengthFn, layers, matrix) {
  if (matrix.length !== layers.length - 1) {
    return false;
  }

  // Evaluates each layer of the matrix to ensure it contains the expected number of values.
  return compose(
    not,
    isEmpty,
    mapWithIndex(
      (matrixLayer, layerNum) =>
        matrixLayer.length === layerLengthFn(layers, layerNum),
    ),
  )(matrix);
}

/**
 * TODO:
 *
 * @param {*} layersLengthFn
 * @param {*} layers
 * @returns
 */
function buildDefaultMatrix(layersLengthFn, layers) {
  return reduceWithIndex(
    (accumulator, layer, index) => {
      const newLayer = new Array(layers[index + 1]).fill(
        new Array(layersLengthFn(layers, index)).fill(0),
      );
      return accumulator.concat([newLayer]);
    },
    [],
    slice(1, Infinity, layers),
  );
}

function buildMatrix(layerLengthFn, layers, matrix) {
  return isMatrixValid(layerLengthFn, layers, matrix)
    ? matrix
    : buildDefaultMatrix(layerLengthFn, layers);
}

/**
 * TODO:
 *
 * @param {*} layers
 * @param {*} biases
 * @returns
 */
function buildBiases(layers, biases) {
  // The biases should be an array of arrays of arrays.
  // EX:
  // layers = [784, 30, 10];
  // biases = [
  //   Starting with layer index 1, each array's length should match the value.
  //   In this example, this array would have 30 elements.
  //   [
  //     Each internal array should contain a single element, the bias for that neuron.
  //     [],
  //   ],
  //   [
  //     [],
  //   ],
  // ]
  return buildMatrix(() => 1, layers, biases);
}

/**
 * TODO:
 *
 * @param {*} layers
 * @param {*} weights
 * @returns
 */
function buildWeights(layers, weights) {
  // The wieghts should be similar, but one more nested array.
  // EX:
  // layers = [784, 30, 10];
  // weights = [
  //   Starting with layer index 1, each array's length should match the value.
  //   In this example, this array would have 30 elements.
  //   [
  //     Each internal array should container the number of elements from the layer before,
  //     these are the weights input to each neuron. In this example, this array would have 784
  //     values.
  //     [ ],
  //   ],
  //   [
  //     [],
  //   ],
  // ]
  return buildMatrix((layersArr, index) => layersArr[index], layers, weights);
}

/**
 * TODO:
 *
 * @returns
 */
function getModel() {
  console.log('Loading the model.');

  const {
    batchSize = BATCH_SIZE,
    biases = [],
    epochs = EPOCHS,
    layers = LAYERS,
    learningRate = LEARNING_RATE,
    weights = [],
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
