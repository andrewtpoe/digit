const { compose, isEmpty, not, reduce } = require('ramda');

const { BATCH_SIZE, EPOCHS, LAYERS, LEARNING_RATE } = require('../constants');

const { mapWithIndex } = require('../ramda-utils');

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
 * @param {array.<number>} layers An array of numbers indicating how many neurons
 *        should be in each layer.
 * @param {array.<array>} matrix An array of arrays, each as long as the number in
 *        the same index of the layers array
 * @returns {boolean}
 */
function isMatrixValid(layers, matrix) {
  if (layers.length !== matrix.length) {
    return false;
  }

  // Evaluates each layer of the matrix to ensure it contains the expected number of values.
  // TODO: The logic here will most likely need adjustments as development continues.
  return compose(
    not,
    isEmpty,
    mapWithIndex(
      (matrixLayer, layerNum) => matrixLayer.length === layers[layerNum],
    ),
  )(matrix);
}

/**
 * TODO:
 *
 * @param {*} layers
 * @returns
 */
function buildDefaultMatrix(layers) {
  return reduce(
    (accumulator, layer) => accumulator.concat([new Array(layer).fill(0)]),
    [],
    layers,
  );
}

/**
 * TODO:
 *
 * @param {*} layers
 * @param {*} matrix
 * @returns
 */
function buildMatrix(layers, matrix) {
  return isMatrixValid(layers, matrix) ? matrix : buildDefaultMatrix(layers);
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
    biases: buildMatrix(layers, biases),
    epochs,
    layers,
    learningRate,
    weights: buildMatrix(layers, weights),
  };
}

module.exports = { getModel };
