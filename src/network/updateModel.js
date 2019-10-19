const { compose } = require("ramda");

const { chunkTuples, shuffle } = require("../utils");

/**
 *
 *
 * @param {*} modelChanges
 * @param {*} model
 */
function applyModelChanges(modelChanges, model) {
  // applies the changes to the current model
}

/**
 *
 *
 * @param {*} miniBatch
 * @param {*} model
 * @returns {}
 */
function generateModelChanges(miniBatch, model) {
  // run the backprop algorithm
  // generate an array of changes in the biases and weights
}

/**
 *
 *
 * @param {*} miniBatches
 * @param {*} model
 * @returns {*} the new model with all updates applied
 */
function processMiniBatches(miniBatches, model) {
  // for each mini-batch, generate model changes
  // - run the backprop algorithm
  // - generate an array of changes in the biases and weights
  // - apply the changes to the current model
  return model;
}

/**
 * Breaks the training data down in a randomized array of mini batches.
 * Say the initial training data is 50,000 elements (each a tuple), and
 * the batchSize is 10. This will return an array of 5,000 arrays, each
 * containing 10 tuples.
 *
 * @param {array.<array,number>} trainingData An array containing two element arrays ("Tuples").
 *        Each element contains:
 *          0. the pixel activations of the image.
 *          1. the numeric value the image represents
 * @param {object} model the model for the neural network.
 * @returns {array.<array>} grouped mini-batches of data.
 */
function generateMiniBatches(trainingData, { batchSize }) {
  return compose(
    chunkTuples(batchSize),
    shuffle
  )(trainingData);
}

/**
 *
 *
 * @param {array.<array,number>} trainingData An array containing two element arrays ("Tuples").
 *        Each element contains:
 *          0. the pixel activations of the image.
 *          1. the numeric value the image represents
 * @param {object} model the model for the neural network.
 * @returns {object} A complete model object, updated to reflect the training data and
 *          hyperparams provided
 */
function updateModel(trainingData, model) {
  const miniBatches = generateMiniBatches(trainingData, model);
  return processMiniBatches(miniBatches, model);
}

module.exports = { updateModel };
