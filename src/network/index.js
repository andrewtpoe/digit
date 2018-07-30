const { curry, forEach, range } = require('ramda');

const defaultModel = require('../model.json');

function buildActiveModel(model) {
  return {};
}

const evaluate = curry((activeModel, testData) => {
  console.log('Evaluating current model.');
});

const train = curry((initialModel, trainingData, trainingOptions, callback) => {
  console.log('Training Neural Network.');

  const { epochs = 30, miniBatchSize = 10, stepSize = 3.0 } = trainingOptions;

  const cycles = range(0, epochs);

  let updatedModel = initialModel;

  forEach((cycle) => {
    console.log(`Training cycle ${cycle} complete.`);
    callback(updatedModel);
  }, cycles);

  return updatedModel;
});

/**
 * Initializes the neural network.
 *
 * @param {Object} model the model contains specific parameters used to initialize the network.
 *        If not supplied, the model will be loaded from and saved to `src/model.json
 * @property {Array.<Numbers>} model.layers an array of numbers. The first for this data set
 *           must be 784 to match the number of pixels in the base image.
 * @property {Array.} model.biases an array of arrays representing the biases
 * @property {Array.} model.weights an array of arrays representing the weights
 * @returns
 */
function initializeNetwork(model) {
  console.log('Initializing Neural Network.');

  const activeModel = buildActiveModel(model);

  return {
    evaluate,
    train: train(activeModel),
  };
}

module.exports = { initializeNetwork };
