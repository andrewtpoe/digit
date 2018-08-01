const { curry, forEach, range } = require('ramda');

const { buildActiveModel } = require('./buildActiveModel');

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
 * Initializes the neural network. Hyperparams are adjusted via the constants file.
 *
 * @returns {object} An object containing methods you can use to run the neural network.
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
