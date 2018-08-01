const { curry, forEach, range } = require('ramda');

const { getModel } = require('./getModel');

const evaluate = curry((testData, model) => {
  console.log('Evaluating current model.');
});

const train = curry((trainingData, callback) => {
  console.log('Begining training sequence.');

  const initialModel = getModel();

  const { epochs } = initialModel;
  const cycles = range(0, epochs);

  let updatedModel = initialModel;

  forEach((cycle) => {
    console.log(`Training cycle ${cycle} complete.`);
    callback(updatedModel);
  }, cycles);

  return updatedModel;
});

/**
 * Initializes the neural network. HyperParams are adjusted via the constants file.
 *
 * @returns {object} An object containing methods you can use to run the neural network.
 */
function initializeNetwork() {
  console.log('Initializing Neural Network.');

  return {
    evaluate,
    train,
  };
}

module.exports = { initializeNetwork };
