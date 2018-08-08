const { compose, curry, filter, forEach, range, reduce } = require('ramda');

const { noop } = require('../utils');

const { getModel } = require('./getModel');
const { predictValueOfImage } = require('./predictValueOfImage');
const { saveModel } = require('./saveModel');
const { updateModel } = require('./updateModel');

const evaluate = curry((testData, model) => {
  console.log('Evaluating test data.');

  const predictValueOfImageWithModel = predictValueOfImage(model);

  const accuratePredictions = compose(
    filter((prediction) => prediction),
    reduce((accumulator, image) => {
      const { accurate } = predictValueOfImageWithModel(image);
      return accumulator.concat(accurate);
    }, []),
  )(testData);

  console.log(
    `Prediction Accuracy: ${accuratePredictions.length} / ${testData.length}`,
  );
});

const train = curry((trainingData, callback = noop) => {
  console.log('Begining training sequence.');
  console.log('');

  const initialModel = getModel();

  const { epochs } = initialModel;
  const cycles = range(0, epochs);

  let model = initialModel;

  forEach((cycle) => {
    console.log(`Training cycle ${cycle} starting.`);
    model = updateModel(trainingData, model);

    callback(model);
    console.log(`Training cycle ${cycle} complete.`);
    console.log('');
  }, cycles);

  return saveModel(model);
});

/**
 * Initializes the neural network. HyperParams are adjusted via the constants file. This
 * is done through a factory function from the outset to allow for initializing a model
 * in the future.
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
