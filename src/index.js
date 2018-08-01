const { loadData } = require('./data-loader');
const { initializeNetwork } = require('./network');

(async () => {
  const { trainingData, testData } = await loadData;

  const network = await initializeNetwork();

  const trainingOptions = {
    // The number of training cycles to run.
    epochs: 30,
    // The number of images to load into each training cycle.
    miniBatchSize: 10,
    // A factor in how far the network will adjust on each cycle.
    stepSize: 3.0,
  };

  network.train(trainingData, trainingOptions, network.evaluate(testData));
})();
