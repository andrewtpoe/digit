const { loadData } = require('./data-loader');
const { initializeNetwork } = require('./network');

(async () => {
  const { trainingData, validationData, testData } = await loadData;

  const network = initializeNetwork();

  const trainingOptions = { epochs: 30, miniBatchSize: 10, stepSize: 3.0 };

  network.train(trainingData, trainingOptions, network.evaluate(testData));
})();
