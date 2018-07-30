const PythonShell = require('python-shell');

const loadData = new Promise((resolve, reject) => {
  console.log(
    'Loading MNIST image data with the help of Python. This may take a while.',
  );

  // The raw data was originally created by pickling and gzipping the MNIST data set.
  // Since there are no readily available tools for "unpickling" something in JavaScript,
  // we need to get the data converted to a JavaScript readable format.
  PythonShell.run('./src/data-loader/unpickle.py', (error, results) => {
    if (error) {
      reject(error);
    }

    // The data must be serialized to be passed from the Python shell to JavaScript.
    // Thus, these values are strings that must be parsed into an object containing arrays.
    const [trainingData, validationData, testData] = results;

    resolve({
      trainingData: JSON.parse(trainingData),
      validationData: JSON.parse(validationData),
      testData: JSON.parse(testData),
    });
  });
});

module.exports = { loadData };
