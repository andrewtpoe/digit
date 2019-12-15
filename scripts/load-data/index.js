const PythonShell = require("python-shell");
const { zip } = require("ramda");

/**
 * Connects an image (an array of pixel activations) to a value (the number value of the image)
 * in the same data object.
 *
 * @param {object} data an object containing all of the data for this group
 * @property {array.<array>} data.images an array of pixel activation arrays
 * @property {array.<number>} data.values an array of number values matching the index in images.
 * @returns An array in this format [[pixel activations], value]
 */
function formatData({ images, values }) {
  return zip(images, values);
}

const loadData = new Promise((resolve, reject) => {
  console.log(
    "Loading MNIST image data with the help of Python. This may take a while."
  );

  // The raw data was originally created by pickling and gzipping the MNIST data set.
  // Since there are no readily available tools for "unpickling" something in JavaScript,
  // we need to get the data converted to a JavaScript readable format.
  PythonShell.run("./src/data-loader/unpickle.py", (error, results) => {
    if (error) {
      reject(error);
    }

    // The data must be serialized to be passed from the Python shell to JavaScript.
    // Thus, these values are strings that must be parsed into an object containing arrays.
    const [trainingData, validationData, testData] = results;

    resolve({
      trainingData: formatData(JSON.parse(trainingData)),
      validationData: formatData(JSON.parse(validationData)),
      testData: formatData(JSON.parse(testData))
    });
  });
});

module.exports = { loadData };
