/**
 * This file contains the HyperParameters for the Neural Network.
 * Network tuning should be done here.
 */

/**
 * This section defines network architecture parameters
 */

// For this network to run correctly:
// - the first (input) layer MUST have 784 neurons
// - the last (output) layer MUST have 10 neurons
// What happens in between is up to you.
const LAYERS = [784, 30, 10];

/**
 * This section defines network training parameters
 */

// The number of training cycles to run.
const EPOCHS = 3;

// The number of images to load into each training cycle.
const BATCH_SIZE = 10;

// A factor in how far the network will adjust on each cycle.
const LEARNING_RATE = 3.0;

module.exports = {
  BATCH_SIZE,
  EPOCHS,
  LAYERS,
  LEARNING_RATE
};
