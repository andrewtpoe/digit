const { isEmpty, reduce } = require('ramda');

const { DEFAULT_LAYERS } = require('../constants');

const { mapWithIndex } = require('../ramda-utils');

function loadDefaultModel() {
  try {
    return require('../model.json');
  } catch (error) {
    return {};
  }
}

function buildDefaultMatrix(layers) {
  return reduce(
    (accumulator, layer) => accumulator.concat([new Array(layer).fill(0)]),
    [],
    layers,
  );
}

function matrixValid(layers, matrix) {
  if (layers.length !== matrix.length) {
    return false;
  }

  return isEmpty(
    mapWithIndex(
      (matrixLayer, layerNum) => matrixLayer.length === layers[layerNum],
      matrix,
    ),
  );
}

function buildActiveMatrix(layers, matrix) {
  return matrixValid(layers, matrix) ? matrix : buildDefaultMatrix(layers);
}

function buildActiveModel() {
  const {
    layers = DEFAULT_LAYERS,
    biases = [],
    weights = [],
  } = loadDefaultModel();

  return {
    layers,
    biases: buildActiveMatrix(layers, biases),
    weights: buildActiveMatrix(layers, weights),
  };
}

module.exports = { buildActiveModel };
