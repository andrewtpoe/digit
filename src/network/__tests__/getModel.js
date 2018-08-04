const { forEachWithIndex } = require('../../ramda-utils');

const { buildBiases, buildWeights } = require('../getModel');

describe('buildBiases', () => {
  describe('with the default value (an empty array)', () => {
    it('builds the expected array of biases', () => {
      const layers = [3, 2, 1];
      const testBiases = [];
      const biases = buildBiases(layers, testBiases);
      const expectedBiases = [[[0], [0]], [[0]]];
      expect(biases).toEqual(expectedBiases);
    });
  });

  describe('with a valid value', () => {
    it('returns the valid biases', () => {
      const layers = [3, 2, 1];
      const testBiases = [[1.01, 0.09], [1.0]];
      const biases = buildBiases(layers, testBiases);
      expect(biases).toBe(testBiases);
    });
  });
});

describe('buildWeights', () => {
  describe('with the default value (an empty array)', () => {
    it('builds the expected array of biases', () => {
      const layers = [3, 2, 1];
      const defaultWeights = [];
      const weights = buildWeights(layers, defaultWeights);
      const expectedWeights = [[[0, 0, 0], [0, 0, 0]], [[0, 0]]];
      expect(weights).toEqual(expectedWeights);
    });
  });

  describe('with a valid value', () => {
    it('returns the valid weights', () => {
      const layers = [3, 2, 1];
      const testWeights = [
        [[1.01, 0.09, 1.0], [1.02, 1.01, 0.7]],
        [[1.0, 1.19]],
      ];
      const weights = buildWeights(layers, testWeights);
      expect(weights).toBe(testWeights);
    });
  });
});
