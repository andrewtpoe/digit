const { forEach } = require("ramda");

const { buildBiases, buildWeights } = require("../getModel");

const layers = [3, 2, 1];

describe("buildBiases", () => {
  describe("with the invalid biase matrix", () => {
    const invalidBiaseMatrixes = [undefined, [[[0, 0], [0]], [[0, 0]]]];
    const defaultBiases = [[[0], [0]], [[0]]];

    forEach(
      testBiases =>
        it("builds the expected default matrix of biases", () => {
          const biases = buildBiases(layers, testBiases);
          expect(biases).toEqual(defaultBiases);
        }),
      invalidBiaseMatrixes
    );
  });

  describe("with a valid biase matrix", () => {
    it("returns the valid biases", () => {
      const testBiases = [[[1.01], [0.09]], [[1.0]]];
      const biases = buildBiases(layers, testBiases);
      expect(biases).toBe(testBiases);
    });
  });
});

describe("buildWeights", () => {
  describe("with an invalid weight matrix", () => {
    const invalidWeightMatrixes = [undefined, [[[0, 0, 0], [0, 0]], [[0, 0]]]];
    const defaultWeights = [[[0, 0, 0], [0, 0, 0]], [[0, 0]]];

    forEach(
      testWeights =>
        it("builds the expected default matrix of weights", () => {
          const weights = buildWeights(layers, testWeights);
          expect(weights).toEqual(defaultWeights);
        }),
      invalidWeightMatrixes
    );
  });

  describe("with a valid weight matrix", () => {
    it("returns the valid weights", () => {
      const testWeights = [
        [[1.01, 0.09, 1.0], [1.02, 1.01, 0.7]],
        [[1.0, 1.19]]
      ];
      const weights = buildWeights(layers, testWeights);
      expect(weights).toBe(testWeights);
    });
  });
});
