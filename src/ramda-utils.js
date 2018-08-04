const { addIndex, forEach, map, reduce } = require('ramda');

const forEachWithIndex = addIndex(forEach);

const mapWithIndex = addIndex(map);

const reduceWithIndex = addIndex(reduce);

module.exports = { forEachWithIndex, mapWithIndex, reduceWithIndex };
