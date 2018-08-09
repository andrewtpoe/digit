const { addIndex, curry, forEach, map, reduce } = require('ramda');

const forEachWithIndex = addIndex(forEach);

const mapWithIndex = addIndex(map);

const noop = () => {};

const reduceWithIndex = addIndex(reduce);

const shuffler = curry((random, list) => {
  let idx = -1;
  let len = list.length;
  let position;
  let result = [];
  while (++idx < len) {
    position = Math.floor((idx + 1) * random());
    result[idx] = result[position];
    result[position] = list[idx];
  }
  return result;
});

const shuffle = shuffler(Math.random);

module.exports = { forEachWithIndex, mapWithIndex, reduceWithIndex, shuffle };
