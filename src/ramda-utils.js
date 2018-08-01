const { addIndex, map } = require('ramda');

const mapWithIndex = addIndex(map);

module.exports = { mapWithIndex };
