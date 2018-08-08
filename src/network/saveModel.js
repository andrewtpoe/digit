const fs = require('fs');

/**
 * Saves the model
 *
 * @param {object} model a model generated by the program
 */
function saveModel(model) {
  fs.writeFileSync('./src/model.json', JSON.stringify(model), function(err) {
    if (err) {
      return console.log(err);
    }

    console.log('The model was saved!');
  });
}

module.exports = { saveModel };