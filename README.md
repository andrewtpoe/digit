# Neural Networks and Deep Learning (in JavaScript)

This repository contains the code from the book, "Neural Networks and Deep
Learning", rewritten in JavaScript. The original code is written in Python 2.x,
and can be found in the `/original` directory. I took on this project to help
myself more fully grasp the concepts of the Neural Network. I learn best by
building.

As a part of translating the original source to JavaScript, I've attempted to
break apart the terse code. It most likely runs slower, however my hope is that
is may be a bit simpler for less familiar developers to digest.

This is _not_ a production ready neural network (by a long shot). This is more
like a homework/ research project.

## Using the Neural Network

### Prerequisites

This program runs in Node JS, but requires Python 2 as well. Python is used to
initialize and unpickle the data.

### Running the program

- `npm run start`: Runs a train/ evaluate loop based on the HyperParams in
  `src/contstants.js`.
