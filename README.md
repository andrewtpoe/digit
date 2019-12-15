# Digit

_An exploration in artificial intelligence._

## Overview

This project contains a basic, but functional, neural network written in
JavaScript. It is largely based on the code from the book, "Neural Networks and
Deep Learning". The original code is written in Python 2.x, and can be found in
the `/neural-networks-and-deep-learning` directory.

This is _not_ a production ready neural network (by a long shot). This is more
like a homework/ research project. I took on this project to help myself more
fully grasp the concepts of neural networks because I learn best by building.

As a part of translating the original source to JavaScript, I've attempted to
break apart the terse Python code. It most likely runs slower, however my hope
is that is may be a bit simpler for less familiar developers to digest.

## Developing the Neural Network

### Prerequisites

This program runs in Node JS, but requires Python 2 as well. Python is used to
initialize and unpickle the image data.

## Training the Neural Network

- `npm run train`: Runs a train/ evaluate loop based on the HyperParams in
  `/config.json`. This script will generate a new model, train it, log results,
  and ask for confirmation before saving it as the default.

### Available HyperParams

- `BATCH_SIZE`: [Default: 10] The number of images to load into each training
  cycle.
- `EPOCHS`: [Default: 30] The number of training cycles to run.
- `LAYERS`: [Default: [784, 30, 10]] Describes the number of neurons in each
  layer of the network. For this network to run correctly:
  - the first (input) layer MUST have 784 neurons
  - the last (output) layer MUST have 10 neurons
  - What happens in between is up to you.
- `LEARNING_RATE`: [Default: 3.0] A factor in how far the network will adjust on
  each cycle.

## Evaluating the Neural Network

- `npm run evaluate`: Evaluates the currently saved model against the test data.
  If a developer uses the current version of this package, they will be using
  this saved model.

## Using the Neural Network

```js
import network from "digit";

network.evaluate(pixelActivationsArr); // <== returns a number value between 0 and 9
```
