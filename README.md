# Clojure and TensorFlow

A very light layer over Java interop for working with TensorFlow.

```clojure
(ns example.core
  (:require [clojure-tensorflow.ops :as tf]
            [clojure-tensorflow.core :as run]))

;; Define some operations
(def input (tf/constant [[0. 1.] [0. 0.] [1. 1.] [1. 0.]]))
(def weights (tf/variable (repeatedly 2 #(vector (dec (* 2 (rand)))))))

(def shallow-neural-net (tf/sigmoid (tf/matmul input weights)))

;; Run operations
(run/session-run
 [(tf/global-variables-initializer)
  shallow-neural-net])
```

## Usage

### 1. Add this library your dependencies in project.clj

[![Clojars Project](https://img.shields.io/clojars/v/clojure-tensorflow.svg)](https://clojars.org/clojure-tensorflow)

### 2. Add a version of TensorFlow to your dependencies

The easiest option is to add `[org.tensorflow/tensorflow  "1.1.0-rc1"]` to your dependencies. However, TensorFlow is fastest when you use a version compiled for your hardware and OS, and this is especially true if you have a GPU.

Read the official [Installing TensorFlow for Java](https://www.tensorflow.org/install/install_java) guide for more information.

> NOTE: TensorFlow requires at least Java 8 to run. This will already be the default on most machines, but if it isn't for you, it's possible to force lein to use it by adding the :java-cmd "/path/to/java" key to your `project.clj`.

## License

Copyright © 2017 Kieran Browne

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
