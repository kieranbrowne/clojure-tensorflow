# Clojure and TensorFlow

A very light layer over Java interop for working with TensorFlow.

```clojure
(ns example.core
  (:require [clojure-tensorflow.ops :as tf]
            [clojure-tensorflow.core :as run]))

;; Define some operations
(def blah (tf/constant (range 5)))

;; Run operations
(run/session-run [blah])
```

## Usage

[![Clojars Project](https://img.shields.io/clojars/v/org.clojars.kieran/clojure-tensorflow.svg)](https://clojars.org/org.clojars.kieran/clojure-tensorflow)

## License

Copyright © 2017 Kieran Browne

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
