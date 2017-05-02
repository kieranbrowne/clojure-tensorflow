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
;; => [0 1 2 3 4]
```

## Usage

Add the following to your dependencies in `project.clj`.

[![Clojars Project](https://img.shields.io/clojars/v/org.clojars.kieran/clojure-tensorflow.svg)](https://clojars.org/org.clojars.kieran/clojure-tensorflow)

TensorFlow requires at least Java 8 to run. This will already be the default on most machines, but if it isn't for you, it's possible to force lein to use it by adding the :java-cmd "/path/to/java" key to your `project.clj`.

### Running on the GPU
If you have a [CUDA GPU](https://developer.nvidia.com/cuda-gpus) on your machine it's definitely worth using it with TensorFlow.

Just run the following commands in the shell to download the appropriate Java Native Interface files for your OS.

```bash
 TF_TYPE="gpu" 
 OS=$(uname -s | tr '[:upper:]' '[:lower:]')
 mkdir -p ./native
 curl -L \
   "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-${TF_TYPE}-${OS}-x86_64-1.1.0.tar.gz" |
   tar -xz -C ./native
```


## License

Copyright © 2017 Kieran Browne

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
