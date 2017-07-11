(ns clojure-tensorflow.session
  (:require [clojure-tensorflow.graph :refer [graph]]))

(def ^:dynamic session (org.tensorflow.Session. graph))
