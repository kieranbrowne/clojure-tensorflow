(ns clojure-tensorflow.optimizers
  (:require [clojure-tensorflow.gradients
             :refer [gradients apply-gradients relevant-variables]]))

(defn gradient-descent
  "The very simplest optimizer."
  ([cost-fn & weights]
   (apply-gradients weights (apply gradients (cons cost-fn weights))))
  ([cost-fn] (apply (partial gradient-descent cost-fn)
                    (relevant-variables cost-fn))))
