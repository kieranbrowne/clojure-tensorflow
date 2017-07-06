(ns clojure-tensorflow.optimizers
  (:require [clojure-tensorflow.gradients
             :refer [gradients apply-gradients relevant-variables]]
            [clojure-tensorflow.ops :as tf]))

(defn gradient-descent
  "The very simplest optimizer."
  [cost-fn & arguments]
  (let [{l-rate :learning-rate weights :weights}
        (merge {:learning-rate 0.1 :weights (relevant-variables cost-fn)} (apply hash-map arguments))]
    (apply-gradients weights (map #(tf/mult (tf/constant l-rate) %) (apply gradients (cons cost-fn weights)))))
  )
