(ns clojure-tensorflow.layers
  (:require
   [clojure-tensorflow.ops :as ops]
   [clojure-tensorflow.ops :as tf]))

(def layer-defaults
  {:activation tf/sigmoid})

(def rand-synapse (fn [] (dec (* 2 (rand)))))

(defn gen-weights [in out]
  (tf/variable
   (repeatedly
    (.size (.shape in) 1)
    #(repeatedly out rand-synapse))))


(defn linear [previous-layer size & args]
  (let [{activation :activation} (merge layer-defaults (apply hash-map args))]
    (activation
     (tf/matmul previous-layer
                (gen-weights previous-layer size)))))
