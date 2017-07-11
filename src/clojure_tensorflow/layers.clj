(ns clojure-tensorflow.layers
  (:require
   [clojure-tensorflow.ops :as ops]
   [clojure-tensorflow.build :as build]
   ))

(def layer-defaults
  {:activation ops/sigmoid})

(def rand-synapse (fn [] (dec (* 2 (rand)))))

(defn gen-weights [in out]
  (ops/variable
   (repeatedly
    (.size (.shape in) 1)
    #(repeatedly out rand-synapse))))


(defn linear [previous-layer size & args]
  (let [{activation :activation} (merge layer-defaults (apply hash-map args))]
    (activation
     (ops/matmul previous-layer
                (gen-weights previous-layer size)))))


(defn conv2d
  ([input filter padding strides]
   (build/op-builder
    {:operation "Conv2D"
     :attrs {:padding padding
             :strides strides
             }
     :inputs [input filter]}))
  ([input filter] (conv2d input filter "SAME" (long-array [1 1 1 1]))))

(defn max-pool
  ([input ksize strides padding]
   (build/op-builder
    {:operation "MaxPool"
     :attrs {:strides (long-array [1 3 3 1])
             :ksize (long-array [1 3 3 1])
             :padding padding
             }
     :inputs [input]}))
  ([input ksize] (max-pool input ksize ksize "SAME")))
