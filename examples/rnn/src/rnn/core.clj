(ns rnn.core
  (:require [clojure-tensorflow.ops :as tf]
            [clojure-tensorflow.layers :as layer]
            [clojure-tensorflow.optimizers :as optimize]
            [clojure-tensorflow.core :refer [run]]))

;; Training data
(def data
  (clojure.string/join (repeat 5 "meow ")))

(defn encode-one-hot
  "Encode string into one hot vectors"
  [string]
  (let [chars (set (map char string))]
    (map (fn [c] (map #(if (= c %) 1. 0.) chars))
         string)))

(defn decode-one-hot
  "Decode one hot vectors into string"
  [model-output string]
  (let [chars (set (map char string))]
    (apply str
           (map #(key (apply max-key val (zipmap chars %)))
                model-output))))


(def input (tf/constant (encode-one-hot data)))
(def target (tf/constant
             (partition 4 (rest (map conversion-map data)))))


(defn random-normal
  "Generate a tensor of random values with a normal distribution"
  ([shape stddev]
   (let [source (java.util.Random. (rand))]
     ((reduce #(partial repeatedly %2 %1)
              #(.nextGaussian source)
              (reverse shape)))))
  ([shape] (random-normal shape 1)))

(defn zeros
  "Generate a tensor of random values with a normal distribution"
  [shape]
  (let [source (java.util.Random. (rand))]
    (reduce #(repeat %2 %1)
             0.
             (reverse shape))))


;; Network weights
(def input->hidden
  (tf/variable (random-normal [5 18])))

(def hidden->hidden
  (tf/variable (random-normal [18 18])))

(def hidden->output
  (tf/variable (random-normal [18 1])))

;; In an rnn the output of a layer is added to the next input
;; and fed back into the same layer which is typically the only
;; layer in a network.


;; Define network / model
(def hidden-state (tf/variable (zeros [5])))
(def hidden-layer
  (tf/sigmoid
   (tf/add
    (tf/matmul hidden-state hidden->hidden)
    (tf/matmul input input->hidden)
    )))
(def output (tf/matmul hidden-state hidden->output))




;; Run network

(run (tf/global-variables-initializer))


(defn constant [val]
  (let [tensor (utils/clj->tensor val)]
    (op-builder
     {:operation "Const"
      :attrs {:dtype (.dataType tensor)
              :value tensor
              }})))

(run
  (clojure-tensorflow.build/op-builder
   {:operation "Gather"
    :inputs [(tf/constant [1. 2.]) (tf/constant [0 1])]
    }))

(run (tf/assign hidden-state hidden-layer))

(decode-one-hot
 (run output) data)


(def input-dim 2)
(def hidden-dim 6)
