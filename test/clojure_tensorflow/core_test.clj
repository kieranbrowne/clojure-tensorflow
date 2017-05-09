(ns clojure-tensorflow.core-test
  (:require [clojure.test :refer :all]
            [clojure-tensorflow.core :refer :all]
            [clojure-tensorflow.ops :as tf]
            [clojure-tensorflow.build :as build]
            [clojure-tensorflow.gradients :as tf.optimizers]
            ))

(deftest test-session-running
  (testing "Run simple graph"
    (is (= (session-run [(tf/constant [1])])
           [1]))))

(deftest test-basic-feed-forward-neural-network
  (let [input (tf/constant [[1. 0. 1.]])
        output (tf/constant [[0.5]])
        weights (tf/variable [[0.08] [-0.65] [0.44]])
        model (tf/sigmoid (tf/matmul input weights))
        cost (tf/sub output model)]

    (testing "Constant input"
      (is (= (session-run [input]) [[1. 0. 1.]])))

    (testing "Global variables initializer"
      (is (= (session-run
              [(tf/global-variables-initializer)
               weights
               ]) (map (partial map float) [[0.08] [-0.65] [0.44]]))))

    (testing "Compute gradients"
      (is (= (session-run
              [(tf/global-variables-initializer)
               (tf.optimizers/gradients cost weights)
               ]) (map (partial map float) [[0.24960001 0.0 0.24960001]]))))

    (testing "Gradient decent"
      (is (= (session-run
              [(tf/global-variables-initializer)
               (tf.optimizers/gradient-descent cost weights)
               cost
               ]) [[(float -0.00519979)]])))
    ))
