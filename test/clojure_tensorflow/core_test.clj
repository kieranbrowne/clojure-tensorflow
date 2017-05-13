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

(deftest test-gradients
  (let [a (tf/constant 3.)
        b (tf/constant 5.)
        c (tf/add a b)
        d (tf/sub b a)
        e (tf/mult a b)
        f (tf/pow a b)
        g (tf/sigmoid a)]

    (testing "Gradients"
      (is (= (session-run [(tf.optimizers/gradients a a)])
             (float 1.))))

    (testing "Gradients"
      (is (= (session-run [(tf.optimizers/gradients a a)])
             (float 1.))))

    (testing "Gradients sub"
      (is (= (session-run [(tf.optimizers/gradients d a)])
             (float -1.))))

    (testing "Gradients add"
      (is (= (session-run [(tf.optimizers/gradients c a)])
             (float 1.))))

    (testing "Gradients mult"
      (is (= (session-run [(tf.optimizers/gradients e a)])
             (float 5.))))

    (testing "Gradients mult"
      (is (= (session-run [(tf.optimizers/gradients e b)])
             (float 3.))))

    (testing "Gradients pow"
      (is (= (session-run [(tf.optimizers/gradients f a)])
             (float 405.))))

    (testing "Gradients pow"
      (is (= (session-run [(tf.optimizers/gradients f b)])
             (float 266.9628))))

    (testing "Gradients sigmoid"
      (is (= (session-run [(tf.optimizers/gradients g a)])
             (float 266.9628))))
    ))

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
               ]) (map (partial map float) [[-0.24960001 0.0 -0.24960001]]))))

    (testing "Gradient decent"
      (is (= (session-run
              [(tf/global-variables-initializer)
               (tf.optimizers/gradient-descent cost weights)
               cost
               ]) [[(float -0.00519979)]])))

    (let [a (tf/constant 2.)
          b (tf/constant 1.)
          c (tf/add a b)
          d (tf/add b (tf/constant 1.))
          e (tf/mult c d)]

      (testing "Constant deriv"
        (is (= (session-run
                [(tf.optimizers/gradients c a)
                 ]) (float 1.0))))

      (testing "Add deriv"
        (is (= (session-run
                [(tf.optimizers/gradients e c)
                 ]) (float 2.0))))

      (testing "Add deriv via second input"
        (is (= (session-run
                [(tf.optimizers/gradients e d)
                 ]) (float 3.0))))

      (testing "Mult deriv via complex path"
        (is (= (session-run
                [(tf.optimizers/gradients e b)
                 ]) (float 5.0))))

      (testing "Mult deriv via complex path"
        (is (= (session-run
                [(tf.optimizers/gradients e e)
                 ]) (float 1.0))))
      )
    ))



(let [input (tf/constant [[1. 0. 1.]
                          [1. 1. 0.]
                          [0. 1. 1.]
                          [0. 0. 1.]])
      target (tf/constant [[0.]
                           [1.]
                           [1.]
                           [0.]])
      syn-0 (tf/variable (repeatedly 3 #(vector (dec (* 2 (rand))))))
      network (tf/sigmoid (tf/matmul input syn-0))
      error (tf/pow (tf/sub target network) (tf/constant 2.))]
  (session-run
   [(tf/global-variables-initializer)
    ;; (tf.optimizers/gradient-descent error syn-0)
    (repeat 900 (tf.optimizers/gradient-descent error syn-0))
    (tf/mean error)
    network
    ])
  )


(let [input (tf/constant [[1. 0. 1.]
                          [1. 1. 0.]
                          [0. 1. 1.]
                          [0. 0. 1.]])
      target (tf/constant [[0.]
                           [1.]
                           [1.]
                           [0.]])
      rand-synapse (fn [] (dec (* 2 (rand))))
      syn-0 (tf/variable (repeatedly 3 #(repeatedly 4 rand-synapse)))
      syn-1 (tf/variable (repeatedly 4 #(repeatedly 1 rand-synapse)))
      hidden-layer (tf/sigmoid (tf/matmul input syn-0))
      output-layer (tf/sigmoid (tf/matmul hidden-layer syn-1))
      error (tf/pow (tf/sub target output-layer) (tf/constant 2.))]
  (session-run
   [(tf/global-variables-initializer)
    ;; (tf.optimizers/gradient-descent error syn-0)
    (repeat 90 (tf.optimizers/gradient-descent error syn-1))
    (repeat 90 (tf.optimizers/gradient-descent error syn-0))
    ;; output-layer
    (tf/mean error)
    ])
  )
