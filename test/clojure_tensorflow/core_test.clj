(ns clojure-tensorflow.core-test
  (:require [clojure.test :refer :all]
            [clojure-tensorflow.core :refer :all]
            [clojure-tensorflow.ops :as tf]
            [clojure-tensorflow.build :as build]
            ))

(deftest test-session-running
  (testing "Run simple graph"
    (is (= (session-run [(tf/constant [1])])
           [1])))
  )

(deftest test-gradients
  (let [var-two (tf/variable 2.)
        three-pow-two (tf/pow (tf/constant 3.) var-two)]
    (testing "Get gradient for Pow"
      (is (= (session-run [(tf/global-variables-initializer)
                           (tf/gradients three-pow-two var-two)])
             6.)))
    ))

(def input (tf/constant [[1. 0. 1.]]))
(def output (tf/constant [[0.5]]))

(def weights (tf/variable (repeatedly 3 #(vector (dec (* 2 (rand)))))))

(def model (tf/sigmoid (tf/matmul input weights)))

(def cost (tf/sub output model))

(session-run
 [(tf/global-variables-initializer)
  cost])

(session-run
 [(tf/global-variables-initializer)
  (tf/gradients cost weights)])

(session-run
 [(tf/global-variables-initializer)
  ;; (tf/assign weights (tf/sub weights (tf/transpose (tf/gradients cost weights))))
  cost
  ])

(session-run
 [(tf/global-variables-initializer)
  (tf/numerical-gradients cost weights)
  ])

(session-run
 [(tf/get-registered-gradient (tf/sigmoid (tf/constant 9.)))])

()
