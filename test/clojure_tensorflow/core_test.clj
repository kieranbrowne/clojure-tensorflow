(ns clojure-tensorflow.core-test
  (:require [clojure.test :refer :all]
            [clojure-tensorflow.core :refer :all]
            [clojure-tensorflow.ops :as tf]
            ))

(deftest test-session-running
  (testing "Run simple graph"
    (is (= (session-run [(tf/constant [1])])
           [1])))
  )
