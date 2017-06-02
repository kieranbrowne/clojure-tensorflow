(ns clojure-tensorflow.core-test
  (:require [clojure.test :refer :all]
            [clojure-tensorflow.core :refer :all]
            [clojure-tensorflow.ops :as tf]
            [clojure-tensorflow.gradients :as tf.gradients]
            [clojure-tensorflow.optimizers :as tf.optimizers]
            [clojure-tensorflow.layers :as layer]))
            

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
      (is (= (session-run [(tf.gradients/gradients a a)])
             (float 1.))))

    (testing "Gradients"
      (is (= (session-run [(tf.gradients/gradients a a)])
             (float 1.))))

    (testing "Gradients sub"
      (is (= (session-run [(tf.gradients/gradients d a)])
             (float -1.))))

    (testing "Gradients add"
      (is (= (session-run [(tf.gradients/gradients c a)])
             (float 1.))))

    (testing "Gradients mult"
      (is (= (session-run [(tf.gradients/gradients e a)])
             (float 5.))))

    (testing "Gradients mult"
      (is (= (session-run [(tf.gradients/gradients e b)])
             (float 3.))))

    (testing "Gradients pow"
      (is (= (session-run [(tf.gradients/gradients f a)])
             (float 405.))))

    (testing "Gradients pow"
      (is (= (session-run [(tf.gradients/gradients f b)])
             (float 266.9628))))

    (testing "Gradients sigmoid"
      (is (= (session-run [(tf.gradients/gradients g a)])
             (float 0.045176655))))))

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
               weights])
             (map (partial map float) [[0.08] [-0.65] [0.44]]))))

    (testing "Compute gradients"
      (is (= (session-run
              [(tf/global-variables-initializer)
               (tf.gradients/gradients cost weights)])
             (map (partial map float) [[-0.23383345] [0.0] [-0.23383345]]))))

    (testing "Compute gradients without supplying the relevant variables"
      (is (= (session-run
              [(tf/global-variables-initializer)
               (tf.gradients/gradients cost)])
             (map (partial map float) [[-0.23383345] [0.0] [-0.23383345]]))))

    (testing "Gradient decent"
      (is (= (session-run
              [(tf/global-variables-initializer)
               (tf.optimizers/gradient-descent cost weights)
               cost])
             [[(float -0.22862685)]])))


    (testing "Training"
      (is (< (session-run
              [(tf/global-variables-initializer)
               (repeat 100 (tf.optimizers/gradient-descent cost weights))
               (tf/mean (tf/mean cost))])
             0.0001)))

    (let [a (tf/constant 2.)
          b (tf/constant 1.)
          c (tf/add a b)
          d (tf/add b (tf/constant 1.))
          e (tf/mult c d)]

      (testing "Constant deriv"
        (is (= (session-run
                [(tf.gradients/gradients c a)])
               (float 1.0))))

      (testing "Add deriv"
        (is (= (session-run
                [(tf.gradients/gradients e c)])
               (float 2.0))))

      (testing "Add deriv via second input"
        (is (= (session-run
                [(tf.gradients/gradients e d)])
               (float 3.0))))

      (testing "Mult deriv via complex path"
        (is (= (session-run
                [(tf.gradients/gradients e b)])
               (float 5.0))))

      (testing "Mult deriv via complex path"
        (is (= (session-run
                [(tf.gradients/gradients e e)])
               (float 1.0)))))))


(deftest test-training
  (let [rand-seed (java.util.Random. 1)
        rand-synapse #(dec (* 2 (.nextDouble rand-seed)))]


    (testing "Single layer neural net"
      (is (<
           (let [input (tf/constant [[1. 0. 1.]
                                     [1. 1. 0.]
                                     [0. 1. 1.]
                                     [0. 0. 1.]])
                 target (tf/constant [[0. 0.]
                                      [1. 0.]
                                      [1. 0.]
                                      [0. 0.]])
                 syn-0 (tf/variable (repeatedly 3 #(repeatedly 2 rand-synapse)))
                 network (tf/sigmoid (tf/matmul input syn-0))
                 error (tf/pow (tf/sub target network) (tf/constant 2.))]
             (session-run
              [(tf/global-variables-initializer)
               (repeat 400 (tf.optimizers/gradient-descent error syn-0))
               (tf/mean (tf/mean error))]))
           0.001)))


    (testing "Neural net with hidden layer"
      (is (<
           (let [input (tf/constant [[1. 1. 1.]
                                     [1. 0. 1.]
                                     [0. 1. 1.]
                                     [0. 0. 1.]])
                 target (tf/constant [[0.]
                                      [1.]
                                      [1.]
                                      [0.]])
                 syn-0 (tf/variable (repeatedly 3 #(repeatedly 5 rand-synapse)))
                 syn-1 (tf/variable (repeatedly 5 #(repeatedly 1 rand-synapse)))
                 hidden-layer (tf/sigmoid (tf/matmul input syn-0))
                 network (tf/sigmoid (tf/matmul hidden-layer syn-1))
                 error (tf/pow (tf/sub target network) (tf/constant 2.))]
             (session-run
              [(tf/global-variables-initializer)
               (repeat 1000 (tf.optimizers/gradient-descent
                             error syn-0 syn-1))
               (tf/mean (tf/mean error))]))
           0.001)))

    (testing "Neural net with infered variables"
      (is (<
           (let [input (tf/constant [[1. 1. 1.]
                                     [1. 0. 1.]
                                     [0. 1. 1.]
                                     [0. 0. 1.]])
                 target (tf/constant [[0.]
                                      [1.]
                                      [1.]
                                      [0.]])
                 syn-0 (tf/variable (repeatedly 3 #(repeatedly 5 rand-synapse)))
                 syn-1 (tf/variable (repeatedly 5 #(repeatedly 1 rand-synapse)))
                 hidden-layer (tf/sigmoid (tf/matmul input syn-0))
                 network (tf/sigmoid (tf/matmul hidden-layer syn-1))
                 error (tf/pow (tf/sub target network) (tf/constant 2.))]
             (session-run
              [(tf/global-variables-initializer)
               (repeat 1000 (tf.optimizers/gradient-descent error))
               (tf/mean (tf/mean error))]))

           0.001)))))

(deftest test-layer-fns
  (let [x (tf/constant [[1. 0. 1.]])
        y (tf/constant [[1.0]])
        network (-> x
                    (layer/linear 6)
                    (layer/linear 8)
                    (layer/linear 1))
        error (tf/pow (tf/sub y network) (tf/constant 2.))]

    (testing "Layer building"
      (is
       (< (session-run
           [(tf/global-variables-initializer)
            (repeat 100 (tf.optimizers/gradient-descent error))
            (tf/mean (tf/mean error))])
          0.01)))))
