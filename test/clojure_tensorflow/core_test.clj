(ns clojure-tensorflow.core-test
  (:require [clojure.test :refer :all]
            [clojure-tensorflow.core
             :refer [run with-graph with-session]]
            [clojure-tensorflow.ops :as tf]
            [clojure-tensorflow.build :as build]
            [clojure-tensorflow.gradients :as tf.gradients]
            [clojure-tensorflow.optimizers :as tf.optimizers]
            [clojure-tensorflow.layers :as layer]
            [clojure-tensorflow.utils :as utils]))




(deftest test-session-running
  (with-session
    (testing "Run simple operation in default graph"
      (is (= (run (tf/constant [1]))
             [1])))))

(deftest test-gradients
  (with-graph
    (with-session
      (let [a (tf/constant 3.)
            b (tf/constant 5.)
            c (tf/add a b)
            d (tf/sub b a)
            e (tf/mult a b)
            f (tf/pow a b)
            g (tf/sigmoid a)
            h (tf/div a b)]

        (testing "Gradients"
          (is (= (run (tf.gradients/gradients a a))
                 (float 1.))))

        (testing "Gradients"
          (is (= (run (tf.gradients/gradients a a))
                 (float 1.))))

        (testing "Gradients sub"
          (is (= (run (tf.gradients/gradients d a))
                 (float -1.))))

        (testing "Gradients add"
          (is (= (run (tf.gradients/gradients c a))
                 (float 1.))))

        (testing "Gradients mult"
          (is (= (run (tf.gradients/gradients e a))
                 (float 5.))))

        (testing "Gradients mult"
          (is (= (run (tf.gradients/gradients e b))
                 (float 3.))))

        (testing "Gradients div"
          (is (= (run (tf.gradients/gradients h a))
                 (float 0.2)))
          (is (= (run (tf.gradients/gradients h b))
                 (float -0.12))))

        (testing "Gradients pow"
          (is (= (run (tf.gradients/gradients f a))
                 (float 405.))))

        (testing "Gradients pow"
          (is (= (run (tf.gradients/gradients f b))
                 (float 266.9628))))
          (is (= (run (tf.gradients/gradients g a))
                 (float 0.045176655)))))))


(deftest test-basic-feed-forward-neural-network
  (let [input (tf/constant [[1. 0. 1.]])
        output (tf/constant [[0.5]])
        weights (tf/variable [[0.08] [-0.65] [0.44]])
        model (tf/sigmoid (tf/matmul input weights))
        cost (tf/sub output model)]

    (testing "Constant input"
      (is (= (run input) [[1. 0. 1.]])))

    (testing "Global variables initializer"
      (is (= (run
               [(tf/global-variables-initializer)
                weights])
             (map (partial map float) [[0.08] [-0.65] [0.44]]))))

    (testing "Compute gradients"
      (is (= (run
               [(tf/global-variables-initializer)
                (tf.gradients/gradients cost weights)])
             (map (partial map float) [[-0.23383345] [0.0] [-0.23383345]]))))

    (testing "Compute gradients without supplying the relevant variables"
      (is (= (run
               [(tf/global-variables-initializer)
                (tf.gradients/gradients cost)])
             (map (partial map float) [[-0.23383345] [0.0] [-0.23383345]]))))

    (testing "Gradient decent"
      (is (= (run
               [(tf/global-variables-initializer)
                (tf.optimizers/gradient-descent cost :weights [weights] :learning-rate 1.)
                cost])
             [[(float -0.22862685)]])))


    (testing "Training"
      (is (< (run
               [(tf/global-variables-initializer)
                (repeat 100 (tf.optimizers/gradient-descent cost :weights [weights] :learning-rate 1.))
                (tf/mean (tf/mean cost))])
             0.0001)))

    (let [a (tf/constant 2.)
          b (tf/constant 1.)
          c (tf/add a b)
          d (tf/add b (tf/constant 1.))
          e (tf/mult c d)]

      (testing "Constant deriv"
        (is (= (run
                (tf.gradients/gradients c a))
               (float 1.0))))

      (testing "Add deriv"
        (is (= (run
                (tf.gradients/gradients e c))
               (float 2.0))))

      (testing "Add deriv via second input"
        (is (= (run
                (tf.gradients/gradients e d))
               (float 3.0))))

      (testing "Mult deriv via complex path"
        (is (= (run
                (tf.gradients/gradients e b))
               (float 5.0))))

      (testing "Mult deriv via complex path"
        (is (= (run
                (tf.gradients/gradients e e))
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
             (run
              [(tf/global-variables-initializer)
               (repeat 400 (tf.optimizers/gradient-descent error :weights [syn-0] :learning-rate 1.))
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
             (run
              [(tf/global-variables-initializer)
               (repeat 1000 (tf.optimizers/gradient-descent
                             error :weights [syn-0 syn-1] :learning-rate 1.))
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
             (run
              [(tf/global-variables-initializer)
               (repeat 1000 (tf.optimizers/gradient-descent error :learning-rate 1.))
               (tf/mean (tf/mean error))]))

           0.001)))


    (testing "Autoencoder"
      (let [rand-seed (java.util.Random. 1)
            rand-synapse #(dec (* 2 (.nextDouble rand-seed)))]
      (is (<
           (let [inputs (tf/constant [[0.0 0.0 1.0] [0.0 1.0 1.0] [1.0 1.0 1.0] [1.0 0.0 1.0]])
                 outputs inputs
                 weights (tf/variable (repeatedly 3 (fn [] (repeatedly 2 rand-synapse))))
                 bias (tf/variable (repeatedly 4 (fn [] (repeatedly 2 rand-synapse))))
                 weights2 (tf/variable (repeatedly 2 (fn [] (repeatedly 3 rand-synapse))))
                 network (tf/sigmoid
                          (tf/matmul (tf/sigmoid (tf/add (tf/matmul inputs weights) bias))
                                     weights2))
                 error (tf/sum (tf/pow (tf/sub outputs network) (tf/constant 2.))
                               (tf/constant 1))]
              (run
                [(tf/global-variables-initializer)
                 (repeat 500 (tf.optimizers/gradient-descent error :learning-rate 20. :weights [weights bias weights2]))
                 (tf/mean error)])) 0.2))))

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
           (< (run
               [(tf/global-variables-initializer)
                (repeat 100 (tf.optimizers/gradient-descent error :learning-rate 1.))
                (tf/mean (tf/mean error))])
              0.01)))))))


(deftest test-feed
  (with-graph
    (with-session

      (let [x (tf/placeholder :x tf/float32)
            y (tf/placeholder :y tf/float32)]

        (testing "Basic feed"
          (is (= (float 2.)
                 (run (tf/identity x) {:x 2.}))))

        (testing "Basic feed into ops"
          (is (= (float 8.)
                 (run (tf/mult x y)
                   {:x 2. :y 4.}))))

        (testing "Basic feed into ops"
          (is (= (map float [8. 2. 4.])
                 (run (tf/mult x y)
                   {:x [2. 1. 2.] :y [4. 2. 2.]}))))

        ))))

(deftest test-feed-neural-network
  (let [rand-seed (java.util.Random. 1)
        rand-synapse #(dec (* 2 (.nextDouble rand-seed)))]

    (with-graph
      (with-session

        (testing "Neural net with infered variables feed"
          (is (<
               (let [input (tf/placeholder :input tf/float32)
                     target (tf/placeholder :target tf/float32)
                     syn-0 (tf/variable (repeatedly 3 #(repeatedly 5 rand-synapse)))
                     syn-1 (tf/variable (repeatedly 5 #(repeatedly 1 rand-synapse)))
                     hidden-layer (tf/sigmoid (tf/matmul input syn-0))
                     network (tf/sigmoid (tf/matmul hidden-layer syn-1))
                     error (tf/pow (tf/sub target network) (tf/constant 2.))]
                 (run
                   [(tf/global-variables-initializer)
                    (repeat 1000 (tf.optimizers/gradient-descent error :learning-rate 1.))
                    (tf/mean (tf/mean error))]
                   {:input [[1. 1. 1.] [1. 0. 1.] [0. 1. 1.] [0. 0. 1.]]
                    :target [[0.]      [1.]       [1.]       [0.]]}))

               0.001)))
        ))))

(deftest test-linear-regression
  (with-graph
    (with-session

      (let [x (tf/placeholder :x tf/float32)
            y (tf/placeholder :y tf/float32)
            m (tf/variable 1.)
            b (tf/variable 0.)
            line-of-best-fit (tf/add (tf/mult m x) b)
            cost (tf/square (tf/sub y line-of-best-fit))
            ]

        (run (tf/global-variables-initializer))

        (testing "Predict Single"
          (is (= (float 2.)
                 (run line-of-best-fit {:x 2.}))))

        (testing "Predict Vector"
          (is (= (map float [0. 1. 2.])
                 (run line-of-best-fit
                   {:x [0. 1. 2.]}))))

        (testing "Cost Single"
          (is (= (float 4.)
                 (run cost {:x 6. :y 8.}))))

        (testing "Cost Vector"
          (is (= (map float [1. 0. 1.])
                 (run cost
                   {:x [0. 1. 2.] :y [1. 1. 1.]}))))

        (testing "Stochastic gradient descent"
          (is (> (float 0.01)
                 ;; y = 5x+2.5
                 (let [xs [-1. 0. 1. 2.]
                       ys (map #(+ 2.5 (* 5 %)) xs)]
                   (doseq [datum (take 10 (cycle
                                           (map #(hash-map :x %1 :y %2) xs ys)))]
                       (run
                         (tf.optimizers/gradient-descent cost :learning-rate 0.4)
                         datum))
                   (run (tf/mean cost) {:x xs :y ys}))
                 )))

        ))))
