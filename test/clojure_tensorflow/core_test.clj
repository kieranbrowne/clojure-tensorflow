(ns clojure-tensorflow.core-test
  (:require [clojure.test :refer :all]
            [clojure-tensorflow.core
             :refer [run with-graph with-this-graph with-session]]
            [clojure-tensorflow.ops :as tf]
            [clojure-tensorflow.build :as build]
            [clojure-tensorflow.gradients :as tf.gradients]
            [clojure-tensorflow.optimizers :as tf.optimizers]
            [clojure-tensorflow.layers :as layer]
            [clojure-tensorflow.utils :as utils]
            [autodiff.protocols :as ad]
            ))

(defn approx= [a b] (< (Math/abs (- a b)) 1e-3))

(deftest test-session-running
  (with-session
    (testing "Run simple operation in default graph"
      (is (= (run (tf/constant [1.0]))
             [1.0])))))

(deftest test-traversal
  (with-this-graph
    {:a {}
     :b {}
     :c {:inputs [:a :b]}
     :d {:inputs [:c :b]}
     :e {:inputs [:a :b]}
     :f {:inputs [:d :e]}}

    (testing "Parents and Ancestors"
      (is (= #{} (tf.gradients/parents :a)))
      (is (= #{} (tf.gradients/parents :b)))
      (is (= #{:a :b} (tf.gradients/parents :c)))
      (is (= #{:c :b} (tf.gradients/parents :d)))
      (is (= #{:a :b} (tf.gradients/parents :e)))
      (is (= #{:d :e} (tf.gradients/parents :f)))
      (is (= #{:a :b :c :d :e} (tf.gradients/ancestors :f)))
      (is (= #{:a :b} (tf.gradients/ancestors :e)))
      (is (= #{:a :b :c} (tf.gradients/ancestors :d)))
      (is (= #{:a :b} (tf.gradients/ancestors :c)))
      (is (= #{} (tf.gradients/ancestors :b)))
      (is (= #{} (tf.gradients/ancestors :a)))
      )

    (testing "Children and Decendents"
      (is (= #{:c :e} (tf.gradients/children :a)))
      (is (= #{:c :d :e} (tf.gradients/children :b)))
      )

    ))

(let [a (tf/constant 3.)
      b (tf/constant 5.)
      c (tf/pow a b)]
  (run c))


  (deftest test-gradients
    (with-graph
      (with-session
        (let [a (tf/constant 3.)
              b (tf/constant 5.)
              c (tf/add a b)
              d (tf/sub b a)
              e (tf/mul a b)
              f (tf/pow a b)
              g (tf/sigmoid a)
              h (tf/div a b)]

          (testing "Gradients"
            (is (= (float 1.)
                   (run (:f' (tf.gradients/gradients a a)))
                   )))

          (testing "Gradients"
            (is (= (float 1.)
                   (:f' (run (tf.gradients/gradients b b)))
                   )))

          (testing "Gradients sub"
            (is (= (float -1.)
                   (run (:f' (tf.gradients/gradients d a)))
                   )))

          (testing "Gradients add"
            (is (= (ad/->Dual 8. 1.)
                   (run (tf.gradients/gradients c a))
                   )))

          (testing "Gradients mul"
            (is (= (ad/->Dual 15. 5.)
                   (run (tf.gradients/gradients e a))
                   )))

          (testing "Gradients mul"
            (is (= (ad/->Dual 15. 3.)
                   (run (tf.gradients/gradients e b))
                   )))

          (testing "Gradients div"
            (is (= (float 0.2)
                   (run (:f' (tf.gradients/gradients h a)))
                   ))
            (is (= (float -0.12)
                   (run (:f' (tf.gradients/gradients h b)))
                   )))

          (testing "Gradients pow"
            (is (= (float 405.)
                   (run (:f' (tf.gradients/gradients f a)))
                   )))

          (testing "Gradients pow"
            (is (approx= 266.962
                   (run (:f' (tf.gradients/gradients f b)))
                   ))
            (is (= (float 0.045176655)
                   (run (:f' (tf.gradients/gradients g a)))
                   )))))))


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

    ;; (testing "Compute gradients"
    ;;   (is (= (run
    ;;            [(tf/global-variables-initializer)
    ;;             (tf.gradients/gradients cost weights)])
    ;;          (map (partial map float) [[-0.23383345] [0.0] [-0.23383345]]))))

    ;; (testing "Compute gradients without supplying the relevant variables"
    ;;   (is (= (run
    ;;            [(tf/global-variables-initializer)
    ;;             (tf.gradients/gradients cost)])
    ;;          (map (partial map float) [[-0.23383345] [0.0] [-0.23383345]]))))

    ;; (testing "Gradient decent"
    ;;   (is (= (run
    ;;            [(tf/global-variables-initializer)
    ;;             (tf.optimizers/gradient-descent cost :weights [weights] :learning-rate 1.)
    ;;             cost])
    ;;          [[(float -0.22862685)]])))


    ;; (testing "Training"
    ;;   (is (< (run
    ;;            [(tf/global-variables-initializer)
    ;;             (repeat 100 (tf.optimizers/gradient-descent cost :weights [weights] :learning-rate 1.))
    ;;             (tf/mean (tf/mean cost))])
    ;;          0.0001)))

    ;; (let [a (tf/constant 2.)
    ;;       b (tf/constant 1.)
    ;;       c (tf/add a b)
    ;;       d (tf/add b (tf/constant 1.))
    ;;       e (tf/mul c d)]

    ;;   (testing "Constant deriv"
    ;;     (is (= (run
    ;;             (tf.gradients/gradients c a))
    ;;            (float 1.0))))

    ;;   (testing "Add deriv"
    ;;     (is (= (run
    ;;             (tf.gradients/gradients e c))
    ;;            (float 2.0))))

    ;;   (testing "Add deriv via second input"
    ;;     (is (= (run
    ;;             (tf.gradients/gradients e d))
    ;;            (float 3.0))))

    ;;   (testing "Mul deriv via complex path"
    ;;     (is (= (run
    ;;             (tf.gradients/gradients e b))
    ;;            (float 5.0))))

    ;;   (testing "Mul deriv via complex path"
    ;;     (is (= (run
    ;;             (tf.gradients/gradients e e))
    ;;            (float 1.0))))
    ))


;; (deftest test-training
;;   (let [rand-seed (java.util.Random. 1)
;;         rand-synapse #(dec (* 2 (.nextDouble rand-seed)))]


;;     (testing "Single layer neural net"
;;       (is (<
;;            (let [input (tf/constant [[1. 0. 1.]
;;                                      [1. 1. 0.]
;;                                      [0. 1. 1.]
;;                                      [0. 0. 1.]])
;;                  target (tf/constant [[0. 0.]
;;                                       [1. 0.]
;;                                       [1. 0.]
;;                                       [0. 0.]])
;;                  syn-0 (tf/variable (repeatedly 3 #(repeatedly 2 rand-synapse)))
;;                  network (tf/sigmoid (tf/matmul input syn-0))
;;                  error (tf/pow (tf/sub target network) (tf/constant 2.))]
;;              (run
;;               [(tf/global-variables-initializer)
;;                (repeat 400 (tf.optimizers/gradient-descent error :weights [syn-0] :learning-rate 1.))
;;                (tf/mean (tf/mean error))]))
;;            0.001)))


;;     (testing "Neural net with hidden layer"
;;       (is (<
;;            (let [input (tf/constant [[1. 1. 1.]
;;                                      [1. 0. 1.]
;;                                      [0. 1. 1.]
;;                                      [0. 0. 1.]])
;;                  target (tf/constant [[0.]
;;                                       [1.]
;;                                       [1.]
;;                                       [0.]])
;;                  syn-0 (tf/variable (repeatedly 3 #(repeatedly 5 rand-synapse)))
;;                  syn-1 (tf/variable (repeatedly 5 #(repeatedly 1 rand-synapse)))
;;                  hidden-layer (tf/sigmoid (tf/matmul input syn-0))
;;                  network (tf/sigmoid (tf/matmul hidden-layer syn-1))
;;                  error (tf/pow (tf/sub target network) (tf/constant 2.))]
;;              (run
;;               [(tf/global-variables-initializer)
;;                (repeat 1000 (tf.optimizers/gradient-descent
;;                              error :weights [syn-0 syn-1] :learning-rate 1.))
;;                (tf/mean (tf/mean error))]))
;;            0.001)))

;;     (testing "Neural net with infered variables"
;;       (is (<
;;            (let [input (tf/constant [[1. 1. 1.]
;;                                      [1. 0. 1.]
;;                                      [0. 1. 1.]
;;                                      [0. 0. 1.]])
;;                  target (tf/constant [[0.]
;;                                       [1.]
;;                                       [1.]
;;                                       [0.]])
;;                  syn-0 (tf/variable (repeatedly 3 #(repeatedly 5 rand-synapse)))
;;                  syn-1 (tf/variable (repeatedly 5 #(repeatedly 1 rand-synapse)))
;;                  hidden-layer (tf/sigmoid (tf/matmul input syn-0))
;;                  network (tf/sigmoid (tf/matmul hidden-layer syn-1))
;;                  error (tf/pow (tf/sub target network) (tf/constant 2.))]
;;              (run
;;               [(tf/global-variables-initializer)
;;                (repeat 1000 (tf.optimizers/gradient-descent error :learning-rate 1.))
;;                (tf/mean (tf/mean error))]))

;;            0.001)))


;;     (testing "Autoencoder"
;;       (let [rand-seed (java.util.Random. 1)
;;             rand-synapse #(dec (* 2 (.nextDouble rand-seed)))]
;;       (is (<
;;            (let [inputs (tf/constant [[0.0 0.0 1.0] [0.0 1.0 1.0] [1.0 1.0 1.0] [1.0 0.0 1.0]])
;;                  outputs inputs
;;                  weights (tf/variable (repeatedly 3 (fn [] (repeatedly 2 rand-synapse))))
;;                  bias (tf/variable (repeatedly 4 (fn [] (repeatedly 2 rand-synapse))))
;;                  weights2 (tf/variable (repeatedly 2 (fn [] (repeatedly 3 rand-synapse))))
;;                  network (tf/sigmoid
;;                           (tf/matmul (tf/sigmoid (tf/add (tf/matmul inputs weights) bias))
;;                                      weights2))
;;                  error (tf/sum (tf/pow (tf/sub outputs network) (tf/constant 2.))
;;                                (tf/constant 1))]
;;               (run
;;                 [(tf/global-variables-initializer)
;;                  (repeat 1000 (tf.optimizers/gradient-descent error :learning-rate 20. :weights [weights bias weights2]))
;;                  (tf/mean error)])) 0.3))))

;;     (deftest test-layer-fns
;;       (let [x (tf/constant [[1. 0. 1.]])
;;             y (tf/constant [[1.0]])
;;             network (-> x
;;                         (layer/linear 6)
;;                         (layer/linear 8)
;;                         (layer/linear 1))
;;             error (tf/pow (tf/sub y network) (tf/constant 2.))]

;;         (testing "Layer building"
;;           (is
;;            (< (run
;;                [(tf/global-variables-initializer)
;;                 (repeat 100 (tf.optimizers/gradient-descent error :learning-rate 1.))
;;                 (tf/mean (tf/mean error))])
;;               0.01)))))))


(deftest test-feed
  (with-graph
    (with-session

      (let [x (tf/placeholder tf/float32)
            y (tf/placeholder tf/float32)]

        (testing "Basic feed"
          (is (= (float 2.)
                 (run (tf/identity x) {x 2.}))))

        (testing "Basic feed into ops"
          (is (= (float 8.)
                 (run (tf/mul x y)
                   {x 2. y 4.}))))

        (testing "Basic feed into ops"
          (is (= (map float [8. 2. 4.])
                 (run (tf/mul x y)
                   {x [2. 1. 2.] y [4. 2. 2.]}))))

        ))))

;; (deftest test-feed-neural-network
;;   (let [rand-seed (java.util.Random. 1)
;;         rand-synapse #(dec (* 2 (.nextDouble rand-seed)))]

;;     (with-graph
;;       (with-session

;;         (testing "Neural net with infered variables feed"
;;           (is (<
;;                (let [input (tf/placeholder tf/float32)
;;                      target (tf/placeholder tf/float32)
;;                      syn-0 (tf/variable (repeatedly 3 #(repeatedly 5 rand-synapse)))
;;                      syn-1 (tf/variable (repeatedly 5 #(repeatedly 1 rand-synapse)))
;;                      hidden-layer (tf/sigmoid (tf/matmul input syn-0))
;;                      network (tf/sigmoid (tf/matmul hidden-layer syn-1))
;;                      error (tf/pow (tf/sub target network) (tf/constant 2.))]
;;                  (run
;;                    [(tf/global-variables-initializer)
;;                     (repeat 1000 (tf.optimizers/gradient-descent error :learning-rate 1.))
;;                     (tf/mean (tf/mean error))]
;;                    {:input [[1. 1. 1.] [1. 0. 1.] [0. 1. 1.] [0. 0. 1.]]
;;                     :target [[0.]      [1.]       [1.]       [0.]]}))

;;                0.001)))
;;         ))))

;; (deftest test-linear-regression
;;   (with-graph
;;     (with-session

;;       (let [x (tf/placeholder tf/float32)
;;             y (tf/placeholder tf/float32)
;;             m (tf/variable 1.)
;;             b (tf/variable 0.)
;;             line-of-best-fit (tf/add (tf/mul m x) b)
;;             cost (tf/square (tf/sub y line-of-best-fit))
;;             ]

;;         (run (tf/global-variables-initializer))

;;         (testing "Predict Single"
;;           (is (= (float 2.)
;;                  (run line-of-best-fit {:x 2.}))))

;;         (testing "Predict Vector"
;;           (is (= (map float [0. 1. 2.])
;;                  (run line-of-best-fit
;;                    {:x [0. 1. 2.]}))))

;;         (testing "Cost Single"
;;           (is (= (float 4.)
;;                  (run cost {:x 6. :y 8.}))))

;;         (testing "Cost Vector"
;;           (is (= (map float [1. 0. 1.])
;;                  (run cost
;;                    {:x [0. 1. 2.] :y [1. 1. 1.]}))))

;;         (testing "Stochastic gradient descent"
;;           (is (> (float 0.01)
;;                  ;; y = 5x+2.5
;;                  (let [xs [-1. 0. 1. 2.]
;;                        ys (map #(+ 2.5 (* 5 %)) xs)]
;;                    (doseq [datum (take 10 (cycle
;;                                            (map #(hash-map :x %1 :y %2) xs ys)))]
;;                        (run
;;                          (tf.optimizers/gradient-descent cost :learning-rate 0.4)
;;                          datum))
;;                    (run (tf/mean cost) {:x xs :y ys}))
;;                  )))

;;         ))))

(deftest test-tensor-transformations
  (with-graph
    (with-session

      (let [tensor
            (tf/constant [[0 0 0 0 0]
                          [0 0 0 1 0]
                          [0 0 1 1 1]
                          [0 0 0 1 0]
                          [0 0 0 0 0]])]

        (testing "Slice Operation"
          (is (= [[0 1 0] [1 1 1] [0 1 0]]
                 (run
                   (tf/slice
                    tensor
                    (tf/constant [1 2])
                    (tf/constant [3 3])
                    )))))

        (testing "Pad Operation"
          (is (= [[0 0 1 0 0] [0 0 0 0 0]]
                 (run
                   (tf/pad
                    (tf/constant [[1]])
                    (tf/constant [[0 1] [2 2]])
                    )))))

        ))))


(deftest convolutions
  "Convolutional noughts and crosses"
  (with-graph
    (with-session
      (let [cross (tf/constant
                   [[1. 0. 1.]
                    [0. 1. 0.]
                    [1. 0. 1.]])
            nought (tf/constant
                    [[1. 1. 1.]
                     [1. 0. 1.]
                     [1. 1. 1.]])
            input (tf/reshape
                   (tf/constant
                    (vector
                     (run (tf/pad cross  (tf/constant [[3 1] [2 2]])))
                     (run (tf/pad cross  (tf/constant [[0 4] [1 3]])))
                     (run (tf/pad cross  (tf/constant [[2 2] [4 0]])))
                     (run (tf/pad cross  (tf/constant [[4 0] [0 4]])))
                     (run (tf/pad nought (tf/constant [[3 1] [2 2]])))
                     (run (tf/pad nought (tf/constant [[0 4] [1 3]])))
                     (run (tf/pad nought (tf/constant [[2 2] [4 0]])))
                     (run (tf/pad nought (tf/constant [[4 0] [0 4]])))))
                   (tf/constant [8 7 7 1])
                   )
            output (tf/one-hot (tf/constant [0 0 0 0 1 1 1 1]))
            ]


        ;; (testing "Conv2D op"
        ;;   (is (= (utils/to-floats [[[[19.0] [25.0]] [[37.0] [43.0]]]])
        ;;          (run (layer/conv2d
        ;;                (tf/reshape
        ;;                 (tf/constant (map float (range 9)))
        ;;                 (tf/constant [1 3 3 1]))
        ;;                (tf/reshape
        ;;                 (tf/constant (map float (range 4)))
        ;;                 (tf/constant [2 2 1 1]))
        ;;                "VALID"
        ;;                (long-array [1 1 1 1])
        ;;                )))))

        ))))


(deftest auto-type-conversion
  (with-graph
    (with-session

      (testing "Scalar Int"
        (is (= 2 (run (tf/add 1 1)))))

      (testing "Scalar Float"
        (is (= (float 2.0) (run (tf/add 1. 1.)))))

      (testing "PersistentVector"
        (is (= [[(float 2.0)]] (run (tf/add [[1.]] [[1.]])))))

      (testing "PersistentList"
        (is (= [[(float 2.0)]] (run (tf/add '((1.)) '((1.)))))))

      (testing "Collection and Scalar"
        (is (= [[(float 3.0)] [(float 6.)]]
               (run (tf/mul '((1.) [2.]) 3.)))))

      (testing "Reshape"
        (is (= [[0 1 2] [3 4 5] [6 7 8]]
               (run (tf/reshape
                     (range 9)
                     '(3 3)
                     )))))
      )))

;; (run
;;   (tf/reshape
;;    (range 9)
;;    '(3 3)
;;    )
;;   )

;; (run (tf/add (tf/constant 2.)
;;              (ad/coerce (tf/constant 4.))))

(deftest autodiff
  (with-graph
    (with-session

      (let [x (tf/constant 9.0)
            y (tf/constant -2.0)]

        (testing "Running a dual type"
          (is (= (ad/->Dual 6. 0.)
                 (run (ad/add (tf/constant 2.)
                              (ad/coerce (tf/constant 4.))))
                 (run (ad/add (ad/coerce (tf/constant 2.))
                              (tf/constant 4.)))
                 )))

        (testing "Derivative of a constant"
          (is (= 0. (run (:f' (ad/coerce (tf/constant 20.4)))))))

        (testing "Derivative of addition"
          (is (= 1. (run (ad/d ad/add
                               (ad/wrt (tf/constant 1.))
                               (tf/constant 2.))))))
        ))))


;; (deftest parsed-gradients
;;   (with-graph
;;     (with-session

;;       (let [x (tf/constant 9.0)
;;             y (tf/constant -2.0)]

;;         (testing "NegGrad"
;;           (is (= -1.0
;;                  (run
;;                    (tf.gradients/parsed-grad
;;                     (build/op-builder {:operation "Neg" :inputs [x]})
;;                     x)))))

;;         (testing "SqrtGrad"
;;           (is (= 0.1666666716337204
;;                  (run
;;                    (tf.gradients/parsed-grad
;;                     (build/op-builder {:operation "Sqrt" :inputs [x]})
;;                     x)))))

;;         (testing "SinGrad"
;;           (is (= -0.416146844625473
;;                  (run
;;                    (tf.gradients/parsed-grad
;;                     (build/op-builder {:operation "Sin" :inputs [y]})
;;                     y)))))

;;         (testing "DivGrad w.r.t x"
;;           (is (= -0.5
;;                  (run
;;                    (tf.gradients/parsed-grad
;;                     (build/op-builder {:operation "Div" :inputs [x y]})
;;                     x)))))

;;         (testing "DivGrad w.r.t y"
;;           (is (= -2.25
;;                  (run
;;                    (tf.gradients/parsed-grad
;;                     (build/op-builder {:operation "Div" :inputs [x y]})
;;                     y)))))

;;         (testing "SigmoidGrad"
;;           (is (= 1.2336639338172972E-4
;;                  (run
;;                    (tf.gradients/parsed-grad
;;                     (build/op-builder {:operation "Sigmoid" :inputs [x]})
;;                     x)))))

;;         (testing "TanhGrad"
;;           (is (= 0.07065081596374512
;;                  (run
;;                    (tf.gradients/parsed-grad
;;                     (build/op-builder {:operation "Tanh" :inputs [y]})
;;                     y)))))

;;         ))))




;; (with-graph
;;   (with-session
;;     (let [x (tf/constant 9.0)
;;           y (tf/constant -2.0)
;;           z (build/op-builder {:operation "Maximum" :inputs [y x]})]

;;       (run
;;         (tf.gradients/parsed-grad z y)
;;         )
;;       )))


;; (with-graph
;;   (with-session
;;     (let [cross (tf/constant
;;                  [[1. 0. 1.]
;;                   [0. 1. 0.]
;;                   [1. 0. 1.]])
;;           nought (tf/constant
;;                   [[1. 1. 1.]
;;                    [1. 0. 1.]
;;                    [1. 1. 1.]])
;;           input (tf/constant
;;                   (vector
;;                    (run (tf/pad cross  (tf/constant [[3 1] [2 2]])))
;;                    (run (tf/pad cross  (tf/constant [[0 4] [1 3]])))
;;                    (run (tf/pad cross  (tf/constant [[2 2] [4 0]])))
;;                    (run (tf/pad cross  (tf/constant [[4 0] [0 4]])))
;;                    (run (tf/pad nought (tf/constant [[3 1] [2 2]])))
;;                    (run (tf/pad nought (tf/constant [[0 4] [1 3]])))
;;                    (run (tf/pad nought (tf/constant [[2 2] [4 0]])))
;;                    (run (tf/pad nought (tf/constant [[4 0] [0 4]])))))
;;           target (tf/to-float (tf/one-hot (tf/constant [0 0 0 0 1 1 1 1])))

;;           filter (tf/variable (tf/random-normal [3 3 1 1]))

;;           input (tf/reshape input '(-1 7 7 1))
;;           conv1 (layer/conv2d input filter)
;;           mpool (layer/max-pool conv1 (long-array [1 2 2 1]))
;;           rshap (tf/reshape mpool [-1 9])
;;           output (layer/linear rshap 2)

;;           error (tf/square (tf/sub target output))
;;           ]

;;       (run (tf/global-variables-initializer))


;;       ;; (run (tf.gradients/gradient (tf/reshape )))

;;       (run (tf.gradients/gradient error rshap))
;;       (run
;;         (tf.gradients/gradient error filter))

;;       )))


;; (org.tensorflow.Tensor/create
;;  (tf/constant 1))

;; (let [x (tf/constant 1)
;;       y (tf/reshape x (tf/constant [1]))]
;;   (apply (first (get @tf.gradients/registered-gradients
;;         (->
;;          (tf.gradients/paths y x)
;;          first first
;;          :output
;;          .op
;;          .type
;;          )))
;;    (->
;;     (tf.gradients/paths y x)
;;     first first
;;     :output
;;     tf.gradients/get-inputs )
;;    )
;;   )


;; (utils/->clj (.shape (tf/constant [1. 4.])))


;; (with-graph
;;   (with-session
;;     (let [cross (tf/constant
;;                  [[1. 0. 1.]
;;                   [0. 1. 0.]
;;                   [1. 0. 1.]])
;;           nought (tf/constant
;;                   [[1. 1. 1.]
;;                    [1. 0. 1.]
;;                    [1. 1. 1.]])
;;           input (tf/constant
;;                  (clojure.walk/postwalk
;;                   #(if (number? %) (vector %) %)
;;                   (vector
;;                    (run (tf/pad cross  (tf/constant [[3 1] [2 2]])))
;;                    (run (tf/pad cross  (tf/constant [[0 4] [1 3]])))
;;                    (run (tf/pad cross  (tf/constant [[2 2] [4 0]])))
;;                    (run (tf/pad cross  (tf/constant [[4 0] [0 4]])))
;;                    (run (tf/pad nought (tf/constant [[3 1] [2 2]])))
;;                    (run (tf/pad nought (tf/constant [[0 4] [1 3]])))
;;                    (run (tf/pad nought (tf/constant [[2 2] [4 0]])))
;;                    (run (tf/pad nought (tf/constant [[4 0] [0 4]]))))))
;;           ]

;;       (run (tf/conv2d
;;             input
;;             (tf/constant (tf/random-normal [5 5 1 2]))
;;             ))


;;       )))
