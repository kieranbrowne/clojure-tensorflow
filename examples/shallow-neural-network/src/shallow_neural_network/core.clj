(ns shallow-neural-network.core
  (:require [clojure-tensorflow.ops :as tf]
            [clojure-tensorflow.core :as run]))

;;;;;;;;;;;; TRAINING DATA ;;;;;;;;;;;;;;;

(def training-data
  ;; input => output
  [[0. 0. 1.]   [0.]
   [0. 1. 1.]   [1.]
   [1. 1. 1.]   [1.]
   [1. 0. 1.]   [0.]])

(def inputs (tf/constant (take-nth 2 training-data)))
(def outputs (tf/constant (take-nth 2 (rest training-data))))


;;;;;;;;;;; VARIABLE WEIGHTS ;;;;;;;;;;;;;

(def weights
  "Initialise weights as variable tensor of values between -1 and 1"
  (tf/variable
   (repeatedly 3 (fn [] (repeatedly 1 #(dec (rand 2)))))))

;;;;;;;;;; DEFINE OUR NETWORK ;;;;;;;;;;;;

(defn network [weights]
  (tf/sigmoid (tf/matmul inputs weights)))

(defn error [weights]
  (tf/div
   (tf/pow
    (tf/sub outputs (network weights))
    (tf/constant 2.))
   (tf/constant 2.)))



;;;;;;;;; TESTING OUR NETWORK ;;;;;;;;;;;;
;; Errors without training

(run/session-run
 [(tf/global-variables-initializer)
  (tf/mean (error weights))])
;; => [0.14975819]

;; The mean error is too damn high


;;;;;;;;;; GRADIENT DESCENT ;;;;;;;;;;;;;;

(run/session-run
 [(tf/global-variables-initializer)
  (repeat 1000 (tf/gradient-descent weights inputs error))
  (tf/mean (error weights))])
;; => [2.1348597E-4]

;; The mean error is now sufficiently small.
;; That's it.
