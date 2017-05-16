(ns shallow-neural-network.core
  (:require [clojure-tensorflow.ops :as tf]
            [clojure-tensorflow.optimizers :as optimizers]
            [clojure-tensorflow.core :as run]))

;;;;;;;;;;;; Training Data ;;;;;;;;;;;;;;;

(def training-data
  ;; input => output
  [[0. 0. 1.]   [0.]
   [0. 1. 1.]   [1.]
   [1. 1. 1.]   [1.]
   [1. 0. 1.]   [0.]])

(def inputs (tf/constant (take-nth 2 training-data)))
(def outputs (tf/constant (take-nth 2 (rest training-data))))


;;;;;;;;;;; Variable Weights ;;;;;;;;;;;;;

(def weights
  "Initialise weights as variable tensor of values between -1 and 1"
  (tf/variable
   (repeatedly 3 (fn [] (repeatedly 1 #(dec (rand 2)))))))

;;;;;;;;;; Define Our Network ;;;;;;;;;;;;

(def network
  (tf/sigmoid (tf/matmul inputs weights)))

(def error
  (tf/div
   (tf/pow
    (tf/sub outputs network)
    (tf/constant 2.))
   (tf/constant 2.)))


;;;;;;;;; Testing Our Network ;;;;;;;;;;;;

;; Errors without training
(run/session-run
 [(tf/global-variables-initializer)
  (tf/mean error)])
;; => [0.14975819]
;; The mean error is too damn high


;;;;;;;;;; Gradient Descent ;;;;;;;;;;;;;;

(run/session-run
 [(tf/global-variables-initializer)
  (repeat 1000 (optimizers/gradient-descent error weights))
  (tf/mean error)])
;; => [2.1348597E-4]

;; The mean error is now sufficiently small.
;; That's it.
