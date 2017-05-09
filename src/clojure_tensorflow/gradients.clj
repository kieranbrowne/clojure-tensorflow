(ns clojure-tensorflow.gradients
  (:require
   [clojure-tensorflow.build :as build]
   [clojure-tensorflow.utils :as utils]
   [clojure-tensorflow.ops :as ops]
   [clojure-tensorflow.ops :as tf]))

;; Gradients
;; (defn numerical-gradient
;;   "Calculate the approximate gradient of y with respect to x
;;   This is best used for testing differentiated gradients."
;;   ([y x accuracy]
;;    (tf/div
;;     (tf/sub
;;      (func (tf/add x accuracy))
;;      (func x))
;;     accuracy)
;;     )
;;   ([func x] (numerical-gradient func x (tf/constant 0.000001))))

(defn get-op-by-name [n]
  (first (filter #(= (:name %) n) @build/shadow-graph)))

(def get-inputs (comp :inputs get-op-by-name #(.name (.op %))))

(defn depends-on?
  [independent dependent]
  (or (some #(= independent %) (get-inputs dependent))
      (some (partial depends-on? independent) (get-inputs dependent))
      (= dependent independent)))

(defn get-registered-gradient [output]
  (let [inputs (get-inputs output)]
    (case (.type (.op output))
      "Const" (ops/constant 1.)
      "Variable" (ops/constant 1.)
      "Sub" (ops/constant 1.)
      "Pow" (ops/pow (ops/mult (second inputs) (first inputs))
                     (tf/sub (second inputs) (tf/constant 1.)))
      "MatMul" (first inputs) ;; need to work out which is important
      "Mult" (first inputs) ;; need to work out which is important
      "Div" (first inputs) ;; need to work out which is important
      "Sigmoid" (ops/mult (first inputs)
                          (ops/sub (ops/constant 1.) (first inputs)))
      "Mean" (ops/size (first inputs)) ;; need to work out which is important
      )))


;; (defn paths
;;   "Return a list of all paths from one op to another"
;;   [from to])

(defn op-to-op [from to]
  (reverse
   (loop [inputs [from] path []]
     (let [dependents (filter (partial depends-on? to) inputs)]
       (if (empty? dependents)
         path
         (recur (get-inputs (first dependents)) (conj path (first dependents))))))))


(defn gradients
  "The symbolic gradient of y with respect to x.
  For example, if we wanted to calculate the gradient of our
  cost function with respect to our weight, we could use
  `(gradients cost weights)`."
  [y & xs]
  (map
   #(tf/sum (reduce ops/mult
            (map get-registered-gradient
                 (op-to-op y %))) (tf/constant [0]))
   xs))

(defn apply-gradients
  [xs gradients]
  (map #(ops/assign %1 (ops/sub %1 (ops/transpose %2)))
       xs gradients))


;; Optimizers

(defn gradient-descent
  "The very simplest optimizer."
  [cost-fn & weights]
  (apply-gradients weights (apply gradients (cons cost-fn weights))))
