(ns clojure-tensorflow.gradients
  (:refer-clojure :exclude [parents ancestors])
  (:require
   [clojure-tensorflow.build :as build]
   [clojure-tensorflow.graph :as graph]
   [clojure-tensorflow.utils :as utils]
   [clojure-tensorflow.ops :as ops]
   [clojure.string :as string]
   [autodiff.protocols :as ad]
   ))


(defn parents
  [op-name]
  (set (:inputs (op-name @graph/shadow-graph'))))

(defn ancestors
  [op-name]
  (loop [antecedents (parents op-name)
         next-generation (reduce clojure.set/union (map parents op-name))]
    (when (empty? next-generation)
      antecedents
      (recur
       (clojure.set/union antecedents next-generation)
       (reduce clojure.set/union (map parents op-name))
       ))))


(if (empty? #{}) true false)
(clojure.set/union #{1 2} #{2 3})

(defn children
  [op-name]
  (set (filter #(contains? (parents %) op-name) (keys @graph/shadow-graph'))))


;; (defn relevant-variables [op]
;;   (filter #(depends-on? % op)
;;           (map :tf-op
;;                (filter
;;                 #(= (:operation %) "Variable")
;;                 @graph/shadow-graph))))

;; (def x (ops/constant 1))
;; (def y (ops/constant 2))
;; (def z (ops/add x y))
;; (ad/d z x)

(defn gradient [y x]
  (let [g (update @graph/shadow-graph'
                  x ad/wrt)]
    (ad/d y)))


;; (gradient z x)

;; (defn gradient [y x]
;;   (reduce
;;    ops/add
;;    (map
;;     (partial reduce
;;              (fn [gradient node]
;;                ((:chain-fn node)
;;                 (get-registered-gradient node) gradient))
;;              (ops/constant 1.))
;;     (paths y x))))


;; (defn gradients
;;   "The symbolic gradient of y with respect to xs.
;;   For example, if we wanted to calculate the gradient of our
;;   cost function with respect to our weight, we could use."
;;   ([y & xs] (map (partial gradient y) xs))
;;   ([y] (apply (partial gradients y) (relevant-variables y))))


;; (defn numerical-gradients
;;   "Calculate the approximate gradient of y with respect to x
;;   This is best used for testing differentiated gradients."
;;   [y & xs]
;;   (ops/div
;;    (ops/sub
;;     (ops/add (first xs) (ops/constant 0.000001))
;;     (first xs))
;;    (ops/constant 0.000001)))

;; (defn apply-gradients
;;   [variables gradients]
;;   (map #(ops/assign %1 (ops/sub %1 %2))
;;        variables gradients))
