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

(defn get-node [k]
  (@graph/shadow-graph' k))

(def op-fns
  {:Mul ops/mul
   :Add ops/add
   })
(def get-op-fn (comp op-fns keyword :operation get-node))

(def get-op-inputs (comp :inputs get-node))

(defn parents
  [op-name]
  (set (:inputs (op-name @graph/shadow-graph'))))

(defn ancestors
  [op-name]
  (loop [anc #{} xs [op-name]]
    (let [generation (apply clojure.set/union (map parents xs))]
      (if (empty? generation)
        anc
        (recur
         (reduce clojure.set/union anc (map parents xs))
         generation
         )))))

(defn path [a b]
  (loop [p [] step a]
    (if (not ((ancestors step) b))
      p
      (recur
       (conj p step)
       (first
        (filter #(contains? (ancestors %) b) (get-op-inputs step)))
       )))
  )


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
