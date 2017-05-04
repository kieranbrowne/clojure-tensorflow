(ns clojure-tensorflow.ops
  (:require
   [clojure-tensorflow.build :as build :refer [op-builder]]
   [clojure-tensorflow.utils :as utils]
   [clojure-tensorflow.ops :as tf]))

(defn global-variables-initializer []
  @build/global-variables)

;; value ops

(defn constant [val]
  (let [tensor (utils/clj->tensor val)]
    (op-builder
     {:operation "Const"
      :attrs {:dtype (.dataType tensor)
              :value tensor
              }})))

(defn assign [var val]
  (op-builder
   {:operation "Assign"
    :inputs [var (if (utils/tf-obj? val) val (constant val))]
    }))

(defn variable
  ([val] (variable val {}))
  ([val bits]
   (let [tensor (utils/clj->tensor val)
         var (op-builder
          (merge
           {:operation "Variable"
            :attrs {:shape (utils/tensor->shape tensor)
                    :dtype (.dataType tensor)}
            } bits))]
     (swap! build/global-variables conj (assign var val))
     var)))

(defn placeholder [datatype]
  (op-builder
   {:operation "Placeholder"
    :attrs {:dtype datatype}
    }))


;; math ops

(defn mult [a b]
  (op-builder
   {:operation "Mul"
    :inputs [a b]}))

(defn div [a b]
  (op-builder
   {:operation "Div"
    :inputs [a b]}))

(defn add [a b]
  (op-builder
   {:operation "Add"
    :inputs [a b]}))

(defn sub [a b]
  (op-builder
   {:operation "Sub"
    :inputs [a b]}))

(defn sum
  ([t] (sum t (constant 0)))
  ([t dims]
   (op-builder
    {:operation "Sum"
     :inputs [t dims]})))

(defn tanh [a]
  (op-builder
   {:operation "Tanh"
    :inputs [a]}))

(defn sigmoid [a]
  (op-builder
   {:operation "Sigmoid"
    :inputs [a]}))

(defn pow [a b]
  (op-builder
   {:operation "Pow"
    :inputs [a b]}))

(defn size [a]
  (op-builder
   {:operation "Size"
    :inputs [a]}))

(defn abs [a]
  (op-builder
   {:operation "Abs"
    :inputs [a]}))

(defn mean [a]
  (op-builder
   {:operation "Mean"
    :inputs [a (constant 0)]}))

(defn transpose [a]
  (op-builder
   {:operation "Transpose"
    :inputs [a (constant [1 0])]}))

(defn matmul [a b]
  (op-builder
   {:operation "MatMul"
    :inputs [a b]})
  )


;; Gradients
(defn numerical-gradient
  "Calculate the approximate gradient of function with respect to x
  This is best used for testing differentiated gradients."
  ([func x accuracy]
   (tf/div
    (tf/sub
     (func (tf/add x accuracy))
     (func x))
    accuracy)
    )
  ([func x] (numerical-gradient func x (tf/constant 0.000001))))


;; Optimizers

;; I'm going to take the dodgy route for now and just use
;; numerical gradients.
;; TODO: Replace with computed gradients
(defn gradient-descent [target coefficient minimize]
  (tf/assign
   target
   (tf/sub target
           (tf/matmul
            (tf/transpose coefficient)
            (tf/numerical-gradient minimize target)))))
