(ns clojure-tensorflow.ops
  (:require
   [clojure-tensorflow.build :as build :refer [op-builder]]
   [clojure-tensorflow.utils :as utils]
   ))

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

(defn placeholder [node-name datatype]
  (op-builder
   {:operation "Placeholder"
    :node-name (name node-name)
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
     :attrs {:keep_dims true}
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

(def square #(pow % (constant 2.)))

(defn log [a]
  (op-builder
   {:operation "Log"
    :inputs [a]}))

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
    :inputs [a b]}))

(defn dot-a [a b]
  (op-builder
   {:operation "MatMul"
    :inputs [a b]
    :attrs {:transpose_a true}
    }))

(defn dot-b [a b]
  (op-builder
   {:operation "MatMul"
    :inputs [a b]
    :attrs {:transpose_b true}
    }))

(defn identity [a]
  (op-builder
   {:operation "Identity"
    :inputs [a]}))

(def float32 org.tensorflow.DataType/FLOAT)
(def int32 org.tensorflow.DataType/INT32)
(def int64 org.tensorflow.DataType/INT64)
(def float64 org.tensorflow.DataType/DOUBLE)
