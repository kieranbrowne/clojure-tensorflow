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

(defn identity [a]
  (op-builder
   {:operation "Identity"
    :inputs [a]}))


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

(def registered-gradients
  {"Const" (partial constant 1.)
   "Pow" #(mult %1 %2)
   "MatMul" #(mult %1 %2)
   "Sigmoid" #(mult % (sub (constant 1.) %))
   "Sub" #(mult %1 %2)
   "Mean" mean
   })

(defn get-registered-gradient [output]
  (let [inputs (get-inputs output)]
    (case (.type (.op output))
      "Const" (constant 1.)
      "Variable" (constant 1.)
      "Sub" (constant 1.)
      ;; "Pow" (mult )
      "MatMul" (first inputs) ;; need to work out which is important
      "Mult" (first inputs) ;; need to work out which is important
      "Sigmoid" (mult (first inputs) (sub (constant 1.) (first inputs)))
      )))

(defn input-path [x y]
  (reduce (map ()) (:inputs x))
  )

(defn get-op-by-name [n]
  (first (filter #(= (:name %) n) @build/shadow-graph)))
(def get-inputs (comp :inputs get-op-by-name #(.name (.op %))))

(defn depends-on? [independent dependent]
  (or (some #(= independent %) (get-inputs dependent))
      (some (partial depends-on? independent) (get-inputs dependent))
      (= dependent independent)
  ))

(get-registered-gradient mu)
(def mu (tf/matmul xzc xzc))
(def xzc (tf/constant 1.))

(depends-on? xzc (tf/mult (tf/constant 1.) (tf/mult (tf/constant 1.) (tf/constant 1.))))
(depends-on? xzc xzc)

;; general rule is that we sum the (chain rule) derivatives of all paths
;; from weights to cost
(defn paths [from to])

(defn op-to-op [from to]
  (reverse
   (loop [inputs [from] path []]
     (let [dependents (filter (partial depends-on? to) inputs)]
       (if (empty? dependents)
         path
         (recur (get-inputs (first dependents)) (conj path (first dependents))))))))

(map #(.name (.op %))
     (op-to-op (tf/mult (tf/constant 1.) (tf/mult (tf/constant 1.) xzc)) xzc))


(defn gradients
  "The symbolic gradient of y with respect to x.
  For example, if we wanted to calculate the gradient of our
  loss function with respect to our weight, we could use
  `(gradients loss weights)`."
  [y & xs]
  (reduce tf/mult
          (map tf/get-registered-gradient
               (tf/op-to-op y (first xs))))
  )




(defn register-gradient [x y]
  1
  )

;; e.g the gradient of ops
;; (register-gradient ["Pow" (mult b a)])


;; Optimizers

;; I'm going to take the dodgy route for now and just use
;; numerical gradients.
;; TODO: Replace with computed gradients
;; (defn gradient-descent [target coefficient minimize]
;;   (tf/assign
;;    target
;;    (tf/sub target
;;            (tf/matmul
;;             (tf/transpose coefficient)
;;             (tf/numerical-gradient minimize target)))))
