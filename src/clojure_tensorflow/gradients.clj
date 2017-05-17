(ns clojure-tensorflow.gradients
  (:require
   [clojure-tensorflow.build :as build]
   [clojure-tensorflow.utils :as utils]
   [clojure-tensorflow.ops :as ops]))


(defn get-op-by-name [n]
  (first (filter #(= (:name %) n) @build/shadow-graph)))

(def get-inputs (comp :inputs get-op-by-name #(.name (.op %))))

(defn depends-on?
  [independent dependent]
  (or (some #(= independent %) (get-inputs dependent))
      (some (partial depends-on? independent) (get-inputs dependent))
      (= dependent independent)))

(def registered-gradients
  (atom {"Const" [(fn [& in] (ops/constant 1.))]
         "Pow" [(fn [& in] (ops/mult (second in) (ops/pow (first in) (ops/sub (second in) (ops/constant 1.)))))
                (fn [& in] (ops/mult (ops/log (first in)) (ops/pow (first in) (second in))))]
         "Variable" [(fn [& in] (ops/constant 1.))]
         "Add" (repeat 2 (fn [& in] (ops/constant 1.)))
         "Sub" [(fn [& in] (ops/constant 1.))
                (fn [& in] (ops/constant -1.))]
         "MatMul" [(fn [& in] (second in))
                   (fn [& in] (first in))]
         "Mul" [(fn [& in] (second in))
                (fn [& in] (first in))]
         "Div" [(fn [& in] (second in))
                (fn [& in] (first in))]
         "Sigmoid" [(fn [& in] (ops/mult (ops/sigmoid (first in))
                                        (ops/sub (ops/constant 1.) (ops/sigmoid (first in)))))]
         }))

(defn register-gradient [op-type functions]
  (swap! registered-gradients op-type functions))

(defn get-registered-gradient
  [node]
  (let [{output :output which :which} node]
    (apply (which (get @registered-gradients (.type (.op output)))) (get-inputs output))))

(defn collate-paths [from to path-atom path]
  (let [dependents (filter (partial depends-on? to) (get-inputs from))
        which-dependents (map #(.indexOf (get-inputs from) %) dependents)]
    (if (= from to)
      (swap! path-atom conj
             (conj path {:output (ops/constant 1.0)
                         :which first
                         :chain-fn ops/mult}))
      (doall
       (map
        #(collate-paths
          %1 to path-atom
          (conj path
                {:output from
                 :which (fn [x] (nth x %2))
                 :chain-fn
                 (case (.type (.op from))
                   "MatMul" (if (= 0 %2)
                              (comp ops/transpose ops/dot-b)
                              ops/dot-a)
                   ops/mult)}))
        dependents which-dependents)))))

(defn paths
  "Get all paths from one op to another"
  [from to]
  (let [paths (atom [])]
    (collate-paths from to paths [])
    @paths))

(defn relevant-variables [op]
  (filter #(depends-on? % op)
          (map :tf-op
               (filter
                #(= (:operation %) "Variable")
                @build/shadow-graph))))

(defn gradient [y x]
  (reduce
   ops/add
   (map
    (partial reduce
             (fn [gradient node]
               ((:chain-fn node)
                (get-registered-gradient node) gradient))
             (ops/constant 1.))
    (paths y x))))

(defn gradients
  "The symbolic gradient of y with respect to xs.
  For example, if we wanted to calculate the gradient of our
  cost function with respect to our weight, we could use."
  ([y & xs] (map (partial gradient y) xs))
  ([y] (apply (partial gradients y) (relevant-variables y))))


(defn numerical-gradients
  "Calculate the approximate gradient of y with respect to x
  This is best used for testing differentiated gradients."
  [y & xs]
  (ops/div
   (ops/sub
    (ops/add (first xs) (ops/constant 0.000001))
    (first xs))
   (ops/constant 0.000001)))

(defn apply-gradients
  [xs gradients]
  (map #(ops/assign %1 (ops/sub %1 %2))
       xs gradients))
