(ns clojure-tensorflow.gradients
  (:require
   [clojure-tensorflow.build :as build]
   [clojure-tensorflow.utils :as utils]
   [clojure-tensorflow.ops :as ops]
   [clojure-tensorflow.ops :as tf]
   [clojure-tensorflow.gradients :as tf.optimizers]
   ))


(defn get-op-by-name [n]
  (first (filter #(= (:name %) n) @build/shadow-graph)))

(def get-inputs (comp :inputs get-op-by-name #(.name (.op %))))

(defn depends-on?
  [independent dependent]
  (or (some #(= independent %) (get-inputs dependent))
      (some (partial depends-on? independent) (get-inputs dependent))
      (= dependent independent)))


(defn get-registered-gradient
  ([node ]
   (let [{output :output which :which} node
         inputs (get-inputs output)]
    (case (.type (.op output))
      "Const" (ops/constant 1.)
      "Variable" (ops/constant 1.)
      "Add" (ops/constant 1.)
      "Sub" (which [(ops/constant 1.)
                    (ops/constant -1.)])
      "Pow" (which [
             (ops/mult (second inputs) (ops/pow (first inputs)  (ops/sub (second inputs) (ops/constant 1.))))
             (ops/mult (ops/log (first inputs)) (ops/pow (first inputs) (second inputs)))
             ])
      "MatMul" (which (reverse inputs)) ;; need to work out which is important
      "Mul" (which (reverse inputs))
      "Div" (which (reverse inputs))
      "Sigmoid" (ops/mult (ops/sigmoid (first inputs))
                          (ops/sub (ops/constant 1.) (ops/sigmoid (first inputs))))
      "Mean" (ops/size (first inputs)) ;; need to work out which is important
      "Sum" (first inputs)
      ))))


(def collate-paths
  (fn [from to path-atom path]
    (let [dependents (filter (partial depends-on? to) (get-inputs from))
          which-dependents (map #(.indexOf (get-inputs from) %) dependents)]
      (if (= from to)
        (swap! path-atom conj (conj path {:output (tf/constant 1.0) :which first :chain-fn tf/mult}))
        (doall
         (map #(collate-paths %1 to path-atom
                              (conj path {:output from
                                          :which (fn [x] (nth x %2))
                                          :chain-fn (if (= "MatMul" (.type (.op from)))
                                                      (fn [a b] (tf/matmul (tf/transpose a) b))
                                                      ;; (fn [a b] (tf/matmul b (tf/transpose a)))
                                                      ;; (fn [a b] (tf/matmul a b))
                                                      tf/mult)
                                          ;; :chain-fn (if (some (partial = "MatMul") (map (fn [x] (.type (.op x))) (get-inputs from))) tf/matmul tf/mult)
                                          ;; :chain-fn ( (partial = "MatMul") (map (fn [x] (.type (.op x))) (get-inputs from)))
                                          }))
              dependents which-dependents)))
      )))

(contains? #{1} [5 6 7])


(defn paths
  "Get all paths from one op to another"
  [from to]
  (let [paths (atom [])]
    (collate-paths from to paths [])
    @paths
    ))


(defn add-positions [path]
     (map
      (fn [i]
        {:which #(nth %
                      (.indexOf (get-inputs
                                 (nth path (max (count path) (inc i))))
                                (nth path i)))
         :output (nth path i)}) (range (count path))))

(defn gradients
  "The symbolic gradient of y with respect to x.
  For example, if we wanted to calculate the gradient of our
  cost function with respect to our weight, we could use
  `(gradients cost weights)`."
  [y & xs]
  (map (fn [x]
         (reduce tf/add
                 (map
                  (comp
                   ;; #(case (.numDimensions (.shape %)) 2 (ops/sum %) %)
                   #(reduce (fn [gradient node]
                              ((:chain-fn node)
                               (tf.optimizers/get-registered-gradient node) gradient)
                              )
                            (tf/constant 1.) %))
                  (paths y x))))
       xs))


(defn numerical-gradients
  "Calculate the approximate gradient of y with respect to x
  This is best used for testing differentiated gradients."
  [y & xs]
  (ops/div
   (ops/sub
    (ops/add (first xs) (ops/constant 0.000001))
    (first xs))
   (ops/constant 0.000001))
  )

(defn apply-gradients
  [xs gradients]
  (map #(ops/assign %1 (ops/sub %1 %2))
       xs gradients))


;; Optimizers

(defn gradient-descent
  "The very simplest optimizer."
  [cost-fn & weights]
  (apply-gradients weights (apply gradients (cons cost-fn weights))))
