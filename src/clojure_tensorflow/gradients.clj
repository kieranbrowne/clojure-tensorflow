(ns clojure-tensorflow.gradients
  (:require
   [clojure-tensorflow.build :as build]
   [clojure-tensorflow.utils :as utils]
   [clojure-tensorflow.ops :as ops]
   [clojure-tensorflow.ops :as tf]))


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


(defn op-to-op [from to]
  (reverse
   (loop [inputs [from] path []]
     (let [dependents (filter (partial depends-on? to) inputs)]
       (if (empty? dependents)
         path
         (recur (get-inputs (first dependents)) (conj path (first dependents))))))))


(def collate-paths
  (fn [from to path-atom path]
    (let [dependents (filter (partial depends-on? to) (get-inputs from))
          which-dependents (map #(.indexOf (get-inputs from) %) dependents)]
      (if (= from to)
        (swap! path-atom conj (conj path {:output (tf/constant 1.0) :which first}))
        (doall
         (map #(collate-paths %1 to path-atom
                              (conj path {:output from
                                          :which (fn [x] (nth x %2))
                                          ;; :which %2
                                          }))
              dependents which-dependents)))
      )))

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

;; (nth [1 2 ] 0)

;; (let [a (ops/constant 2.)
;;       b (ops/constant 1.)
;;       c (ops/add a b)
;;       d (ops/add b (ops/constant 1.))
;;       e (ops/mult c d)]

;;        ;; (map :which
;;        ;;      (first (paths e b)))
;;   ;; (gradients e a)

;;   ;; (-> 
;;   ;;     (first
;;   ;;      (first
;;   ;;       (map add-positions
;;   ;;            (paths e a))))
;;   ;;     :output
;;   ;;     get-inputs
;;   ;;     first
;;   ;;     .op
;;   ;;     .name
;;   ;;     ;; (fn [ins] (map #(.name (.op %)) ins))
;;   ;;     )
;; ;; ((:which
;; ;;  (second
;; ;;    (add-positions
;; ;;     (first (paths e b))))) (range 1 3))
;;   ;; (gradients e a)
;; )


(defn gradients
  "The symbolic gradient of y with respect to x.
  For example, if we wanted to calculate the gradient of our
  cost function with respect to our weight, we could use
  `(gradients cost weights)`."
  [y & xs]
  (map
   (comp
    #(case (.numDimensions (.shape %)) 2 (ops/sum %) %)
    #(reduce ops/add (map (comp
           (partial reduce ops/mult) ;; chain rule
           (partial map get-registered-gradient))
           (paths y %))
             ))
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
  (map #(ops/assign %1 (ops/sub %1 (ops/transpose %2)))
       xs gradients))


;; Optimizers

(defn gradient-descent
  "The very simplest optimizer."
  [cost-fn & weights]
  (apply-gradients weights (apply gradients (cons cost-fn weights))))
