(ns clojure-tensorflow.gradients
  (:require
   [clojure-tensorflow.build :as build]
   [clojure-tensorflow.graph :as graph]
   [clojure-tensorflow.utils :as utils]
   [clojure-tensorflow.ops :as ops]
   [clojure.string :as string]
   [clojure-tensorflow.ops :as tf]))


(defn get-op-by-name [n]
  (first (filter #(= (:name %) n) @graph/shadow-graph)))

(def get-inputs (comp :inputs get-op-by-name #(.name (.op %))))

(defn depends-on?
  [independent dependent]
  (or (some #(= independent %) (get-inputs dependent))
      (some (partial depends-on? independent) (get-inputs dependent))
      (= dependent independent)))



;; Read gradients file
(def parsed-gradients
  (read-string
   (slurp "resources/gradients.edn")))

(nth
 (get parsed-gradients "Abs") 4)

(get parsed-gradients "Sub")

(defn- maybe-update [m k f]
  (if (get m k)
    (update m k f)
    m))

(defn- unroll-gradient [gradient starting-ops return-map?]
  (loop [ops starting-ops
         grad-ops gradient]
      (let [op (-> (first grad-ops)
                   ;; replace with ops calced in previous loops
                   (update :inputs (partial map #(get ops %)))
                   ;; I have a feeling we'll also need to do something with
                   ;; the attributes
                   (update
                    :attrs
                    (fn [attrs]
                      (-> attrs
                          (maybe-update
                           :dtype
                           #(if (= % "$T") (.dataType (get ops "x")) %))
                          (maybe-update
                           :value
                           #(if (= (type %) java.lang.Double) (utils/->tensor %) %))
                          (maybe-update
                           :SrcT
                           #(if (= (str %) "DT_FLOAT") tf/float32 %))
                          (maybe-update
                           :DstT
                           #(if (= (str %) "$T") (.dataType  (get ops "x")) %))
                          )))
                   ;; build operation
                   (build/op-builder))]
        (if (empty? (rest grad-ops))
          (if return-map?
            (assoc ops (:grad-scope-ref (first grad-ops)) op)
            op)
          (recur (assoc ops (:grad-scope-ref (first grad-ops)) op)
                 (rest grad-ops))))))

(defn parsed-grad [op wrt]
  "Return the auto gradient parsed from tensorflow C code"
  (let [pgrad (get parsed-gradients (.type (.op op)))]
    (if pgrad

      (case (:type pgrad)

        "Unary"
        (unroll-gradient
              (:gradient pgrad)
              {"dx" (tf/constant 1.)
               "dy" (tf/constant 1.)
               "x"  (first (get-inputs op))
               "y"  op}
              false)

        "Binary"
        (get (unroll-gradient
              (:gradient pgrad)
              {"dx" (tf/constant 1.)
               "dy" (tf/constant 1.)
               "dz" (tf/constant 1.)
               "x"  (first (get-inputs op))
               "y"  (second (get-inputs op))
               "z"  op}
              true)
             (case (.indexOf (get-inputs op) wrt)
               0 "gx" 1 "gy"
               ))
        )


      ;; if no parsed gradient found throw error
      (throw (ex-info
              (str "No gradient found for operation: "
                   (.type (.op op))) {:operation (.type (.op op))}))
      )))

;; (defn parsed-grad
;;   "Return the auto gradient parsed from tensorflow C code"
;;   [op wrt]
;;   (if (get parsed-gradients (.type (.op op)))
;;     (loop [ops {"y" op
;;                 "x" (first (get-inputs op))
;;                 "y" (second (get-inputs op))
;;                 "dx" (tf/constant 1.)
;;                 "dy" (tf/constant 1.)
;;                 "dz" (tf/constant 1.)
;;                 }
;;            grad-ops (get parsed-gradients (.type (.op op)))]
;;       (let [op (-> (first grad-ops)
;;                    ;; replace with ops calced in previous loops
;;                    (update :inputs (partial map #(get ops %)))
;;                    ;; I have a feeling we'll also need to do something with
;;                    ;; the attributes
;;                    (update
;;                     :attrs
;;                     (fn [attrs]
;;                       (-> attrs
;;                           (maybe-update
;;                            :dtype
;;                            #(if (= % "$T") (.dataType op) %))
;;                           (maybe-update
;;                            :value
;;                            #(if (= (type %) java.lang.Double) (utils/->tensor %) %))
;;                           (maybe-update
;;                            :SrcT
;;                            #(if (= (str %) "DT_FLOAT") tf/float32 %))
;;                           (maybe-update
;;                            :DstT
;;                            #(if (= (str %) "$T") (.dataType op) %))
;;                           )))
;;                    ;; build operation
;;                    (build/op-builder))]
;;         (if (empty? (rest grad-ops))
;;           op
;;           (recur (assoc ops (:grad-scope-ref (first grad-ops)) op)
;;                  (rest grad-ops)))))
;;     (throw (ex-info (str "No gradient found for operation: " (.type (.op op)))))
;;     ))



(def registered-gradients
  (atom {"Const" [(fn [& in] (ops/constant 1.))]
         "Pow" [(fn [& in] (ops/mult (second in) (ops/pow (first in) (ops/sub (second in) (ops/constant 1.)))))
                (fn [& in] (ops/mult (ops/log (first in)) (ops/pow (first in) (second in))))]
         "Variable" [(fn [& in] (ops/constant 1.))]
         "Reshape" [(fn [& in] (first in))]
         "MaxPool" [(fn [& in] (ops/constant 1.))]
         "Add" (repeat 2 (fn [& in] (ops/constant 1.)))
         "Sub" [(fn [& in] (ops/constant 1.))
                (fn [& in] (ops/constant -1.))]
         "MatMul" [(fn [& in] (second in))
                   (fn [& in] (first in))]
         "Conv2D" [(fn [& in] (second in))
                   (fn [& in] (first in))]
         "Mul" [(fn [& in] (second in))
                (fn [& in] (first in))]
         "Div" [(fn [& in] (ops/pow (second in) (ops/constant -1.)))
                (fn [& in] (ops/sub (ops/constant 0.)
                                    (ops/mult (first in)
                                                (ops/pow (second in) (ops/constant -2.)))))]
         ;; TODO
         ;; "Transpose" [(fn [& in] (first in)
         ;;               #_(prn "IN" in)
         ;;               #_(ops/transpose (first in)))]
         "Sum" [(fn [& in] (ops/constant 1.))]
         "Sigmoid" [(fn [& in]
                      (ops/mult
                       (ops/sigmoid (first in))
                       (ops/sub (ops/constant 1.)
                                (ops/sigmoid (first in)))))]
         "Relu" [(fn [& in]
                   (ops/minimum (ops/constant 1.)
                    (ops/maximum (first in) (ops/constant 0.))))]
         }))

(defn register-gradient [op-type functions]
  (swap! registered-gradients op-type functions))

(defn get-registered-gradient
  [node]
  (let [{output :output which :which} node
        grad-type (.type (.op output))
        grad (get @registered-gradients grad-type)]
    (when-not grad
      (throw (ex-info (str "No gradient found for operation: \"" grad-type "\". Try registering one with clojure-tensorflow.gradients/register-gradient") {:type grad-type})))
    (apply (which grad) (get-inputs output))))

(defn- collate-paths [from to path-atom path]
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
                   "Conv2D" (if (= 0 %2)
                              (comp ops/transpose ops/dot-b)
                              ops/dot-a)
                   "Reshape"
                   (fn [& x] (ops/reshape
                             (first x)
                           (utils/->clj
                            (.shape (first (get-inputs from))))))


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
                @graph/shadow-graph))))

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
  [variables gradients]
  (map #(ops/assign %1 (ops/sub %1 %2))
       variables gradients))
