(ns clojure-tensorflow.build
  (:require
   [clojure-tensorflow.graph
    :refer [graph global-variables shadow-graph shadow-graph']]
   [clojure-tensorflow.utils :as utils]
   [autodiff.protocols :as ad]

   [clojure-tensorflow.graph :as graph])
  (:import [autodiff.protocols.Dual]))


(defn op-builder
  "Returns a function which creates an operation for the graph"
  ([op-profile] (op-builder op-profile graph))
  ([op-profile graph]
   (let [{:keys [operation node-name attrs inputs]
          :or {node-name (str (gensym operation)) attrs {} inputs []}
          } op-profile

         ;; convert clj values to tensorflow operations if necessary
         ;; inputs
         ;; (map #(if (= (type %) org.tensorflow.Output)
         ;;         %
         ;;         (op-builder
         ;;          {:operation "Const"
         ;;           :attrs
         ;;           {:dtype
         ;;            (.dataType
         ;;             (utils/clj->tensor %))
         ;;            :value
         ;;            (utils/clj->tensor %)
         ;;            }})) inputs)

         tf-operation
         (utils/thread graph
              (flatten
               [#(.opBuilder % (name operation) (name node-name))
                ;; set attributes if any
                (map
                 (fn [attr]
                   #(.setAttr % (name (first attr))
                              (second attr)))
                 attrs)
                ;; add inputs if any
                (map (fn [input]
                       #(.addInput % (or (:f input) input))) inputs)
                #(.build %)
                #(.output % 0)]))
         ]
     ;; (swap! shadow-graph conj (assoc op-profile :name node-name :attrs attrs :inputs inputs :tf-op tf-operation))
     ;; (swap! shadow-graph' assoc (keyword node-name) {:operation operation :attrs attrs :inputs (doall (map #(keyword (.name (.op %))) inputs))})
     tf-operation)
   ))


(defmulti build-op class)

(defmethod build-op
  clojure.lang.Keyword
  [op-name]
  (or ;; if already on the graph just return it
   (try (.output (.operation graph (name op-name)) 0)
        (catch Exception e))
      ;; if op not built yet build it
      (-> (op-name @shadow-graph')
          ;; ensure all inputs to op have been built
          (update :inputs (partial map build-op))
          (assoc :node-name op-name)
          op-builder
          )))

(defmethod build-op
  autodiff.protocols.Dual
  [dual-op]
  (-> dual-op
      (update :f build-op)
      (update :f' build-op)))


(defmethod build-op
  org.tensorflow.Output
  [o] o)
(defmethod build-op
  org.tensorflow.Operation
  [o] o)

(defmethod build-op ;; fallback
  ::any
  [o]
  (build-op
   (graph/add-shadow-op
    {:operation "Const"
     :attrs {:dtype (.dataType (utils/clj->tensor o))
             :value (utils/clj->tensor o)
             }})))
