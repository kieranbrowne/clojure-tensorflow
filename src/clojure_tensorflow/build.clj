(ns clojure-tensorflow.build
  (:require [clojure-tensorflow.utils :as utils]))

(def default-graph (new org.tensorflow.Graph))
(def ^:dynamic graph default-graph)

;; As a design choice from tensorflow graph needs to be
(def global-variables (atom []))

;; The shadow graph patches a couple of requirements while we wait for
;; the Java api. We store all relevant information about operations as
;; we add them to the graph. Eventually all of this will be extractable
;; from the java graph / operations objects.
(def shadow-graph (atom []))

(defn op-builder
  "Returns a function which creates an operation for the graph"
  ([op-profile] (op-builder op-profile graph))
  ([op-profile graph]
   (let [{:keys [operation node-name attrs inputs]
          :or {node-name (str (gensym operation)) attrs {} inputs []}
          } op-profile
         tf-operation (utils/thread graph
                       (flatten
                        [#(.opBuilder % operation node-name)
                         ;; set attributes if any
                         (map
                          (fn [attr]
                            #(.setAttr % (name (first attr)) (second attr)))
                          attrs)
                         ;; add inputs if any
                         (map (fn [input]
                                #(.addInput %
                                            (if (fn? input) (input graph) input)
                                            )) inputs)
                         #(.build %)
                         #(.output % 0)]))
         ]
     (swap! shadow-graph conj (assoc op-profile :name node-name :attrs attrs :inputs inputs :tf-op tf-operation))
     tf-operation)))

