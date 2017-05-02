(ns clojure-tensorflow.build
  (:require [clojure-tensorflow.utils :as utils]))

(def default-graph (new org.tensorflow.Graph))

(def global-variables (atom []))

(defn op-builder
  "Returns a function which creates an operation for the graph"
  ([op-profile] (op-builder op-profile default-graph))
  ([op-profile graph]
   (let [{:keys [operation node-name attrs inputs]
          :or {node-name (str (gensym operation)) attrs {} inputs []}
          } op-profile]
     (utils/thread graph
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
                     #(.output % 0)])))))
