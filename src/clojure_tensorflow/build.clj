(ns clojure-tensorflow.build
  (:require
   [clojure-tensorflow.graph
    :refer [graph global-variables shadow-graph shadow-graph']]
   [clojure-tensorflow.utils :as utils]))

(defn op-builder
  "Returns a function which creates an operation for the graph"
  ([op-profile] (op-builder op-profile graph))
  ([op-profile graph]
   (let [{:keys [operation node-name attrs inputs]
          :or {node-name (str (gensym operation)) attrs {} inputs []}
          } op-profile

         ;; convert clj values to tensorflow operations if necessary
         inputs
         (map #(if (= (type %) org.tensorflow.Output)
                 %
                 (op-builder
                  {:operation "Const"
                   :attrs
                   {:dtype
                    (.dataType
                     (utils/clj->tensor %))
                    :value
                    (utils/clj->tensor %)
                    }})) inputs)

         tf-operation
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
                                #(.addInput % input)) inputs)
                         #(.build %)
                         #(.output % 0)]))
         ]
     (swap! shadow-graph conj (assoc op-profile :name node-name :attrs attrs :inputs inputs :tf-op tf-operation))
     (swap! shadow-graph' assoc (keyword node-name) {:operation operation :attrs attrs :inputs (doall (map #(keyword (.name (.op %))) inputs))})
     tf-operation)))


(defn build-op [op-name]
  (let [op-def (op-name @shadow-graph')]
       (-> op-def
           (update :input (partial map build-op))
           op-builder
       )))


;; (defn shadow-builder [definition]
;;   (fn [] 1)
;;   )
