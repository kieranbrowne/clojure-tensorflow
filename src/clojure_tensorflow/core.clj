(ns clojure-tensorflow.core
  (:require [clojure-tensorflow.utils :as utils]
            [clojure-tensorflow.ops :as ops]
            [clojure-tensorflow.graph
             :refer [graph global-variables shadow-graph shadow-graph']]
            [clojure-tensorflow.session :refer [session]]
            [clojure.spec.alpha :as s]
            [autodiff.protocols :as ad]
            [clojure-tensorflow.build :as build]))



(defn feed
  "Feed value to placeholder
  Pass a map of locations to values"
  ([runner feed-map]
   (utils/thread
     runner
     (map (fn [[key val]]
            #(.feed % (name key) (utils/clj->tensor val))) feed-map))))


(defmulti get-name class)
(defmethod get-name org.tensorflow.Output [o] (-> o .op .name))
(defmethod get-name org.tensorflow.Operation [op] (-> op .name))


(defn run
  "Call session runner on single op.
  Returns tensor object"
  ([op-name] (run op-name {}))
  ([op feed-map]
   ;; {:pre [(s/valid? ::op-name op-name)]}
   (if (coll? op)
     ;; if run on AutoDiff Dual type, run function and its derivative
     (if (= (type op) (type (ad/->Dual 1 1)))
       (-> op
           (update :f run)
           (update :f' run))
       ;; if run on a list of operations, run all and return the last
       (do (-> session .runner (feed feed-map))
           (last (map #(run % feed-map) (flatten op)))))
     ;; if run on a single op return it
     (do
       (build/build-op op)
       (-> session
           .runner
           (feed feed-map)
           (.fetch (get-name (build/build-op op)))
           .run
           (.get 0)
           utils/tensor->clj
           )))))



(defmacro with-graph [& body]
  `(binding [graph (org.tensorflow.Graph.)
             global-variables (atom [])
             shadow-graph (atom [])
             shadow-graph' (atom {})]
     (try ~@body
       (finally (.close graph)))))

(defmacro with-session [& body]
  `(binding [session (org.tensorflow.Session. graph)]
     (try ~@body
       (finally (.close session)))))



(with-graph
  (with-session
    (let [a (ops/constant 2.)
          b (ops/constant 3.)
          d (ops/placeholder org.tensorflow.DataType/FLOAT)
          c (ops/add a d)
          ]
      (run
            c
            ;; (.name a)
            ;; (build/build-op c)
            ;; (build/build-op d)
        {d 1.}
        )
      ;; (build/build-op
      ;;  (ops/placeholder org.tensorflow.DataType/FLOAT))
      ;; (get-name (build/build-op c))
      ;; (build/build-op c)
      ;; (build/build-op c)
      ;; d
      ;; (derivative)
      )
    )
  )

(with-graph
  (with-session
    (let []
      @shadow-graph'
      ;; (run (ad/d c))
      )
    ))
