(ns clojure-tensorflow.core
  (:require [clojure-tensorflow.utils :as utils]
            [clojure-tensorflow.ops :as ops]
            [clojure-tensorflow.graph
             :refer [graph global-variables shadow-graph]]
            [clojure-tensorflow.session :refer [session]]))



(defn feed
  "Feed value to placeholder
  Pass a map of locations to values"
  ([runner feed-map]
   (utils/thread
     runner
     (map (fn [[key val]]
            #(.feed % (name key) (utils/clj->tensor val))) feed-map))))


(defn run
  "Call session runner on single op.
  Returns tensor object"
  ([op] (run op {}))
  ([op feed-map]
   (if (coll? op)
     ;; if run on a list of operations, run all and return the last
     (do (-> session .runner (feed feed-map))
         (last (map #(run % feed-map) (flatten op))))
     ;; if run on a single op return it
     (-> session
         .runner
         (feed feed-map)
         (.fetch (.name (.op op)))
         .run
         (.get 0)
         utils/tensor->clj
         ))))


(defmacro with-graph [& body]
  `(binding [graph (org.tensorflow.Graph.)
             global-variables (atom [])
             shadow-graph (atom [])]
     (try ~@body
       (finally (.close graph)))))

(defmacro with-session [& body]
  `(binding [session (org.tensorflow.Session. graph)]
     (try ~@body
       (finally (.close session)))))
