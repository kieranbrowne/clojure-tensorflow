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

(s/valid? :clojure-tensorflow.ops/op-def {:operation "3"})


(defn run
  "Call session runner on single op.
  Returns tensor object"
  ([op-name] (run op-name {}))
  ([op-name feed-map]
   ;; {:pre [(s/valid? ::op-name op-name)]}
   (if (coll? op-name)
     ;; if run on AutoDiff Dual type, run function and its derivative
     (if (= (type op-name) (type (ad/->Dual 1 1)))
       (-> op-name
           (update :f run)
           (update :f' run))
       ;; if run on a list of operations, run all and return the last
       (do (-> session .runner (feed feed-map))
           (last (map #(run % feed-map) (flatten op-name)))))
     ;; if run on a single op return it
     (-> session
         .runner
         (feed feed-map)
         (.fetch (.name (.op op-name)))
         .run
         (.get 0)
         utils/tensor->clj
         ))))



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
          c (ops/add a b)]
      (run
        (build/build-op c))
      ;; (derivative)
      )
    )
  )

(with-graph
  (with-session
    (let [a (ops/constant 1)
          b (ops/constant 2)
          c (ops/mult a b)]
      @shadow-graph'
      ;; (run (ad/d c))
      )
    ))

(def a (ops/constant 9.))
(def b (ops/constant 2.))
(run (ad/d (partial ad/mul a) b))
(deref shadow-graph')

(derivative :Const17695 1)

(defn derivative
  [op wrt]
  (let [g @shadow-graph'
        op (keyword op)
        wrt (keyword wrt)]
    (update g wrt #(ad/coerce % 1))
    (op g)
    ))
