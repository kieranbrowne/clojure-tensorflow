(ns clojure-tensorflow.graph)

(def ^:dynamic graph (new org.tensorflow.Graph))

;; As a design choice from tensorflow graph needs to be
(def ^:dynamic global-variables (atom []))

;; The shadow graph patches a couple of requirements while we wait for
;; the Java api. We store all relevant information about operations as
;; we add them to the graph. Eventually all of this will be extractable
;; from the java graph / operations objects.
(def ^:dynamic shadow-graph (atom []))
(def ^:dynamic shadow-graph' (atom {}))

(defn add-shadow-op
  "Coerce op-def, add to shadow graph and return its key"
  ([op-def op-name]
   (when-not (contains? @shadow-graph' op-name)
     (swap! shadow-graph' assoc op-name op-def))
   op-name)
  ([op-def] (add-shadow-op op-def (keyword (gensym (:operation op-def)))))
  )
