(ns clojure-tensorflow.save
  (:require
   [clojure-tensorflow.gradients
    :refer [relevant-variables get-op-by-name]]
   [clojure-tensorflow.ops :as ops]
   [clojure-tensorflow.core :refer [session-run]]
   ))

(defn collate-vars [sess vars]
  (reduce #(let [node-name %2]
             (assoc %1 (.name (.op %2))
                    (session-run sess [%2])))
          {}
          (flatten (map relevant-variables [v]))))


(defn save-vars
  "Save values of a list of variables or models containing variables
  to a file."
  [sess file-name vars]
  (with-open [w (clojure.java.io/writer file-name)]
    (binding [*out* w]
      (pr (collate-vars sess vars)))))

(defn load-vars
  "Returns a set of assign operations to set variable values from a file"
  [file-name]
  (with-open [r (java.io.PushbackReader.
                 (clojure.java.io/reader file-name))]
    (binding [*read-eval* false]
      (map #(let [[node-name val] %]
              (ops/assign (:tf-op (get-op-by-name node-name))
                         val))
        (read r)))))
