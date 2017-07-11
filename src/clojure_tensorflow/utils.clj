(ns clojure-tensorflow.utils
  (:require [clojure-tensorflow.graph :as graph]
            [clojure-tensorflow.session :as session])
  (:import [org.tensorflow
            Tensor
            Shape
            DataType]))

(defn recursively
  "Apply function to all items in nested data structure if
  condition function is met."
  [apply-if-fn func data]
  (if (apply-if-fn data)
    (func (map (partial recursively apply-if-fn func) data))
    data))

(defn make-coll
  "Make a collection of x,y,z... dimensions"
  [fill & dims]
  (case (count dims)
    0 fill
    1 (repeat (first dims) fill)
    (repeat (first dims) (apply (partial make-coll fill) (rest dims)))
    ))

(def array?
  "Works like coll? but returns true if argument is array"
  #(= \[ (first (.getName (.getClass %)))))

(def tensor->shape
  #(let [arr (.shape %)]
     (if (> (count arr) 0)
       (Shape/make
        (aget arr 0)
        (java.util.Arrays/copyOfRange arr 1 (count arr)))
       (Shape/scalar))))

(defn tf-vals [v]
  "Convert value into type acceptable to TensorFlow
  Persistent data structures become arrays
  Longs become 32bit integers
  Doubles become floats"
  (cond
    (coll? v)
    (if (coll? (first v))
      (to-array (map tf-vals v))
      (case (.getName (type (first v)))
        "java.lang.Long" (int-array v)
        "java.lang.Int" (int-array v)
        "java.lang.Double" (float-array v)
        "java.lang.Float" (float-array v)))
    (= (.getName (type v)) "java.lang.Long") (int v)
    (= (.getName (type v)) "java.lang.Int") (int v)
    (= (.getName (type v)) "java.lang.Double") (float v)
    (= (.getName (type v)) "java.lang.Float") (float v)
    ;; anything else
    true v))


(defn output-shape [tensor]
  (let [shape (tensor->shape tensor)
        dims (map #(.size shape %)
                  (range (.numDimensions shape)))
        d (case (.name (.dataType tensor))
            "INT32" 0
            "INT64" 0
            "FLOAT" 0.0
            "DOUBLE" 0.0)]
    (tf-vals
     (apply (partial make-coll d) dims))))

(defn get-tensor-val [tensor]
  (let [copy-to (output-shape tensor)
        dtype (.name (.dataType tensor))
        ]
    (cond
      (array? copy-to) (.copyTo tensor copy-to)
      (= "FLOAT" dtype) (double (.floatValue tensor))
      (= "DOUBLE" dtype) (.doubleValue tensor)
      (= "INT32" dtype) (long (.intValue tensor))
      (= "INT64" dtype) (.longValue tensor))))


(def tensor->clj (comp (partial recursively array? vec) get-tensor-val))

(def clj->tensor #(Tensor/create (tf-vals %)))

(defn tf-obj? [x]
  (if (re-find #"org.tensorflow" (.getName (class x)))
    true false))

(def to-floats (partial clojure.walk/postwalk #(if (coll? %) % (float %))))


(defn thread
  "Approximately equivalent to -> macro.
  Required because -> must run at compile time"
  [val functions] (reduce #(%2 %1) val functions))

(defprotocol Conversion
  (->clj [x] "Return clojure values")
  (->tensor [x] "Return tensor values")
  (->output [x] "Return tensorflow output object")
  )

(extend-type org.tensorflow.Shape
  Conversion
  (->clj [x] (map (fn [i] (.size x i))
                  (range (.numDimensions x)))))

(extend-type org.tensorflow.Tensor
  Conversion
  (->output [x]
    (-> graph/graph
        (.opBuilder "Const" (str (gensym)))
        (.setAttr "dtype" (.dataType x))
        (.setAttr "value" x)
        .build
        (.output 0)))

  (->clj [x]
    ((comp (partial recursively array? vec) get-tensor-val) x))
  )

(extend-type org.tensorflow.Output
  Conversion
  (->tensor [x]
    (-> session/session .runner (.fetch (.name (.op x))) .run (.get 0))))

(extend-protocol Conversion
  java.lang.Long
  (->tensor [x] (->tensor (int x)))
  java.lang.Double
  (->tensor [x] (->tensor (float x)))
  java.lang.Float
  (->tensor [x] (Tensor/create x))
  java.lang.Integer
  (->tensor [x] (Tensor/create x))
  clojure.lang.PersistentVector
  (->tensor [x] (Tensor/create (tf-vals x)))
  clojure.lang.PersistentList
  (->tensor [x] (->tensor (vec x))))
