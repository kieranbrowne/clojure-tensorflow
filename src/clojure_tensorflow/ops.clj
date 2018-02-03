(ns clojure-tensorflow.ops
  (:refer-clojure :exclude [cast concat identity])
  (:require
   [clojure-tensorflow.build :as build :refer [op-builder]]
   [clojure-tensorflow.graph :as graph :refer [add-shadow-op]]
   [clojure-tensorflow.utils :as utils]
   [autodiff.protocols :as ad]
   [autodiff.core :refer [extend-types]]
   [clojure.spec.alpha :as s]

   [clojure-tensorflow.ops :as tf])
  (:import [autodiff.protocols.AutoDiff]))


(defn global-variables-initializer []
  @graph/global-variables)

(s/def ::operation (s/or :name string? :key keyword?))
(s/def ::attrs map?)
(s/def ::op-name keyword?)
(s/def ::ad-fn
  (-> ad/AutoDiff :impls (get clojure.lang.Keyword) keys set))
(s/def ::dual-op
  (s/keys :req-un [::f ::f']))

(s/def ::inputs (s/coll-of (s/or :single ::op-name
                                 :dual ::dual-op)))

(s/valid? ::inputs [:1 ])

(s/def ::op-def
  (s/keys :req-un  [::operation]
          :opt-un  [::attrs ::inputs ::ad-fn]))

(defn rebuild-op [op-name]
  (try
    (let [{:keys [ad-fn inputs]} (op-name @graph/shadow-graph')]
      (apply
       (-> ad/AutoDiff :impls (get (class (first inputs))) ad-fn)
       inputs))
    (catch java.lang.NullPointerException e
      (str "test")
      ;; (throw (Exception. "my exception message"))
      '())
    ;; (catch java.lang.IllegalArgumentException e
    ;;   (str "caught exception: " (.getMessage e)))
    ;; (catch Exception e
    ;;   (str "caught exception: " (.getMessage e)))
    ))


(defn prime [op-name]
  {:pre [(s/valid? ::op-name op-name)]
   :post [(s/valid? ::op-name %)]}
  (keyword (str (name op-name) "'")))

(defn leibniz-notation [y x]
  {:pre [(s/valid? ::op-name y) (s/valid? ::op-name x)]
   :post [(s/valid? ::op-name %)]}
  (keyword (str "d" (name y) "/d" (name x))))



(defmacro pull [ns vlist]
  `(do ~@(for [i vlist]
           `(def ~i ~(symbol (str ns "/" i))))))

;; (pull autodiff.protocols (mul add sub sigmoid negate))

(defn constant
  ([val name]
   (let [tensor (utils/clj->tensor val)]
     (add-shadow-op
      {:operation "Const"
       :attrs {:dtype (.dataType tensor)
               :value tensor
               }} name)))
  ([val]
   (let [tensor (utils/clj->tensor val)]
     (add-shadow-op
      {:operation "Const"
       :attrs {:dtype (.dataType tensor)
               :value tensor
               }}))))


(extend-type
    clojure.lang.Keyword

  ad/AutoDiff

  (add [a b]
    (if (and (keyword? a) (keyword? b))
      (add-shadow-op
       {:operation "Add"
        :inputs [a b]})
      (ad/add (ad/coerce a) (ad/coerce b))))

  (mul [a b]
    (if (and (keyword? a) (keyword? b))
      (add-shadow-op
       {:operation "Mul"
        :inputs [a b]})
      (ad/mul (ad/coerce a) (ad/coerce b))
      ))
  (div [a b]
    (if (and (keyword? a) (keyword? b))
      (add-shadow-op
       {:operation "Div"
        :inputs [a b]})
      (ad/div (ad/coerce a) (ad/coerce b))
      ))
  (sigmoid [a]
    (add-shadow-op
     {:operation "Sigmoid"
      :inputs [a]}))
  (negate [a]
    (add-shadow-op
     {:operation "Neg"
      :inputs [a]
      :ad-fn :negate}))

  (sub [a b]
    (if (and (keyword? a) (keyword? b))
      (add-shadow-op
       {:operation "Sub"
        :inputs [a b]})
      (ad/sub (ad/coerce a) (ad/coerce b))))

  (pow [a b]
    (if (and (keyword? a) (keyword? b))
      (add-shadow-op
       {:operation "Pow"
        :inputs [a b]})
      (ad/pow (ad/coerce a) (ad/coerce b))))
  (log [a]
    (add-shadow-op
     {:operation "Log"
      :inputs [a]}))

  (val-of-type [t v]
    (case v
      0 (constant v :zero)
      1 (constant v :one)
      (constant v)))
  (val-like [t v]
    (ad/add
     (ad/mul t (constant 0.))
     (constant (float v)))
    )
  )
;; value ops


(defn assign [var val]
  (add-shadow-op
   {:operation "Assign"
    :inputs [var (if (utils/tf-obj? val) val (constant val))]
    }))

(defn variable
  ([val] (variable val {}))
  ([val bits]
   (let [tensor (utils/clj->tensor val)
         var (add-shadow-op
          (merge
           {:operation "Variable"
            :attrs {:shape (utils/tensor->shape tensor)
                    :dtype (.dataType tensor)}
            } bits))]
     (swap! graph/global-variables conj (assign var val))
     var)))

(defn placeholder [datatype]
  (add-shadow-op
   {:operation "Placeholder"
    ;; :node-name (name node-name)
    :attrs {:dtype datatype}
    }))


(defn tf-val? [x]
  (or (= (type x) org.tensorflow.Output)
      (= (type x) org.tensorflow.Operation)))







;; math ops



(defn sum
  ([t] (sum t (constant 0) false))
  ([t dims] (sum t (constant 0) false))
  ([t dims keep-dims]
   (op-builder
    {:operation "Sum"
     :attrs {:keep_dims keep-dims}
     :inputs [t dims]})))

(defn tanh [a]
  (add-shadow-op
   {:operation "Tanh"
    :inputs [a]}))

(defn relu [a]
  (add-shadow-op
   {:operation "Relu"
    :inputs [a]}))

(defn softmax [a]
  (add-shadow-op
   {:operation "Softmax"
    :inputs [a]}))

(defn maximum [a b]
  (add-shadow-op
   {:operation "Maximum"
    :inputs [a b]}))

(defn reduce-max [a]
  (add-shadow-op
   {:operation "Max"
    :inputs [a (constant -1)]}))

(defn minimum [a b]
  (add-shadow-op
   {:operation "Minimum"
    :inputs [a b]}))

(defn gather [params indices]
  (add-shadow-op
   {:operation "Gather"
    :attrs {:validate_indices true}
    :inputs [params indices]}))

(defn slice [input begin size]
  (add-shadow-op
   {:operation "Slice"
    :inputs [input begin size]}))

(defn pad [input paddings]
  (add-shadow-op
   {:operation "Pad"
    :inputs [input paddings]}))

(defn reshape [input shape]
  (add-shadow-op
   {:operation "Reshape"
    :inputs [input shape]}))

(defn concat [tensors axis]
  (add-shadow-op
   {:operation "Concat"
    :inputs [tensors axis]}))

(defn size [a]
  (add-shadow-op
   {:operation "Size"
    :inputs [a]}))

(defn abs [a]
  (add-shadow-op
   {:operation "Abs"
    :inputs [a]}))

(defn mean [a]
  (add-shadow-op
   {:operation "Mean"
    :inputs [a (constant 0)]}))

(defn size [a]
  (add-shadow-op
   {:operation "Size"
    :inputs [a]}))

(defn transpose [a]
  (add-shadow-op
   {:operation "Transpose"
    :inputs [a (constant [1 0])]}))

(defn matmul [a b]
  (add-shadow-op
   {:operation "MatMul"
    :inputs [a b]}))

(defn dot-a [a b]
  (add-shadow-op
   {:operation "MatMul"
    :inputs [a b]
    :attrs {:transpose_a true}
    }))

(defn dot-b [a b]
  (add-shadow-op
   {:operation "MatMul"
    :inputs [a b]
    :attrs {:transpose_b true}
    }))

(defn identity [a]
  (add-shadow-op
   {:operation "Identity"
    :inputs [a]}))

(defn unstack
  ([value num axis]
   (add-shadow-op
    {:operation "Unpack"
     :inputs [value]
     :attrs {:num num
             :axis axis}}))
  ([value num] (unstack value num 0)))

(defn stack
  ([value axis]
   (add-shadow-op
    {:operation "Pack"
     :inputs [value]
     :attrs {:axis axis}}
    ))
  ([value] (stack value 0)))

(def float32 org.tensorflow.DataType/FLOAT)
(def int32 org.tensorflow.DataType/INT32)
(def int64 org.tensorflow.DataType/INT64)
(def float64 org.tensorflow.DataType/DOUBLE)


(defn cast [a dtype]
  (add-shadow-op
   {:operation "Cast"
    :inputs [a]
    :attrs {:DstT dtype}
    }))

(def to-float #(cast % float32))
(def to-int32 #(cast % int32))


(defn one-hot
  ([indices depth on-value off-value]
   (add-shadow-op
    {:operation "OneHot"
     :inputs [(to-int32 indices) (to-int32 depth) on-value off-value]}))
  ([indices depth] (one-hot indices depth (constant 1) (constant 0)))
  ([indices] (one-hot indices (ad/add (constant 1) (to-int32 (reduce-max indices))) (constant 1) (constant 0)))
  )

(defn random-normal
  "Generate a tensor of random values with a normal distribution"
  ([shape stddev]
   (let [source (java.util.Random. (rand))]
     ((reduce #(partial repeatedly %2 %1)
              #(.nextGaussian source)
              (reverse shape)))))
  ([shape] (random-normal shape 0.35)))

(pull autodiff.protocols (negate))

(defn auto-convert [x] (if (keyword? x) x (constant x)))
(defn check-args [& args] (map #(if (or (keyword? %) (ad/dual? %)) % (constant %)) args))
(defn applicative [fn] (partial apply fn))

(defn mul [a b]
  (apply ad/mul (map auto-convert [a b]))
  )

;; (defn add [a b]
;;   (apply ad/mul (map auto-convert [a b]))
;;   )

(def add (comp (applicative ad/add) check-args))
(def sub (comp (applicative ad/sub) check-args))
(def mul (comp (applicative ad/mul) check-args))
(def div (comp (applicative ad/div) check-args))
(def sigmoid (comp (applicative ad/sigmoid) check-args))
(def pow (comp (applicative ad/pow) check-args))
(def square #(pow % (constant 2.)))
