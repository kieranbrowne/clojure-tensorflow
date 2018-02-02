(ns clojure-tensorflow.core
  (:require [clojure-tensorflow.utils :as utils]
            [clojure-tensorflow.ops :as ops]
            [clojure-tensorflow.graph
             :refer [graph global-variables shadow-graph shadow-graph']]
            [clojure-tensorflow.session :refer [session]]
            [clojure.spec.alpha :as s]
            [clojure-tensorflow.gradients :as grad]
            [autodiff.protocols :as ad]
            [clojure-tensorflow.build :as build]
            [clojure-tensorflow.ops :as tf])
  (:import [autodiff.protocols.Dual]))



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
       (do
         (doall (map build/build-op (flatten op)))
         ;; (-> session .runner (feed feed-map))
           (last (map #(run % feed-map) (flatten op)))))
     ;; if run on a single op return it
     (-> session
         .runner
         (feed feed-map)
         (.fetch (get-name (build/build-op op)))
         .run
         (.get 0)
         utils/tensor->clj
         )
     )))



(defmacro with-graph [& body]
  `(binding [graph (org.tensorflow.Graph.)
             global-variables (atom [])
             shadow-graph (atom [])
             shadow-graph' (atom {})]
     (try ~@body
       (finally (.close graph)))))

(defmacro with-this-graph [g & body]
  `(binding [graph (org.tensorflow.Graph.)
             global-variables (atom [])
             shadow-graph' (atom ~g)]
     (try ~@body
          (finally (.close graph)))))

(defmacro with-session [& body]
  `(binding [session (org.tensorflow.Session. graph)]
     (try ~@body
       (finally (.close session)))))

(with-this-graph
     {:x {:operation :Const
          :attrs {:dtype (.dataType (utils/clj->tensor 3))
                  :value (utils/clj->tensor 3)
                  }}
      :y {:operation :Const
          :attrs {:dtype (.dataType (utils/clj->tensor 2))
                  :value (utils/clj->tensor 2)
                  }}
      :z {:operation :Mul
          :inputs [:x :wrty]}
      }
  (ad/wrt :y)

  ;; (build/build-op :z)
  ;; (with-session (run :dzdx {:x 2.}))
  ;; (build/build-op :y)
  ;; (run (ad/wrt :y))

  ;; (let [y' (ad/wrt :y)]
  ;;   ;; (build/build-op y')
  ;;   (run :z))

  ;; (get-name
  ;;  (build/build-op :y))
  (with-session

    (let [wrty (ad/wrt :y)
          z (tf/mul :x wrty)]
      (run (:f' z)))
    )

  ;; (with-session (run (ad/wrt :z) {:x 2.}))
  ;; (ad/wrt :z)
  ;; (with-session
  ;;   (build/build-op :x)
  ;;   (build/build-op :y)
  ;;   (build/build-op :z)
  ;;   (run :x))
  )

(with-this-graph
  {:x {}
   :y {}
   :z {:inputs [:x :y]}
   :a {:inputs [:z :x]}
   }

  (reduce clojure.set/union
          (grad/parents :a)
          (map grad/parents (grad/parents :a)))
  ;; (map grad/parents (grad/parents :a))
  ;; (grad/antecedents :a)
  ;; (with-session (run (ad/wrt :z) {:x 2.}))
  ;; (ad/wrt :z)
  (with-session
    (build/build-op :x)
    (build/build-op :y)
    (build/build-op :z)
    (run :x))
  )

(defn deriv
  [graph op wrt]
  ()
  )


;; (build/build-op {:operation :Placeholder
;;                  :attrs {:dtype (utils/clj->tensor 3)}})




;; (def x (ops/constant 1))
;; (def y (ops/constant 2))
;; (def z (ops/add x y))

;; (defn children
;;   [op-name]
;;   (->> @shadow-graph'
;;        (map (comp :inputs val))
;;        (filter (complement nil?))
;;        flatten set))

;; (children z)
;; (parents z)
;; (children y)
;; (parents x)



;; (ops/leibniz-notation y :r)

;; (def x (ops/constant 1))
;; (x @shadow-graph')

(defn derivative [y x]
  (cond
    (= (class y) autodiff.protocols.Dual) (throw (ex-info "not sure why this fails" {}))
    (= x y) (ad/coerce x 1) ;; likely culprit
    (= (:operation (y @shadow-graph')) "Const") (do (println y) y)
    :default (ops/rebuild-op(ops/add-shadow-op
                             (update (y @shadow-graph')
                                     :inputs
                                     (partial map #(derivative % x)))
                             (ops/leibniz-notation y x)
                             ))))

;; (def a
;;   (ops/add-shadow-op
;;    (update-in
;;     (z @shadow-graph')
;;     [:inputs 0]
;;     ad/coerce
;;     )))


;; (s/explain :clojure-tensorflow.ops/op-def
;;            (update-in
;;             (z @shadow-graph')
;;             [:inputs 0]
;;             ad/coerce
;;             ))
;; (run a)

;; (run
;;   (derivative z y))

;; (run
;;   (derivative z z))




;; (with-graph
;;   (with-session

;;     (let [x (ops/constant 2.)
;;           ;; y (ops/constant 2.)
;;           ;; z (ops/add x y)
;;           y (ops/sigmoid x)
;;           z (ops/sigmoid y)
;;           d (derivative z x)

;;           ;; a (swap! shadow-graph' :blerp (ad/coerce x))
;;           ]

;;       ;; (run (ops/rebuild-op d))

;;       ;; (run :blerp)
;;       (run (derivative z x))

;;       ;; (run
;;       ;;   (ops/rebuild-op x))
;;       ;; (let [{:keys [ad-fn inputs]} (z @shadow-graph')]
;;       ;;   (apply
;;       ;;    (find-protocol-method ad/AutoDiff ad-fn 0)
;;       ;;    inputs)
;;       ;;   )
;;       ;; (let [{:keys [ad-fn inputs]} (z @shadow-graph')]
;;       ;;   ad-fn
;;       ;;   (apply (find-protocol-method ad/AutoDiff ad-fn 0)
;;       ;;          inputs)
;;       ;;   )

;;       ;; (derivative z x)
;;       ;; (binding [shadow-graph'
;;       ;;           (atom (make-prime @shadow-graph' x))
;;       ;;           ]
;;       ;;   ;; @shadow-graph'
;;       ;;   (run (derivative z x)))

;;       ;; (d @shadow-graph')
;;       ;; (run d)
;;       ;; ((:f (first (parents d))) @shadow-graph')
;;       ;; (run
;;       ;;   (derivative z x))
;;       ;; (run
;;       ;;   (build/build-op z))
;;       ;; (run d)
;;       ;; (z @shadow-graph')
;;       ;; (run)
;;       ;; (run
;;       ;;   (derivative z x))
;;       ;; (deref shadow-graph')
;;       ;; (run
;;       ;;   (derivative z y))
;;       ;; (run (ad/mul (ad/coerce x) (ad/add (ad/coerce z 1) y)))
;;       )))


;; (ops/constant 3.1)
;; (ops/constant 1)
;; ;; (run
;; (build/build-op (ops/constant 1))
;; (with-graph
;;   (with-session
;;     (let [a (ops/constant 1)
;;           b (ops/constant 1)
;;           c (ops/add a b)
;;           ]
;;       ;; (run (ops/add (ops/constant 1)
;;       ;;               (ops/constant 1)
;;       ;;               ))
;;       (run
;;         (ad/coerce c))
;;       (run
;;         (build/build-op
;;          c))
;;       ;; (deref shadow-graph')
;;       )
;;     ))
