(ns rnn.core
  (:require [clojure-tensorflow.ops :as tf]
            [clojure-tensorflow.layers :as layer]
            [clojure-tensorflow.optimizers :as optimize]
            [clojure-tensorflow.core :refer [session-run]]))

;; Training data
(def data
  (clojure.string/join (repeat 5 "meow ")))

(def conversion-map
  (apply merge (map (fn [char num]
                {char (double (/ num (dec (count (set data)))))})
              (set data) (range (count (set data))))))


(defn convert-back [model-output]
  (map (partial (comp (partial apply str) map)
                (fn [x] (get (clojure.set/map-invert conversion-map)
                            (first (sort-by #(Math/abs (- x %))
                                            (vals conversion-map))))
                  )) model-output))



(def input (tf/constant
            (partition 4 (map conversion-map data))))
(def target (tf/constant
             (partition 4 (rest (map conversion-map data)))))


(defn random-normal
  "Generate a tensor of random values with a normal distribution"
  ([shape stddev]
   (let [source (java.util.Random. (rand))]
     ((reduce #(partial repeatedly %2 %1)
              #(.nextGaussian source)
              (reverse shape)))))
  ([shape] (random-normal shape 1)))



(def input-weights
  "Initialise weights as variable tensor of values between -1 and 1"
  (tf/variable (random-normal [4 18])))

(def hidden-weights
  "Initialise weights as variable tensor of values between -1 and 1"
  (tf/variable (random-normal [18 18])))

(def output-weights
  "Initialise weights as variable tensor of values between -1 and 1"
  (tf/variable (random-normal [18 4])))

;; Define network / model
(def network
  (comp
   #(tf/sigmoid (tf/matmul % output-weights))
   (apply comp (repeat 5 #(tf/sigmoid (tf/matmul % hidden-weights))))
   #(tf/sigmoid (tf/matmul % input-weights))
   ))


(def error
  ;; the squared difference is a good one
  (tf/square (tf/sub target (network input))))

;; Train Network
(def sess (clojure-tensorflow.core/session))

;; initialise global variables
(session-run sess [(tf/global-variables-initializer)])

;; use gradient decent to train the network
(session-run sess
 [(repeat 100 (optimize/gradient-descent error))
  (tf/mean (tf/mean error))])
;; the error is now very small.

(defn get-next-char [input]
  (last (last
         (convert-back
          (session-run
           sess
           [(network
             (tf/constant
              [(map conversion-map input)]))])))))

;; Generate 20 characters with the trained network
(loop [n 20 string "meow"]
  (if (zero? n)
    string
    (recur (dec n) (str string (get-next-char (apply str (take-last 4 string)))))))
;; => "meow meow meow meow meow"

