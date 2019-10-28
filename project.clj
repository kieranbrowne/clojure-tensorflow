(defproject clojure-tensorflow/clojure-tensorflow "0.2.4"
  :description "A very light layer over Java interop for working with TensorFlow"
  :url "http://github.com/kieranbrowne/clojure-tensorflow"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :lein-release {:deploy-via :clojars}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.tensorflow/tensorflow  "1.15.0"]
                 [environ "1.1.0"]])
