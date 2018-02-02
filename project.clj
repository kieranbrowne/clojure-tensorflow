(defproject clojure-tensorflow/clojure-tensorflow "0.2.4"
  :description "A very light layer over Java interop for working with TensorFlow"
  :url "http://github.com/kieranbrowne/clojure-tensorflow"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :lein-release {:deploy-via :clojars}
  :dependencies [[org.clojure/clojure  "1.9.0-alpha16"]
                 [org.tensorflow/tensorflow  "1.2.0"]
                 [kieranbrowne/autodiff "0.1.5-SNAPSHOT"]
                 [environ "1.1.0"]]
  :plugins [[lein-cloverage "1.0.9"]])
