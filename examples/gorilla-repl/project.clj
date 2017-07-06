(defproject gorilla-test  "0.1.0-SNAPSHOT"
  :description  "A test project for Gorilla REPL."
  :dependencies  [[org.clojure/clojure  "1.8.0"]
                  [clojure-tensorflow "0.2.4"]]
  :main ^:skip-aot gorilla-test.core
  :target-path  "target/%s"
  :plugins  [[lein-gorilla  "0.4.0"]]
  :profiles  {:uberjar  {:aot :all}})
