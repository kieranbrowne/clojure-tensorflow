(ns clojure-tensorflow.core)

(def default-graph (new org.tensorflow.Graph))

(def global-variables (atom []))

(defn global-variables-initializer []
  @global-variables)

(defn session
  "Create a session"
  ([graph] (new org.tensorflow.Session graph))
  ([] (session default-graph)))

(defn op-builder
  "Returns a function which creates an operation for the graph"
  ([op-profile] (op-builder op-profile default-graph))
  ([op-profile graph]
   (let [{:keys [operation node-name attrs inputs]
		  :or {node-name (str (gensym operation)) attrs {} inputs []}
		  } op-profile]
	 (utils/thread graph
				   (flatten
					 [#(.opBuilder % operation node-name)
					  ;; set attributes if any
					  (map
						(fn [attr]
						  #(.setAttr % (name (first attr)) (second attr)))
						attrs)
					  ;; add inputs if any
					  (map (fn [input]
							 #(.addInput %
										 (if (fn? input) (input graph) input)
										 )) inputs)
					  #(.build %)
					  #(.output % 0)]))) graph))



;; ops

(defn constant [val]
  (let [tensor (clj->tensor val)]
    (op-builder
     {:operation "Const"
      :attrs {:dtype (.dataType tensor)
              :value tensor
              }})))

(defn assign [var val]
  (op-builder
   {:operation "Assign"
    :inputs [var (if (utils/tf-obj? val) val (constant val))]
    }))

(defn variable
  ([val] (variable val {}))
  ([val bits]
   (let [tensor (clj->tensor val)
         var (op-builder
          (merge
           {:operation "Variable"
            :attrs {:shape (utils/tensor->shape tensor)
                    :dtype (.dataType tensor)}
            } bits))]
     (swap! global-variables conj (assign var val))
     var)))

(defn placeholder [datatype]
  (op-builder
   {:operation "Placeholder"
    :attrs {:dtype datatype}
    }))

(defn mult [a b]
  (op-builder
   {:operation "Mul"
    :inputs [a b]}))

(defn div [a b]
  (op-builder
   {:operation "Div"
    :inputs [a b]}))

(defn add [a b]
  (op-builder
   {:operation "Add"
    :inputs [a b]}))

(defn sub [a b]
  (op-builder
   {:operation "Sub"
    :inputs [a b]}))

(defn sum
  ([t] (sum t (constant 0)))
  ([t dims]
   (op-builder
    {:operation "Sum"
     :inputs [t dims]})))

(defn tanh [a]
  (op-builder
   {:operation "Tanh"
    :inputs [a]}))

(defn sigmoid [a]
  (op-builder
   {:operation "Sigmoid"
    :inputs [a]}))

(defn pow [a b]
  (op-builder
   {:operation "Pow"
    :inputs [a b]}))

(defn size [a]
  (op-builder
   {:operation "Size"
    :inputs [a]}))

(defn abs [a]
  (op-builder
   {:operation "Abs"
    :inputs [a]}))

(defn mean [a]
  (op-builder
   {:operation "Mean"
    :inputs [a (constant 0)]}))

(defn transpose [a]
  (op-builder
   {:operation "Transpose"
    :inputs [a (constant [1 0])]}))

(defn matmul [a b]
  (op-builder
   {:operation "MatMul"
    :inputs [a b]}))


(defn op-run
  "Call session runner on single op.
  Returns tensor object"
  ([op] (op-run tf/default-graph op))
  ([graph op] (op-run graph (Session. graph) op {}))
  ([graph session op] (op-run graph session op {}))
  ([graph session op feed-map]
  (-> session
      .runner
      (feed feed-map)
      (.fetch (.name (.op (if (fn? op) (op graph) op))))
      .run
      (.get 0)
      )))

(defn session-run
  "Run list of ops, return last"
  ([ops] (session-run tf/default-graph ops))
  ([graph ops] (session-run graph (Session. graph) ops))
  ([graph session ops]
   (let [ops (flatten ops)
         op-run (partial op-run graph session)]

     ;; initialise global variables
     (map op-run @global-variables)

     ;; run first n ops to set up state
     (doseq [op (butlast ops)]
       (op-run op))

     ;; run final op and return value
     (tensor->clj
      (op-run (last ops))))))

