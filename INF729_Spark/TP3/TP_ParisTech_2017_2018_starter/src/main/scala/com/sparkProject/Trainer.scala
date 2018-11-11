package com.sparkProject


import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    import spark.implicits._
    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/


    /** 1 - Charger le dataframe **/

    val df: DataFrame = spark
      .read
      .parquet("/Users/fabienfluro/Documents/Repo_Git/Fabien_Fluro/INF729_Spark/TP3/prepared_trainingset")


    //df.show(50)
    //df.printSchema()

    /** 2- Utiliser les données textuelles **/
    /** a **/
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    //val dfTokenizered = tokenizer.transform(df)
    //dfTokenizered.printSchema()
    //dfTokenizered.select("text", "tokens").show(5)

    /** b **/
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("tokensClean")

    //val dfRemoved: DataFrame = remover.transform(remover)
    //dfRemoved.select("tokens", "tokensClean").show(5)

    /** c **/
    val vectorize = new CountVectorizer()
      .setInputCol("tokensClean")
      .setOutputCol("vectorize")
      //.setVocabSize(3)
      //.setMinDF(2)
      //.fit(dfRemoved)

    //val dfVectorized: DataFrame  = vectorize.transform(dfRemoved)
    //dfVectorized.select("tokensClean", "vectorized").show(5)

    /** d **/
    val tfidf = new IDF()
      .setInputCol("vectorize")
      .setOutputCol("tfidf")

    //val idfModel = idf.fit(dfVectorized)

    //val dfIdf: DataFrame = idfModel.transform(dfVectorized)
    //dfIdf.select("vectorized", "tfidf").show(10)


    /** 3 - Convertir les catégories en données numériques **/

    /** e **/
    val indexerCountry = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("skip")


    /** f **/
    val indexerCurrency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("skip")


    /** g **/
    val encoderCountry = new OneHotEncoder()
      .setInputCol("country_indexed")
      .setOutputCol("country_onehot")


    val encoderCurrency = new OneHotEncoder()
      .setInputCol("currency_indexed")
      .setOutputCol("currency_onehot")

    //val encoder = new OneHotEncoderEstimator().setInputCols(Array("country_indexed", "currency_indexed")).setOutputCols(Array("countryVect", "currencyVect"))


    /** 4. Mettre les données sous une forme utilisable par Spark.ML **/
    /** h **/
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"))
      .setOutputCol("features")


    /** i **/
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setTol(1.0e-6)
      .setMaxIter(300)

    /** j **/
    val pipeline = new Pipeline()
      .setStages(Array(
        tokenizer, remover, vectorize,
        tfidf, indexerCountry, indexerCurrency,
        encoderCountry, encoderCurrency, assembler, lr))


    /** 5 Entraînement et tuning du modèle **/

    /** k **/
    val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed = 2810)


    /** l **/

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, 10e-8 to 10e-2 by 2.0)
      .addGrid( vectorize.minDF, 55.0 to 95.0 by 20.0).build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    val model = trainValidationSplit.fit(training)


    // Sauvegarder le model sur le disque
    model.write.overwrite().save("/Users/fabienfluro/Documents/Repo_Git/Fabien_Fluro/INF729_Spark/TP3/spark-logistic-regression-model")

    // Sauvegarder le pipeline sur le disque
    trainValidationSplit.write.overwrite().save("/Users/fabienfluro/Documents/Repo_Git/Fabien_Fluro/INF729_Spark/TP3/gridsearch-model")

    // load it back in during production
    val sameModel = TrainValidationSplitModel.load("/Users/fabienfluro/Documents/Repo_Git/Fabien_Fluro/INF729_Spark/TP3/spark-logistic-regression-model")



    /** m **/
    val df_WithPredictions = sameModel.transform(test)

    println("f-score : " + evaluator.evaluate(df_WithPredictions))

    /** n **/
    df_WithPredictions.groupBy("final_status", "predictions").count().show()

  }
}
