package com.sparkProject



import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.functions._

object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP
    // on vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation de la SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc et donc aux mécanismes de distribution des calculs.)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/


    /* notebook zeppelin */

    //1

    println("1.a ===========================================")
    val path = "/Users/fabienfluro/Documents/MS_BGD/INF729_Spark/TP/train_clean.csv"

    val df = spark.read.format("csv").option("header", "true").load(path)

    //2

    println("1.b ===========================================")
    println("Number rows:" + df.count() + ", Number columns:" + df.columns.size)

    //3
    println("1.c ===========================================")
    df.show(5)

    //4
    println("1.d ===========================================")
    df.printSchema()



    println("1.e ===========================================")


    val dfCasted = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline", $"deadline".cast(IntegerType))
      .withColumn("state_changed_at", $"state_changed_at".cast(IntegerType))
      .withColumn("created_at", $"created_at".cast(IntegerType))
      .withColumn("launched_at", $"launched_at".cast(IntegerType))
      .withColumn("backers_count", $"backers_count".cast(IntegerType))
      .withColumn("final_status", $"final_status".cast(IntegerType))

    dfCasted.printSchema()


    println("2.a ===========================================")

    dfCasted.describe("goal", "deadline", "state_changed_at", "created_at", "launched_at", "backers_count", "final_status").show()

    println("2.b ===========================================")


  /*
    //df2.filter(df2("goal") > 21).show()
    df2.createOrReplaceTempView("table")

    spark.sql("SELECT * FROM table WHERE disable_communication = 'FALSE'").count()
    //df2.groupBy(df2("disable_communication") = "False").count().show()
*/
    dfCasted.groupBy("country").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("disable_communication").count.orderBy($"count".desc).show(100)

    dfCasted.groupBy("currency").count.orderBy($"count".desc).show(100)
    dfCasted.select("deadline").dropDuplicates.show()
    dfCasted.groupBy("state_changed_at").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("backers_count").count.orderBy($"count".desc).show(100)
    dfCasted.select("goal", "final_status").show(30)
    dfCasted.groupBy("country", "currency").count.orderBy($"count".desc).show(50)


    //println("2.c ===========================================")
    //dfCasted.withColumn("disable_communication", $"disable_communication").drop("disable_communication")


    val df2 = dfCasted.drop("disable_communication")
    df2.printSchema()

    //println("2.d ===========================================")
    //df2.withColumn("disable_communication", df("disable_communication")).drop("disable_communication")

    val dfNoFutur = df2.drop("backers_count", "state_changed_at")

    //println("2.e ===========================================")

    dfNoFutur.where($"country" === "False")
      .groupBy("currency")
      .count
      .orderBy($"count".desc)
      .show(50)

    def udfCountry = udf{(country: String, currency: String) =>
      if (country == "False")
        currency
      else
        country
    }

    def udfCurrency = udf{(currency: String) =>
      if ( currency != null && currency.length != 3 )
        null
      else
        currency //: ((String, String) => String)  pour éventuellement spécifier le type
    }

    val dfCountry = dfNoFutur
      .withColumn("country2", udfCountry($"country", $"currency"))
      .withColumn("currency2", udfCurrency($"currency"))
      .drop("country", "currency")

    dfCountry
      .groupBy("country2", "currency2")
      .count.orderBy($"count".desc)
      .show(50)


    println("2.f ===========================================")

    dfCountry
      .groupBy("final_status")
      .count
      .orderBy($"count".desc)
      .show(50)

    val dfFiltered = dfCountry.filter($"final_status".isin(0, 1))


    println("3 ===========================================")

    val dfDurations = dfFiltered
      .withColumn("deadline2", from_unixtime($"deadline"))
      .withColumn("created_at2", from_unixtime($"created_at"))
      .withColumn("launched_at2", from_unixtime($"launched_at"))
      .withColumn("days_campaign", datediff($"deadline2", $"launched_at2"))
      .withColumn("hours_prepa", round(($"launched_at" - $"created_at")/3600.0, 3))
      .filter($"hours_prepa" >= 0 && $"days_campaign" >= 0)
      .drop("created_at", "deadline", "launched_at")


    val dfLower: DataFrame = dfDurations
      .withColumn("name", lower($"name"))
      .withColumn("desc", lower($"desc"))
      .withColumn("keywords", lower($"keywords"))

    val dfText= dfLower
      .withColumn("text", concat_ws(" ", $"name", $"desc", $"keywords"))


    println("4 Valeurs nulles ===========================================")

    val dfReady: DataFrame = dfText
      .filter($"goal" > 0)
      .na
      .fill(Map(
        "days_campaign" -> -1,
        "hours_prepa" -> -1,
        "goal" -> -1,
        "country2" -> "unknown",
        "currency2" -> "unknown"
      ))


    dfReady.show(50)
    println(dfReady.count)


    println("5 ===========================================")

    dfReady
      .write
      .mode(SaveMode.Overwrite)
      .parquet("/Users/fabienfluro/Documents/MS_BGD/INF729_Spark/TP/prepared_trainingset")


  }

}
