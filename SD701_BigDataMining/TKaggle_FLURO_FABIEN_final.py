# Databricks notebook source
# MAGIC %md #Kaggle Competition
# MAGIC 
# MAGIC In this session lab, we are going to compete in a Kaggle Competition.
# MAGIC 
# MAGIC First, we are going to upload the `train` and `test` datasets to databricks using the following route:
# MAGIC 
# MAGIC *Data -> Add Data -> Upload File*
# MAGIC 
# MAGIC **Note:** You have the option to select the location to store the files within DBFS.

# COMMAND ----------

# MAGIC %md Once the files are uploaded, we can use them in our environment.
# MAGIC 
# MAGIC You will need to change /FileStore/tables/train.csv with the name of the files and the path(s) that you chose to store them.
# MAGIC 
# MAGIC **Note 1:** When the upload is complete, you will get a confirmation along the path and name assigned. Filenames might be slightly modified by Databricks.
# MAGIC 
# MAGIC **Note 2:** If you missed the path and filename message you can navigate the DBFS via: *Data -> Add Data -> Upload File -> DBFS* or checking the content of the path `display(dbutils.fs.ls("dbfs:/FileStore/some_path"))`

# COMMAND ----------

train_data0 = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferSchema='true').load('/FileStore/tables/train_set-51e11.csv')

test_data0 = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferSchema='true').load('/FileStore/tables/test_set-b5f57.csv')


# COMMAND ----------

# MAGIC %md 
# MAGIC ##Etude préliminaire des données brutes
# MAGIC 
# MAGIC Nous commancons notre étude par une visualisation du scatter plot des données numériques afin d'identifier une éventuelle relation "évidente" ou avoir une idée de la distribution des données.

# COMMAND ----------

display(train_data0)

# COMMAND ----------

import pandas as pd
pd.DataFrame(train_data0.take(5), columns=train_data0.columns).transpose()

# COMMAND ----------

train_data0_numerical = train_data0.select("Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points", "Cover_Type")

display(train_data0_numerical)

# COMMAND ----------

df = train_data0_numerical.describe().toPandas()


# COMMAND ----------

df.transpose().head(100)

# COMMAND ----------

numeric_features = [t[0] for t in train_data0_numerical.dtypes if t[1] == 'int' or t[1] == 'double']

print(numeric_features)

# COMMAND ----------

sampled_data = train_data0_numerical.sample(False, 0.1).toPandas()

axs = pd.scatter_matrix(sampled_data, figsize=(12, 12));

# Rotate axis labels and remove axis ticks
n = len(sampled_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())


# COMMAND ----------

print('Train data size: {} rows, {} columns'.format(train_data0.count(), len(train_data0.columns)))
print('Test data size: {} rows, {} columns'.format(test_data0.count(), len(test_data0.columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion sur le feature engineering
# MAGIC 
# MAGIC Le scatter plot nous a amené à sélectionner plusieurs combinaisons des features. Malheureusement aucune sélection testé n'a permit une amélioration du score. La tentative manuelle ayant été infructueuse nous avons tenter d'utiliser des méthodes approuvées.
# MAGIC 
# MAGIC Plusieurs tentatives de sélection de données ont été effectuées en utilisant une PCA. En sélectionnant 3, 10, 20 et 30 variables n'ont, hélas, pas permis une amélioration du score (score observé entre 0.4 et 0.6)
# MAGIC 
# MAGIC De même l'ajout de nouvelles features (calculer à partir de features déjà présentes) n'a, de même, pas apporté d'amélioration sur les prédictions de nos classifiers. Comme exemples testés, nous pouvons citer:
# MAGIC - La distance directe à une source d'eau (en se basant sur les distances verticales et horizontales)
# MAGIC - L'élévation au carré (Elevation**2)

# COMMAND ----------

# MAGIC %md
# MAGIC We will use the `VectorAssembler()` to merge our feature columns into a single vector column as requiered by Spark methods.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

vector_assembler = VectorAssembler(inputCols=["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points", "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9", "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14", "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24", "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29", "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34", "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40"], outputCol="features")


# COMMAND ----------

# MAGIC %md
# MAGIC Nous séparons notre set de données afin de pouvoir calculer un score "intermédiaire" sans avoir besoin d'uploader notre résultat sur Kaggle pour chaque test.

# COMMAND ----------

(train_data, test_data) = train_data0.randomSplit([0.7, 0.3], seed = 100)

# COMMAND ----------

# MAGIC %md
# MAGIC En teste plusieurs classifiers:
# MAGIC - Regression logistique
# MAGIC - Random forest
# MAGIC - Arbre de décision
# MAGIC 
# MAGIC En entrainant nos types/classes de classifiers avec plusieurs paramères (maxDepth, maxBins, impurity, ...) on obtient le modèle optimale qui maximise notre score ainsi que les paramètres associés.
# MAGIC 
# MAGIC Ci-dessous: l'implémentation de la logique de sélection du meilleur modèle.

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.feature import PCA
 
#pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")  
#rf = DecisionTreeClassifier(labelCol="Cover_Type", featuresCol="pcaFeatures", maxDepth=30)

rf = DecisionTreeClassifier(labelCol="Cover_Type", featuresCol="features", maxDepth=30, maxBins=45, impurity="entropy")

#pipeline = Pipeline(stages=[vector_assembler, pca, rf])

pipeline = Pipeline(stages=[vector_assembler, rf])

evaluator = MulticlassClassificationEvaluator(labelCol="Cover_Type", predictionCol="prediction", metricName="accuracy")

'''
paramGrid = (ParamGridBuilder()
            .addGrid(rf.checkpointInterval, [2, 5, 10])
            .addGrid(rf.maxBins, [10, 32, 64])
            #.addGrid(pca.k, [3, 10, 20, 30])
            .build())

#####
trainValSplit = TrainValidationSplit(estimator=pipeline, evaluator=evaluator)

trainValSplit.setEstimatorParamMaps(paramGrid)
#####


#####

#crossval = CrossValidator(estimator=pipeline, evaluator=evaluator, numFolds=3)

#crossval.setEstimatorParamMaps(paramGrid)
######


model = trainValSplit.fit(train_data).bestModel


#model = crossval.fit(train_data).bestModel

######
'''


model = pipeline.fit(train_data)

predictions = model.transform(test_data)



# COMMAND ----------

# MAGIC %md
# MAGIC Récupération du meilleur classifier ainsi que les paramètres associés :

# COMMAND ----------


bestclassifier = model.stages[3]._java_obj
bestParams = bestclassifier.getMaxBins()

# COMMAND ----------

bestPipeline = cvModel.bestModel
bestLRModel = bestPipeline.stages[2]
bestParams = bestLRModel.extractParamMap()

# COMMAND ----------

predictions = predictions.withColumn("features", predictions["prediction"].cast("int"))

#print(predictions)

accuracy = evaluator.evaluate(predictions)


print("Accuracy = %g " % accuracy)

# COMMAND ----------

predictions0 = model.transform(test_data0)

# COMMAND ----------

predictions0 = model.transform(test_data0)
predictions0 = predictions0.withColumn("Cover_Type", predictions0["prediction"].cast("int"))

predictions0.printSchema()


# COMMAND ----------

# Select columns Id and prediction
(predictions0
 .repartition(1)
 .select('Id', 'Cover_Type')
 .write
 .format('com.databricks.spark.csv')
 .options(header='true')
 .mode('overwrite')
 .save('/FileStore/kaggle-submission'))

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/FileStore/kaggle-submission"))

# COMMAND ----------

# MAGIC %md To be able to download the predictions file, we need its name (`part-*.csv`):

# COMMAND ----------

# MAGIC %md Files stored in /FileStore are accessible in your web browser via `https://<databricks-instance-name>.cloud.databricks.com/files/`.
# MAGIC   
# MAGIC For this example:
# MAGIC 
# MAGIC https://community.cloud.databricks.com/files/kaggle-submission/part-*.csv?o=######
# MAGIC 
# MAGIC where `part-*.csv` should be replaced by the name displayed in your system  and the number after `o=` is the same as in your Community Edition URL.
# MAGIC 
# MAGIC 
# MAGIC Finally, we can upload the predictions to kaggle and check what is the perfromance.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC 
# MAGIC Notre classifier optimal est un arbre de décision de profondeur maximal égale à 30, le nombre maximal de bins égale à 45 et une impurité entropique. Notre classifier catégorise les types de forêt avec une précision de 86%. 
# MAGIC 
# MAGIC La phase déterminante de ce sujet est de sélectionner les variables déterminantes (par transformation ou simple sélection de variables existantes). Il est donc dommage que cette phase eu été un échec.
# MAGIC 
# MAGIC Il est à noter que des difficultés ont été rencontrés lors de l'utilisation de Databricks (crash du cluster, durée de calcul trop lonque,...) et ont limités le nombre de test lancés. Pour le prochain projet de ce type, je m'orienterai plus vers une solution de calcul en local car plus stable et plus rapide (retour des autres étudiants)
