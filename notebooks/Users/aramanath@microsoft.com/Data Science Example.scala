// Databricks notebook source
// MAGIC %md
// MAGIC ###Machine Learning
// MAGIC 
// MAGIC The following examples involve using MLlib algorithms

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ## Financial Fraud Detection using LR ML model
// MAGIC 
// MAGIC In this notebook, we will showcase the use of logistic regression & Random Forest ML models to predict fraudulent financial transactions. The input data for this downloaded from Kaggle - https://www.kaggle.com/ntnu-testimon/paysim1.

// COMMAND ----------

// DBTITLE 1,Read the input data and create temp table
// MAGIC %python
// MAGIC 
// MAGIC # File location and type
// MAGIC file_location = "/FileStore/tables/PS_20174392719_1491204439457_log.parquet"
// MAGIC file_type = "parquet"
// MAGIC 
// MAGIC # CSV options
// MAGIC infer_schema = "true"
// MAGIC first_row_is_header = "true"
// MAGIC delimiter = ","
// MAGIC 
// MAGIC # The applied options are for CSV files. For other file types, these will be ignored.
// MAGIC df = spark.read.format(file_type) \
// MAGIC   .option("inferSchema", infer_schema) \
// MAGIC   .option("header", first_row_is_header) \
// MAGIC   .option("sep", delimiter) \
// MAGIC   .load(file_location)
// MAGIC 
// MAGIC df.createOrReplaceTempView("financial_logs")
// MAGIC display(df)

// COMMAND ----------

// DBTITLE 1,Schema of the data frame
// MAGIC %python
// MAGIC 
// MAGIC # Print the schema
// MAGIC df.printSchema()

// COMMAND ----------

// DBTITLE 1,Describe the data frame
// MAGIC %python
// MAGIC 
// MAGIC display(df.describe())

// COMMAND ----------

// DBTITLE 1,Number of transactions of each type
// MAGIC %sql
// MAGIC -- Organize by Type
// MAGIC select type, count(1) from financial_logs group by type

// COMMAND ----------

// DBTITLE 1,Amounts in each type of transaction
// MAGIC %sql
// MAGIC select type, sum(amount) from financial_logs group by type

// COMMAND ----------

// DBTITLE 1,No. of Fraud Transactions
// MAGIC %sql
// MAGIC 
// MAGIC SELECT count(*) from financial_logs where isFraud = 1

// COMMAND ----------

// DBTITLE 1,No. of non-Fraud Transactions
// MAGIC %sql
// MAGIC 
// MAGIC SELECT count(*) from financial_logs where isFraud = 0

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ###This dataset requires stratified sampling to balance the dataset

// COMMAND ----------

// DBTITLE 1,Run Stratified Sampling
// MAGIC %python
// MAGIC 
// MAGIC stratified_df = df.sampleBy('isFraud', fractions={0: 8213/6354407, 1: 1.0})
// MAGIC stratified_df.groupby('isFraud').count().show()

// COMMAND ----------

// DBTITLE 1,Build a Correlation Matrix now
// MAGIC %python
// MAGIC 
// MAGIC from pyspark.mllib.stat import Statistics
// MAGIC import pandas as pd
// MAGIC 
// MAGIC num_df = stratified_df.select('step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFraud', 'isFlaggedFraud')
// MAGIC col_names = num_df.columns
// MAGIC features = num_df.rdd.map(lambda row: row[0:])
// MAGIC corr_mat=Statistics.corr(features, method="pearson")
// MAGIC corr_df = pd.DataFrame(corr_mat)
// MAGIC corr_df.index, corr_df.columns = col_names, col_names

// COMMAND ----------

// DBTITLE 1,Plot the Correlation Matrix
// MAGIC %python
// MAGIC 
// MAGIC import seaborn as sns
// MAGIC import matplotlib.pyplot as plt
// MAGIC plt.figure(figsize=(10,8))
// MAGIC display(sns.heatmap(corr_df, annot = True, fmt = '.3f'))

// COMMAND ----------

// DBTITLE 1,Dropping the highly correlated columns
// MAGIC %python
// MAGIC 
// MAGIC cols_to_drop = ['oldbalanceDest','oldbalanceOrg']
// MAGIC df = stratified_df.drop(*cols_to_drop)
// MAGIC display(df)

// COMMAND ----------

// DBTITLE 1,Prepare the datasets
// MAGIC %python
// MAGIC 
// MAGIC from pyspark.ml import Pipeline
// MAGIC from pyspark.ml.feature import StringIndexer
// MAGIC #from pyspark.ml.feature import OneHotEncoderEstimator
// MAGIC from pyspark.ml.feature import VectorAssembler
// MAGIC 
// MAGIC # Remove unwanted cols
// MAGIC non_number_cols = ['nameOrig','nameDest']
// MAGIC df = df.drop(*non_number_cols)
// MAGIC 
// MAGIC # Pipeline Stages
// MAGIC stages = []
// MAGIC 
// MAGIC # Encodes a string column of labels to a column of label indices
// MAGIC indexer = StringIndexer(inputCol = "type", outputCol = "typeIndexed")
// MAGIC stages += [indexer]
// MAGIC 
// MAGIC # VectorAssembler is a transformer that combines a given list of columns into a single vector column
// MAGIC assembler = VectorAssembler(inputCols = ['typeIndexed', 'step', 'amount', 'newbalanceOrig', 'newbalanceDest'], outputCol = "features")
// MAGIC stages += [assembler]
// MAGIC 
// MAGIC # Create a Pipeline.
// MAGIC pipeline = Pipeline(stages=stages)
// MAGIC 
// MAGIC # Run the feature transformations.
// MAGIC #  - fit() computes feature statistics as needed.
// MAGIC #  - transform() actually transforms the features.
// MAGIC pipelineModel = pipeline.fit(df)
// MAGIC df = pipelineModel.transform(df)
// MAGIC 
// MAGIC # Remove type & isFlaggedFraud cols
// MAGIC df = df.drop('type')
// MAGIC df = df.drop('isFlaggedFraud')
// MAGIC 
// MAGIC display(df)

// COMMAND ----------

// DBTITLE 1,Split training:test::80:20
// MAGIC %python
// MAGIC 
// MAGIC # Random split to training and test
// MAGIC trainingData, testData = df.randomSplit([0.8, 0.2], seed = 12345)
// MAGIC print(trainingData.count())
// MAGIC print(testData.count())

// COMMAND ----------

// DBTITLE 1,Verify Training Data
// MAGIC %python
// MAGIC display(trainingData)

// COMMAND ----------

// DBTITLE 1,Use the Logistic Regression Model
// MAGIC %python
// MAGIC 
// MAGIC from pyspark.ml.classification import LogisticRegression
// MAGIC 
// MAGIC # Create initial LogisticRegression model
// MAGIC lr = LogisticRegression(labelCol="isFraud", featuresCol="features", maxIter=10)
// MAGIC 
// MAGIC # Train model with Training Data
// MAGIC lrModel = lr.fit(trainingData)
// MAGIC 
// MAGIC # Make predictions on test data using the transform() method.
// MAGIC # LogisticRegression.transform() will only use the 'features' column.
// MAGIC predictions = lrModel.transform(testData)
// MAGIC display(predictions)

// COMMAND ----------

// DBTITLE 1,Model Evaluation
// MAGIC %python
// MAGIC from pyspark.ml.evaluation import BinaryClassificationEvaluator
// MAGIC 
// MAGIC # Evaluate model
// MAGIC evaluator = BinaryClassificationEvaluator(labelCol='isFraud',rawPredictionCol="rawPrediction")
// MAGIC evaluator.evaluate(predictions)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ###Can we try a different ML model?

// COMMAND ----------

// DBTITLE 1,Now trying with RandomForest Classifier ML Model
// MAGIC %python
// MAGIC 
// MAGIC from pyspark.ml.classification import RandomForestClassifier
// MAGIC from pyspark.mllib.evaluation import MulticlassMetrics
// MAGIC 
// MAGIC #generate model on splited dataset
// MAGIC rf = RandomForestClassifier(labelCol='isFraud', featuresCol='features')
// MAGIC fit = rf.fit(trainingData)
// MAGIC transformed = fit.transform(testData)

// COMMAND ----------

// MAGIC %python
// MAGIC from pyspark.ml.evaluation import BinaryClassificationEvaluator
// MAGIC 
// MAGIC # Evaluate model
// MAGIC evaluator = BinaryClassificationEvaluator(labelCol='isFraud',rawPredictionCol="rawPrediction")
// MAGIC evaluator.evaluate(transformed)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ### Trying with ParamGridBuilder & CrossValidation
// MAGIC 
// MAGIC More info here - https://docs.databricks.com/applications/machine-learning/mllib/binary-classification-mllib-pipelines.html & https://spark.apache.org/docs/latest/ml-tuning.html

// COMMAND ----------

// DBTITLE 1,Init the ParamGridBuilder
// MAGIC %python
// MAGIC 
// MAGIC from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
// MAGIC 
// MAGIC paramGrid = (ParamGridBuilder()
// MAGIC              .addGrid(rf.maxDepth, [2, 4, 6])
// MAGIC              .addGrid(rf.maxBins, [20, 60])
// MAGIC              .addGrid(rf.numTrees, [5, 20])
// MAGIC              .build())

// COMMAND ----------

// DBTITLE 1,Init CrossValidator & train
// MAGIC %python
// MAGIC 
// MAGIC # Create 3-fold CrossValidator
// MAGIC cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
// MAGIC cvModel = cv.fit(trainingData)

// COMMAND ----------

// DBTITLE 1,Test Model and Evaluate
// MAGIC %python
// MAGIC # Use test set here so we can measure the accuracy of our model on new data
// MAGIC predictions = cvModel.transform(testData)
// MAGIC # Evaluate best model
// MAGIC evaluator.evaluate(predictions)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ### THE END