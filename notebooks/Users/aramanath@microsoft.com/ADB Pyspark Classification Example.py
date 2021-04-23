# Databricks notebook source
# MAGIC %md
# MAGIC ###Machine Learning
# MAGIC 
# MAGIC The following examples involve using MLlib algorithms

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Financial Fraud Detection using LR ML model
# MAGIC 
# MAGIC In this notebook, we will showcase the use of logistic regression & Random Forest ML models to predict fraudulent financial transactions. The input data for this downloaded from Kaggle - https://www.kaggle.com/ntnu-testimon/paysim1.

# COMMAND ----------

# DBTITLE 1,Read the input data and create temp table
# File location and type
file_location = "/FileStore/tables/PS_20174392719_1491204439457_log.parquet"
file_type = "parquet"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

df.createOrReplaceTempView("financial_logs")
display(df)


# COMMAND ----------

# DBTITLE 1,Schema of the data frame
# Print the schema
df.printSchema()

# COMMAND ----------

# DBTITLE 1,Describe the data frame
display(df.describe())

# COMMAND ----------

# DBTITLE 1,Number of transactions of each type
# MAGIC %sql
# MAGIC -- Organize by Type
# MAGIC select type, count(1) from financial_logs group by type

# COMMAND ----------

# DBTITLE 1,Amounts in each type of transaction
# MAGIC %sql
# MAGIC select type, sum(amount) from financial_logs group by type

# COMMAND ----------

# DBTITLE 1,No. of Fraud Transactions
# MAGIC %sql
# MAGIC 
# MAGIC SELECT count(*) from financial_logs where isFraud = 1

# COMMAND ----------

# DBTITLE 1,No. of non-Fraud Transactions
# MAGIC %sql
# MAGIC 
# MAGIC SELECT count(*) from financial_logs where isFraud = 0

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###This dataset requires stratified sampling to balance the dataset

# COMMAND ----------

# DBTITLE 1,Run Stratified Sampling
stratified_df = df.sampleBy('isFraud', fractions={0: 8213/6354407, 1: 1.0})
stratified_df.groupby('isFraud').count().show()

# COMMAND ----------

# DBTITLE 1,Build a Correlation Matrix now
from pyspark.mllib.stat import Statistics
import pandas as pd

num_df = stratified_df.select('step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFraud', 'isFlaggedFraud')
col_names = num_df.columns
features = num_df.rdd.map(lambda row: row[0:])
corr_mat=Statistics.corr(features, method="pearson")
corr_df = pd.DataFrame(corr_mat)
corr_df.index, corr_df.columns = col_names, col_names

# COMMAND ----------

# DBTITLE 1,Plot the Correlation Matrix
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
display(sns.heatmap(corr_df, annot = True, fmt = '.3f'))

# COMMAND ----------

# DBTITLE 1,Dropping the highly correlated columns
cols_to_drop = ['oldbalanceDest','oldbalanceOrg']
df = stratified_df.drop(*cols_to_drop)
display(df)

# COMMAND ----------

# DBTITLE 1,Prepare the datasets
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
#from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml.feature import VectorAssembler

# Remove unwanted cols
non_number_cols = ['nameOrig','nameDest']
df = df.drop(*non_number_cols)

# Pipeline Stages
stages = []

# Encodes a string column of labels to a column of label indices
indexer = StringIndexer(inputCol = "type", outputCol = "typeIndexed")
stages += [indexer]

# VectorAssembler is a transformer that combines a given list of columns into a single vector column
assembler = VectorAssembler(inputCols = ['typeIndexed', 'step', 'amount', 'newbalanceOrig', 'newbalanceDest'], outputCol = "features")
stages += [assembler]

# Create a Pipeline.
pipeline = Pipeline(stages=stages)

# Run the feature transformations.
#  - fit() computes feature statistics as needed.
#  - transform() actually transforms the features.
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)

# Remove type & isFlaggedFraud cols
df = df.drop('type')
df = df.drop('isFlaggedFraud')

display(df)

# COMMAND ----------

# DBTITLE 1,Split training:test::80:20
# Random split to training and test
trainingData, testData = df.randomSplit([0.8, 0.2], seed = 12345)
print(trainingData.count())
print(testData.count())

# COMMAND ----------

# DBTITLE 1,Verify Training Data
display(trainingData)

# COMMAND ----------

# DBTITLE 1,Use the Logistic Regression Model
from pyspark.ml.classification import LogisticRegression

# Create initial LogisticRegression model
lr = LogisticRegression(labelCol="isFraud", featuresCol="features", maxIter=10)

# Train model with Training Data
lrModel = lr.fit(trainingData)

# Make predictions on test data using the transform() method.
# LogisticRegression.transform() will only use the 'features' column.
predictions = lrModel.transform(testData)
display(predictions)

# COMMAND ----------

# DBTITLE 1,Model Evaluation
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Evaluate model
evaluator = BinaryClassificationEvaluator(labelCol='isFraud',rawPredictionCol="rawPrediction")
evaluator.evaluate(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Can we try a different ML model?

# COMMAND ----------

# DBTITLE 1,Now trying with RandomForest Classifier ML Model
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.evaluation import MulticlassMetrics

#generate model on splited dataset
rf = RandomForestClassifier(labelCol='isFraud', featuresCol='features')
fit = rf.fit(trainingData)
transformed = fit.transform(testData)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Evaluate model
evaluator = BinaryClassificationEvaluator(labelCol='isFraud',rawPredictionCol="rawPrediction")
evaluator.evaluate(transformed)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Trying with ParamGridBuilder & CrossValidation
# MAGIC 
# MAGIC More info here - https://docs.databricks.com/applications/machine-learning/mllib/binary-classification-mllib-pipelines.html & https://spark.apache.org/docs/latest/ml-tuning.html

# COMMAND ----------

# DBTITLE 1,Init the ParamGridBuilder
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [2, 4, 6])
             .addGrid(rf.maxBins, [20, 60])
             .addGrid(rf.numTrees, [5, 20])
             .build())

# COMMAND ----------

# DBTITLE 1,Init CrossValidator & train
# Create 3-fold CrossValidator
cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
cvModel = cv.fit(trainingData)

# COMMAND ----------

# DBTITLE 1,Test Model and Evaluate
# Use test set here so we can measure the accuracy of our model on new data
predictions = cvModel.transform(testData)
# Evaluate best model
evaluator.evaluate(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### THE END

# COMMAND ----------

