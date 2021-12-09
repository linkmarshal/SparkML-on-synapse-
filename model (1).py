#!/usr/bin/env python
# coding: utf-8

# In[129]:


# initialize sql context
from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)


# In[130]:


#Todo: using SQLContext to read csv and assign to dataframe
df = sqlContext.read.csv('abfss://pocfsn@pocdls.dfs.core.windows.net/dbo.pocdata.csv', header=True, inferSchema= True)	


# In[131]:


#Todo:printSchema
df =df.drop('CUSTID', 'Latest_TxnDate', 'State', 'LGA', 'Account_df.printSchema()Open_Date')
df.printSchema()


# In[132]:


df


# In[133]:


# Import all from `sql.types`
from pyspark.sql.types import *

# Write a custom function to convert the data type of DataFrame columns
def convertColumn(df, names, newType):
    for name in names: 
        df = df.withColumn(name, df[name].cast(newType))
    return df 
# List of continuous features
CONTI_FEATURES  = ['No_of_Accounts', 
'GDP_in_Billions_of_USSD',
'Inflation',
'Population',
'txn_amount_M1',
 'txn_vol_M1',
 'txn_amount_M2',
 'txn_vol_M2',
 'txn_amount_M3',
 'txn_vol_M3',
 'F1']
# Convert the type
df = convertColumn(df, CONTI_FEATURES, FloatType())
# Check the dataset
df.printSchema()


# In[134]:


df.show(5, truncate = False)


# In[135]:


#import libraries for pipeline
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler


# In[170]:


# 1. Encode the categorical data
CATE_FEATURES = ['Gender',
'Age_Band',
'Tenure',
'Education', 
'Marital_Status', 
'Segment', 
'Occupation', 
'F2', 
'Recency', 
'Freq_M1',
'Freq_M2',
'F3']
# stages in our Pipeline
stages = [] 


for categoricalCol in CATE_FEATURES:
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()],
                                     outputCols=[categoricalCol + "classVec"])
    
    stages += [stringIndexer, encoder]


# In[165]:


# 2. Index the label feature
# Convert label into label indices using the StringIndexer
#CHURN_stringIdx =  StringIndexer(inputCol="CHURN", outputCol="newCHURN")
#stages += [CHURN_stringIdx]


# In[171]:


# 3. Add continuous variable
assemblerInputs = [c + "classVec" for c in CATE_FEATURES] + CONTI_FEATURES


# In[174]:


# 4. Assemble the steps
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]


# In[173]:


stages


# In[169]:


# Create a Pipeline.
pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(df)
model = pipelineModel.transform(df)


# In[142]:


model


# In[143]:


# To make the computation faster, you convert model to a DataFrame.
#You need to select newlabel and features from model using map.

from pyspark.ml.linalg import DenseVector
input_data = model.rdd.map(lambda x: (x["newCHURN"], DenseVector(x["features"])))


# In[144]:


# import 
# from pyspark.ml.linalg import DenseVector
df_train = sqlContext.createDataFrame(input_data, ["CHURN", "features"])


# In[145]:


df


# In[146]:


df_train.show(2)


# In[147]:


# Split the data into train and test sets
train_data, test_data = df_train.randomSplit([.8,.2],seed=1234)


# In[148]:


# Let's count how many people with income below/above 50k in both training and test set
train_data.groupby('CHURN').agg({'CHURN': 'count'}).show()	


# In[149]:


test_data.groupby('CHURN').agg({'CHURN': 'count'}).show()


# In[150]:


#You initialize lr by indicating the label column and feature columns. 
# Import `LogisticRegression`
from pyspark.ml.classification import LogisticRegression

# Initialize `lr`
lr = LogisticRegression(labelCol="CHURN",
                        featuresCol="features",
                        maxIter=10,
                        regParam=0.3)

# Fit the data to the model
clr = linearModel = lr.fit(train_data)


# In[151]:


# Print the coefficients and intercept for logistic regression
print("Coefficients: " + str(linearModel.coefficients))
print("Intercept: " + str(linearModel.intercept))


# In[152]:


#To generate prediction for your test set, you can use linearModel with transform() on test_data
# Make predictions on test data using the transform() method.
predictions = linearModel.transform(test_data)


# In[153]:


predictions.printSchema()


# In[154]:


selected = predictions.select("CHURN", "prediction", "probability")
selected.show(20)


# In[155]:


#We need to look at the accuracy metric to see how well (or bad) the model performs.

def accuracy_m(model): 
    predictions = model.transform(test_data)
    cm = predictions.select("CHURN", "prediction")
    acc = cm.filter(cm.CHURN == cm.prediction).count() / cm.count()
    print("Model accuracy: %.3f%%" % (acc * 100)) 

accuracy_m(model = linearModel)


# In[156]:


### Use ROC 
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Evaluate model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="CHURN")
print(evaluator.evaluate(predictions))
print(evaluator.getMetricName())


# In[157]:


#you can tune the hyperparameters. 
#Similar to scikit learn you create a parameter grid, and you add the parameters you want to tune. 
#To reduce the time of the computation, you only tune the regularization parameter with only two values.
#use 
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5])
             .build())


# In[158]:


from time import *
start_time = time()

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator, numFolds=5)

# Run cross validations
cvModel = cv.fit(train_data)
# likely take a fair amount of time
end_time = time()
elapsed_time = end_time - start_time
print("Time to train model: %.3f seconds" % elapsed_time)


# In[159]:


#accuracy of cv selected model
accuracy_m(model = cvModel)


# In[160]:


#We can exctract the recommended parameter by chaining cvModel.bestModel with extractParamMap()

bestModel = cvModel.bestModel
bestModel.extractParamMap()


# In[161]:


from onnxmltools.convert.common.data_types import StringTensorType
from onnxmltools.convert.common.data_types import FloatTensorType


# In[162]:


initial_types = [
    ("CHURN", StringTensorType([1, 1])),
    ("Gender", StringTensorType([1, 1])), 
    ("Age_Band", StringTensorType([1, 1])), 
    ("Tenure", StringTensorType([1, 1])), 
    ("Occupation", StringTensorType([1, 1])), 
    ("Education", StringTensorType([1, 1])), 
    ("Segment", StringTensorType([1, 1])), 
    ("Marital_Status", StringTensorType([1, 1])), 
    ("No_of_Accounts", FloatTensorType([1, 1])), 
    ("GDP_in_Billions_of_USSD", FloatTensorType([1, 1])),
    ("Inflation", FloatTensorType([1, 1])), 
    ("Population", FloatTensorType([1, 1])), 
    ("txn_amount_M1", FloatTensorType([1, 1])), 
    ("txn_vol_M1", FloatTensorType([1, 1])), 
    ("txn_amount_M2", FloatTensorType([1, 1])), 
    ("txn_vol_M2", FloatTensorType([1, 1])), 
    ("txn_amount_M3", FloatTensorType([1, 1])), 
    ("txn_vol_M3", FloatTensorType([1, 1])), 
    ("F1", FloatTensorType([1, 1])), 
    ("F2", StringTensorType([1, 1])), 
    ("Recency", StringTensorType([1, 1])), 
    ("Freq_M1", StringTensorType([1, 1])), 
    ("Freq_M2", StringTensorType([1, 1])), 
    ("F3", StringTensorType([1, 1]))]


# In[164]:


from onnxmltools import convert_sparkml
from onnxmltools.utils import save_model
model_onnx = convert_sparkml(pipelineModel, 'churn prediction model', initial_types)
model_onnx


# In[165]:


with open("model.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())


# In[166]:


connection_string = "DefaultEndpointsProtocol=https;AccountName=pocdls;AccountKey=MawNAHr8jfo2t8zJliDhgcv+Kq/ePPsPG8/wbZ8ab4UAdak7wpqOPYFbN+EBiweTIs/jh1YLZGm84FObJ6TH9Q==;EndpointSuffix=core.windows.net"
from azure.storage.blob import BlobClient
blob = BlobClient.from_connection_string(conn_str=connection_string, container_name="models",blob_name="onnx/model.onnx")

with open("./model.onnx", "rb") as data:
    blob.upload_blob(data, overwrite = True)


# In[167]:


import onnxruntime


# In[152]:


with open(os.path.join(os.getcwd(), 'churn prediction model'), "wb") as f:
    f.write(model_onnx.SerializeToString())


# In[153]:


session = onnxruntime.InferenceSession(os.path.join(os.getcwd(), 'churn prediction model'), None)
input_name = session.get_inputs()[0].name
print("Input name:", input_name)
input_shape = session.get_inputs()[0].shape
print("Input shape:", input_shape)
input_type = session.get_inputs()[0].type
print("Input type:", input_type)


# In[158]:


output_name = session.get_outputs()[0].name
print("output name:", output_name)
output_shape = session.get_outputs()[0].shape
print("output shape:", output_shape)
output_type = session.get_outputs()[0].type
print("output type:", output_type)


# In[156]:


predictions = session.run([CHURN_stringIdx],
{   
   initial_types
    })[0]


# In[176]:


import datetime
import pandas as pd

smodel = model_onnx.SerializeToString().hex()
models_tbl = 'ML_Models'
model_name = "churn prediction model"

# Create a DataFrame containing a single row with model name, training time and
# the serialized model, to be appended to the models table
now = datetime.datetime.now()
dfm = pd.DataFrame({'name':[model_name], 'timestamp':[now], 'model':[smodel]})
sdfm = spark.createDataFrame(dfm)
sdfm.show()


# In[177]:


extentsCreationTime = sc._jvm.org.joda.time.DateTime.now()

sp = sc._jvm.com.microsoft.kusto.spark.datasink.SparkIngestionProperties(
        True, None, None, None, None, extentsCreationTime, None, None)


# In[180]:


sdfm.write.format("com.microsoft.kusto.spark.synapse.datasource") .option("spark.synapse.linkedService", "<DX pool linked service>") .option("kustoDatabase", "<DX pool DB>") .option("kustoTable", models_tbl) .option("sparkIngestionPropertiesJson", sp.toString()) .option("tableCreateOptions","CreateIfNotExist") .mode("Append") .save()
Score in ADX

