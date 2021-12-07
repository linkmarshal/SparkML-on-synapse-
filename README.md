# SparkML-on-synapse-
# initialize sql context
from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)

#Todo: using SQLContext to read csv and assign to dataframe
df = sqlContext.read.csv('abfss://pocfsn@pocdls.dfs.core.windows.net/dbo.pocdata.csv', header=True, inferSchema= True)	

#Todo:printSchema
df =df.drop('CUSTID', 'Latest_TxnDate', 'State', 'LGA', 'Account_df.printSchema()Open_Date')
df.printSchema()

df

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

df.show(5, truncate = False)

#import libraries for pipeline
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

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

# 2. Index the label feature
# Convert label into label indices using the StringIndexer
CHURN_stringIdx =  StringIndexer(inputCol="CHURN", outputCol="newCHURN")
stages += [CHURN_stringIdx]

# 3. Add continuous variable
assemblerInputs = [c + "classVec" for c in CATE_FEATURES] + CONTI_FEATURES

# 4. Assemble the steps
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

stages

# Create a Pipeline.
pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(df)
model = pipelineModel.transform(df)

model

# To make the computation faster, you convert model to a DataFrame.
#You need to select newlabel and features from model using map.

from pyspark.ml.linalg import DenseVector
input_data = model.rdd.map(lambda x: (x["newCHURN"], DenseVector(x["features"])))

# import 
# from pyspark.ml.linalg import DenseVector
df_train = sqlContext.createDataFrame(input_data, ["CHURN", "features"])

df

df_train.show(2)

# Split the data into train and test sets
train_data, test_data = df_train.randomSplit([.8,.2],seed=1234)

# Let's count how many people with income below/above 50k in both training and test set
train_data.groupby('CHURN').agg({'CHURN': 'count'}).show()	

test_data.groupby('CHURN').agg({'CHURN': 'count'}).show()

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

# Print the coefficients and intercept for logistic regression
print("Coefficients: " + str(linearModel.coefficients))
print("Intercept: " + str(linearModel.intercept))

#To generate prediction for your test set, you can use linearModel with transform() on test_data
# Make predictions on test data using the transform() method.
predictions = linearModel.transform(test_data)

predictions.printSchema()

selected = predictions.select("CHURN", "prediction", "probability")
selected.show(20)

#We need to look at the accuracy metric to see how well (or bad) the model performs.

def accuracy_m(model): 
    predictions = model.transform(test_data)
    cm = predictions.select("CHURN", "prediction")
    acc = cm.filter(cm.CHURN == cm.prediction).count() / cm.count()
    print("Model accuracy: %.3f%%" % (acc * 100)) 

accuracy_m(model = linearModel)

### Use ROC 
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Evaluate model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="CHURN")
print(evaluator.evaluate(predictions))
print(evaluator.getMetricName())

#you can tune the hyperparameters. 
#Similar to scikit learn you create a parameter grid, and you add the parameters you want to tune. 
#To reduce the time of the computation, you only tune the regularization parameter with only two values.
#use 
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5])
             .build())

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

#accuracy of cv selected model
accuracy_m(model = cvModel)

#We can exctract the recommended parameter by chaining cvModel.bestModel with extractParamMap()

bestModel = cvModel.bestModel
bestModel.extractParamMap()

from onnxmltools.convert.common.data_types import StringTensorType
from onnxmltools.convert.common.data_types import FloatTensorType

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

from onnxmltools import convert_sparkml
from onnxmltools.utils import save_model
model_onnx = convert_sparkml(pipelineModel, 'churn prediction model', initial_types)
model_onnx

with open("model.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())

connection_string = "DefaultEndpointsProtocol=https;AccountName=pocdls;AccountKey=MawNAHr8jfo2t8zJliDhgcv+Kq/ePPsPG8/wbZ8ab4UAdak7wpqOPYFbN+EBiweTIs/jh1YLZGm84FObJ6TH9Q==;EndpointSuffix=core.windows.net"
from azure.storage.blob import BlobClient
blob = BlobClient.from_connection_string(conn_str=connection_string, container_name="models",blob_name="onnx/model.onnx")

with open("./model.onnx", "rb") as data:
    blob.upload_blob(data, overwrite = True)
