from pyalink.alink import *

import pandas as pd

AlinkGlobalConfiguration.setPrintProcessInfo(True)


import json
source = CsvSourceBatchOp()\
		.setFilePath('https://alink-test-2.oss-cn-shanghai.aliyuncs.com/xgboost/vs.txt')\
		.setSchemaStr('id bigint, vec string')\
		.setFieldDelimiter('|')
model = VectorNearestNeighborTrainLocalOp()\
          .setIdCol("id")\
          .setSelectedCol("vec")\
          .setMetric("EUCLIDEAN")\
          .linkFrom(source)
result = VectorNearestNeighborPredictLocalOp()\
        .setSelectedCol("vec")\
        .setOutputCol("topN")\
        .setTopN(3)\
        .setNumThreads(6)\
        .linkFrom(model, source)

result.printStatistics()
