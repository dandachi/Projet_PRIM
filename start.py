# -*- coding: utf-8 -*-
"""
Éditeur de Spyder
>mvn clean package -Pdist,native-win -DskipTests -Dtar -Dmaven.javadoc.skip=true -X
Ceci est un script temporaire.
"""
#################CONNECTING PYTHON TO SPARK####################################
"needed profile for running spark"
import sys
import os

if 'SPARK_HOME' not in os.environ:
       os.environ['SPARK_HOME']='C:\spark'

if 'PYTHONPATH' not in os.environ:
       os.environ['PYTHONPATH']='C:\spark\python;C:\spark\python\lib\py4j-0.8.2.1-src.zip'       

SPARK_HOME=os.environ['SPARK_HOME']
PYTHONPATH=os.environ['PYTHONPATH']

sys.path.insert(0,os.path.join(PYTHONPATH,SPARK_HOME,"python","build"))
sys.path.insert(0,os.path.join(PYTHONPATH,SPARK_HOME,"python"))
###############################################################################

"start of code"
########################INITIALIZING###########################################
from pyspark import SparkContext,SparkConf
from pyspark.mllib import recommendation as mlrecom
conf = (SparkConf()
         .setMaster("local")
         .setAppName("pyspark")
         .set("spark.executor.memory","6g"))
sc=SparkContext(conf=conf)
#sc.stop()
rawUserArtistData = sc.textFile('C:\\Users\\Dandachi\\Desktop\\Projet_PRIM\\ds\\user_artist_data.txt')
rawArtistData=sc.textFile('C:\\Users\Dandachi\\Desktop\\Projet_PRIM\\ds\\artist_data.txt')
rawArtistAlias=sc.textFile('C:\\Users\Dandachi\\Desktop\\Projet_PRIM\\ds\\artist_alias.txt')
###############################################################################
k=0

########################FUNCTIONS##############################################
def artistToId(line):
    splitted=line.split('\t')
    if(len(splitted)<2):
        return [(0,"x")]
    else:
        aid=splitted[0]
        name=splitted[1]
        try:
            return [(int(aid),name.strip())]
        except:
            return [(0,"x")]
        
def artistToAlias(line):
    tokens=line.split('\t')
    try:
        return [(int(tokens[0]),int(tokens[1]))]
    except:
        return [(9999,0)]
        
def prepareRawUserArtistData(line,bArtistAlias):
    userID,artistId,count=map(int,line.split(' '))
    finalArtistID=bArtistAlias.value.get(artistId,artistId)
    return mlrecom.Rating(userID,finalArtistID,count)
    
###############################################################################


########################CODE###################################################
artistByID=rawArtistData.flatMap(lambda line: artistToId(line))
artistAlias=rawArtistAlias.flatMap(lambda line:artistToAlias(line)).collectAsMap()

bArtistAlias=sc.broadcast(artistAlias)
trainData=rawUserArtistData.map(lambda line:prepareRawUserArtistData(line,bArtistAlias)).cache()

model=mlrecom.ALS.trainImplicit(trainData,10,5,0.01,1.0)
