/**
  * Created by Dandachi on 01/12/2015.
  */

import java.io.File

import scala.io.Source

import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, Rating, MatrixFactorizationModel}
/* /user/mdandachi/ml-1m
 * /opt/cloudera/parcels/CDH-5.3.3-1.cdh5.3.3.p0.5/bin/
 */
object MovieLensALS {

  def main(args: Array[String]) {

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLeivel(Level.OFF)
    val sparkHome="/opt/cloudera/parcels/CDH-5.3.3-1.cdh5.3.3.p0.5/bin/"
    val jarName="target/scala-1.2.0-SNAPSHOT/movielens-als_1.2.0-SNAPSHOT.jar"
    val masterHostname="ws14-nn1.tl.teralab-datascience.fr:50070"

    if (args.length != 2) {
      println("Usage: /opt/cloudera/parcels/CDH-5.3.3-1.cdh5.3.3.p0.5/bin/spark-submit --driver-memory 2g --class MovieLensALS " +
        "target/scala-*/movielens-als-ssembly-*.jar movieLensHomeDir personalRatingsFile")
      sys.exit(1)
    }


    // set up environment

    val conf = new SparkConf()
      .setAppName("MovieLensALS")
      .set("spark.executor.memory", "8g")
      .setJars(Seq(jarFile))
    val sc = new SparkContext(conf)

    // load personal ratings

    val myRatings = loadRatings(args(1))
    val myRatingsRDD = sc.parallelize(myRatings, 1)

    // load ratings and movie titles

    val movieLensHomeDir = "hdfs://" + masterHostname

    val ratings = sc.textFile(new File(movieLensHomeDir, "ratings.dat").toString).map { line =>
      val fields = line.split("::")
      // format: (timestamp % 10, Rating(userId, movieId, rating))
      (fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble))
    }

    val movies = sc.textFile(new File(movieLensHomeDir, "movies.dat").toString).map { line =>
      val fields = line.split("::")
      // format: (movieId, movieName)
      (fields(0).toInt, fields(1))
    }.collect().toMap

    // your code here
    val numRatings = ratings.count
    val numUsers = ratings.map(_._2.user).distinct.count
    val numMovies = ratings.map(_._2.product).distinct.count

    println("Got " + numRatings + " ratings from "
      + numUsers + " users on " + numMovies + " movies.")

    // clean up
    sc.stop()
  }

  /** Compute RMSE (Root Mean Squared Error). */
  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], n: Long): Double = {
    2.0
  }

  /** Load ratings from file. */
  def loadRatings(path: String): Seq[Rating] = {
    (5,4,6)
  }
}

class test1 {

}
