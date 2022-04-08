package predict

import org.rogach.scallop._
import org.apache.spark.rdd.RDD
import ujson._

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.math
import shared.predictions._
import org.apache.spark.broadcast.Broadcast


class kNNConf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val train = opt[String](required = true)
  val test = opt[String](required = true)
  val separator = opt[String](default=Some("\t"))
  val num_measurements = opt[Int](default=Some(10))
  val json = opt[String]()
  verify()
}

object kNN extends App {
  // Remove these lines if encountering/debugging Spark
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val spark = SparkSession.builder()
    .master("local[1]")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR") 

  println("")
  println("******************************************************")

  var conf = new PersonalizedConf(args) 
  println("Loading training data from: " + conf.train()) 
  val train = load(spark, conf.train(), conf.separator()).collect()
  println("Loading test data from: " + conf.test()) 
  val test = load(spark, conf.test(), conf.separator()).collect()
  
  //measure time
  val knn_time = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    val knn_predictor = predictor_knn(train,300)
    val knn_mae = predictor_MAE(test, knn_predictor)
    knn_mae
  }))
  val knn_time_result = knn_time.map(t => t._2)
  print(knn_time_result)
  val res_knn = mean(knn_time.map(t => t._1))
 
  val knn1=compute_similarities_knn_for_user_u(train,1)

  // Save answers as JSON
  def printToFile(content: String, 
                  location: String = "./answers.json") =
    Some(new java.io.PrintWriter(location)).foreach{
      f => try{
        f.write(content)
      } finally{ f.close }
  }
  conf.json.toOption match {
    case None => ; 
    case Some(jsonFile) => {
      
      val answers = ujson.Obj(
        "Meta" -> ujson.Obj(
          "1.Train" -> conf.train(),
          "2.Test" -> conf.test(),
          "3.Measurements" -> conf.num_measurements()
        ),
        "N.1" -> ujson.Obj(
          "1.k10u1v1" -> ujson.Num(compute_similarity_knn_with_user_v(1,1,10,knn1)), // Similarity between user 1 and user 1 (k=10)
          "2.k10u1v864" -> ujson.Num(compute_similarity_knn_with_user_v(1,864,10,knn1)), // Similarity between user 1 and user 864 (k=10)
          "3.k10u1v886" -> ujson.Num(compute_similarity_knn_with_user_v(1,886,10,knn1)), // Similarity between user 1 and user 886 (k=10)
          "4.PredUser1Item1" -> ujson.Num(predictor_knn(train,10)(1,1)) // Prediction of item 1 for user 1 (k=10)
        ),
        "N.2" -> ujson.Obj(
          "1.kNN-Mae" -> List(10,30,50,100,200,300,400,800,943).map(k => 
              List(
                k,
                knn_mae(train,test,k)
              )
          ).toList
        ),
        "N.3" -> ujson.Obj(
          "1.kNN" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(knn_time_result)),
            "stddev (ms)" -> ujson.Num(std(knn_time_result))
          )
        )
      )
      val json = write(answers, 4)
      println(json)
      println("Saving answers in: " + jsonFile)
      printToFile(json, jsonFile)
    }
  }


  println("")
  spark.close()
}
