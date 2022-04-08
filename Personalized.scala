package predict

import org.rogach.scallop._
import org.apache.spark.rdd.RDD
import ujson._

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.math
import shared.predictions._


class PersonalizedConf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val train = opt[String](required = true)
  val test = opt[String](required = true)
  val separator = opt[String](default=Some("\t"))
  val num_measurements = opt[Int](default=Some(0))
  val json = opt[String]()
  verify()
}

object Personalized extends App {
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

  val cosine_predictor = predictor_cosine(train)
  val one_predictor = predictor_one(train)
  val jaccard_predictor = predictor_jaccard(train)

  val user_mean_map = user_to_mean_rating(train)
  val user_normalized_mean = normalized_ratings(train,user_mean_map)
  val user_preprocessed_ratings = compute_pre_processed_ratings(user_normalized_mean)
  val user_item_map = compute_user_item_map(user_preprocessed_ratings)

  val cosine_mae=predictor_MAE(test,cosine_predictor)
  val one_mae=predictor_MAE(test,one_predictor)
  val jaccard_mae=predictor_MAE(test,jaccard_predictor)

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
          "1.Train" -> ujson.Str(conf.train()),
          "2.Test" -> ujson.Str(conf.test()),
          "3.Measurements" -> ujson.Num(conf.num_measurements())
        ),
        "P.1" -> ujson.Obj(
         
          "1.PredUser1Item1" -> ujson.Num(one_predictor(1,1)), // Prediction of item 1 for user 1 (similarity 1 between users)
          "2.OnesMAE" -> ujson.Num(one_mae) // MAE when using similarities of 1 between all users
        ),
        "P.2" -> ujson.Obj(
          
          "1.AdjustedCosineUser1User2" -> ujson.Num(cosine_similarity(user_item_map,1,2)), // Similarity between user 1 and user 2 (adjusted Cosine)
          "2.PredUser1Item1" -> ujson.Num(cosine_predictor(1,1)),  // Prediction item 1 for user 1 (adjusted cosine)
          "3.AdjustedCosineMAE" -> ujson.Num(cosine_mae) // MAE when using adjusted cosine similarity
        ),
        "P.3" -> ujson.Obj(
          
          "1.JaccardUser1User2" -> ujson.Num(jaccard_similarity(user_item_map,1,2)), // Similarity between user 1 and user 2 (jaccard similarity)
          "2.PredUser1Item1" -> ujson.Num(jaccard_predictor(1,1)),  // Prediction item 1 for user 1 (jaccard)
          "3.JaccardPersonalizedMAE" -> ujson.Num(jaccard_mae) // MAE when using jaccard similarity
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
