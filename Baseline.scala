package predict

import org.rogach.scallop._
import org.apache.spark.rdd.RDD

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.math
import shared.predictions._


class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val train = opt[String](required = true)
  val test = opt[String](required = true)
  val separator = opt[String](default=Some("\t"))
  val num_measurements = opt[Int](default=Some(4))
  val json = opt[String]()
  verify()
}

object Baseline extends App {
  // Remove these lines if encountering/debugging Spark
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val spark = SparkSession.builder()
    .master("local[1]")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR") 

  println("")
  println("******************************************************")

  var conf = new Conf(args) 
  // For these questions, data is collected in a scala Array 
  // to not depend on Spark
  println("Loading training data from: " + conf.train()) 
  val train = load(spark, conf.train(), conf.separator()).collect()
  println("Loading test data from: " + conf.test()) 
  val test = load(spark, conf.test(), conf.separator()).collect()


    
  //Global predictor
  val global_time = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    val global_predictor = predictor_global_avg(train)
    val global_mae = predictor_MAE(test, global_predictor)
    global_mae
  }))
  val global_time_result = global_time.map(t => t._2)
  val res_global = mean(global_time.map(t => t._1))

  //User predictor
  val user_time = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    val user_predictor = predictor_user_avg(train)
    val user_mae = predictor_MAE(test, user_predictor)
    user_mae
  }))
  val user_time_result = user_time.map(t => t._2)
  val res_user = mean(user_time.map(t => t._1))

  //Item predictor
  val item_time = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    val item_predictor = predictor_item_avg(train)
    val item_mae = predictor_MAE(test, item_predictor)
    item_mae
  }))
  val item_time_result = item_time.map(t => t._2)
  val res_item = mean(item_time.map(t => t._1))

  //Baseline predictor
  val baseline_time = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    val baseline_predictor = predictor_baseline(train)
    val baseline_mae = predictor_MAE(test, baseline_predictor)
    baseline_mae
  }))
  val baseline_time_result = baseline_time.map(t => t._2)
  val res_baseline = mean(baseline_time.map(t => t._1))

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
      var answers = ujson.Obj(
        "Meta" -> ujson.Obj(
          "1.Train" -> ujson.Str(conf.train()),
          "2.Test" -> ujson.Str(conf.test()),
          "3.Measurements" -> ujson.Num(conf.num_measurements())
        ),
        "B.1" -> ujson.Obj(
          "1.GlobalAvg" -> ujson.Num(compute_global_average(train)), // Datatype of answer: Double
          "2.User1Avg" -> ujson.Num(compute_user_average(train,1)),  // Datatype of answer: Double
          "3.Item1Avg" -> ujson.Num(compute_item_average(train,1)),   // Datatype of answer: Double
          "4.Item1AvgDev" -> ujson.Num(compute_average_deviation(train,1)), // Datatype of answer: Double
          "5.PredUser1Item1" -> ujson.Num(compute_predicted_rating(train,1,1)) // Datatype of answer: Double
        ),
        "B.2" -> ujson.Obj(
          "1.GlobalAvgMAE" -> ujson.Num(res_global), // Datatype of answer: Double
          "2.UserAvgMAE" -> ujson.Num(res_user),  // Datatype of answer: Double
          "3.ItemAvgMAE" -> ujson.Num(res_item),   // Datatype of answer: Double
          "4.BaselineMAE" -> ujson.Num(res_baseline) // Datatype of answer: Double
        ),
        "B.3" -> ujson.Obj(
          "1.GlobalAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(global_time_result)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(global_time_result)) // Datatype of answer: Double
          ),
          "2.UserAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(user_time_result)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(user_time_result)) // Datatype of answer: Double
          ),
          "3.ItemAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(item_time_result)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(item_time_result)) // Datatype of answer: Double
          ),
          "4.Baseline" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(baseline_time_result)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(baseline_time_result)) // Datatype of answer: Double
          )
        )
      )

      val json = ujson.write(answers, 4)
      println(json)
      println("Saving answers in: " + jsonFile)
      printToFile(json.toString, jsonFile)
    }
  }

  println("")
  spark.close()
}
