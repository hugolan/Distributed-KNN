package distributed

import org.rogach.scallop._
import org.apache.spark.rdd.RDD
import ujson._

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.math
import shared.predictions._

class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val train = opt[String](required = true)
  val test = opt[String](required = true)
  val separator = opt[String](default=Some("\t"))
  val master = opt[String](default=Some(""))
  val num_measurements = opt[Int](default=Some(1))
  val json = opt[String]()
  verify()
}

object DistributedBaseline extends App {
  var conf = new Conf(args) 

  // Remove these lines if encountering/debugging Spark
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val spark = if (conf.master() != "") {
    SparkSession.builder().master(conf.master()).getOrCreate()
  } else {
    SparkSession.builder().getOrCreate()
  }
  spark.sparkContext.setLogLevel("ERROR") 

  println("")
  println("******************************************************")

  println("Loading training data from: " + conf.train()) 
  val train = load(spark, conf.train(), conf.separator())
  println("Loading test data from: " + conf.test()) 
  val test = load(spark, conf.test(), conf.separator())

  val measurements = (1 to conf.num_measurements()).map(x => timingInMs(() => {
      val predictor = spark_predictor_baseline(train,spark)
      val mae = spark_predictor_MAE(test, predictor)
      mae
  }))
  val timings = measurements.map(t => t._2) // Retrieve the timing measurements
  val mae = mean(measurements.map(t => t._1))

  val global_avg = spark_compute_global_average(train)
  val avg1 = spark_compute_user_average(train,1)

  val user_all_avg = spark_compute_all_user_average(train)
  val user_all_avg_broadcast = spark.sparkContext.broadcast(user_all_avg.collect().toMap.withDefaultValue(global_avg))

  val item_dev = spark_all_item_deviation(train,user_all_avg_broadcast,global_avg)
  val item_dev_broadcast = spark.sparkContext.broadcast(item_dev.collect().toMap)
  val dev_1 = item_dev_broadcast.value.getOrElse(1,global_avg)
  val pred1 = avg1 + dev_1 * scale(avg1+dev_1,avg1)
  //val predictor = spark_predictor_baseline(train,spark)
  //val mae = spark_predictor_MAE(test,predictor)

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
          "3.Master" -> conf.master(),
          "4.Measurements" -> conf.num_measurements()
        ),
        "D.1" -> ujson.Obj(
          "1.GlobalAvg" -> ujson.Num(global_avg), // Datatype of answer: Double
          "2.User1Avg" -> ujson.Num(avg1),  // Datatype of answer: Double
          "3.Item1Avg" -> ujson.Num(spark_computer_item_average(train,1)),   // Datatype of answer: Double
          "4.Item1AvgDev" -> ujson.Num(dev_1), // Datatype of answer: Double,
          "5.PredUser1Item1" -> ujson.Num(pred1), // Datatype of answer: Double
          "6.Mae" -> ujson.Num(mae) // Datatype of answer: Double
        ),
        "D.2" -> ujson.Obj(
          "1.DistributedBaseline" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(timings)) // Datatype of answer: Double
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
