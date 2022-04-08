package shared

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.broadcast.Broadcast
import scala.collection.mutable.PriorityQueue

package object predictions
{
  case class Rating(user: Int, item: Int, rating: Double)

  def timingInMs(f : ()=>Double ) : (Double, Double) = {
    val start = System.nanoTime() 
    val output = f()
    val end = System.nanoTime()
    return (output, (end-start)/1000000.0)
  }

  def mean(s :Seq[Double]): Double =  if (s.size > 0) s.reduce(_+_) / s.length else 0.0
  def std(s :Seq[Double]): Double = {
    if (s.size == 0) 0.0
    else {
      val m = mean(s)
      scala.math.sqrt(s.map(x => scala.math.pow(m-x, 2)).sum / s.length.toDouble)
    }
  }

  def toInt(s: String): Option[Int] = {
    try {
      Some(s.toInt)
    } catch {
      case e: Exception => None
    }
  }

  def load(spark : org.apache.spark.sql.SparkSession,  path : String, sep : String) : org.apache.spark.rdd.RDD[Rating] = {
       val file = spark.sparkContext.textFile(path)
       return file
         .map(l => {
           val cols = l.split(sep).map(_.trim)
           toInt(cols(0)) match {
             case Some(_) => Some(Rating(cols(0).toInt, cols(1).toInt, cols(2).toDouble))
             case None => None
           }
       })
         .filter({ case Some(_) => true 
                   case None => false })
         .map({ case Some(x) => x 
                case None => Rating(-1, -1, -1)})
  }

//eq. 3
def scale(x : Double, avg : Double) : Double = {
  if(x>avg){
    return 5 - avg
  }else if(x<avg){
    return avg - 1
  }else{
    return 1
  }
}

//calculate normalize deviation by rating and average value
def normalized_deviation(rating : Double, avg : Double) : Double = {
  return (rating - avg)/scale(rating,avg)
}

//---------------------------------------------------
//3.1

//calculate global average (over all users and items)
def compute_global_average(ratings : Array[Rating]) : Double  = {
  val l = ratings.length
  return ratings.map(_.rating).sum/l
}

//calculate average rating for user u over all items
def compute_user_average(ratings : Array[Rating], user : Int)  : Double  = {
  val user_items_rating = ratings.filter(_.user == user).map(_.rating)
  val l = user_items_rating.length
  if(l != 0){
      return user_items_rating.sum/l
  }else{
      return compute_global_average(ratings)
  }
}

//calculate average rating for user
def compute_all_average(ratings: Array[Rating]): Map[Int, Double] = {
    ratings.groupBy(x=>x.user).map{case (key,value) => (key, mean(value.map(x => x.rating)))}
  }

//calculate average rating for item i over all users
def compute_item_average(ratings : Array[Rating], item : Int) : Double  = {
  val item_avg = ratings.filter(_.item == item).map(_.rating)
  val l = item_avg.length
  if(l != 0){
    return item_avg.sum/l
  }else{
    return compute_global_average(ratings)
  }
}

//calculate global average deviation (eq. 4)
def compute_average_deviation(ratings : Array[Rating], item : Int)  : Double = {
  val item_user = ratings.filter(_.item == item).map(x=>normalized_deviation(x.rating,compute_user_average(ratings,x.user)))
  val l = item_user.length
  if(l != 0){
    return item_user.sum/l
  }else{
    return 0.0
  }
}

//calculate predicted rating (eq. 5)
def compute_predicted_rating(ratings : Array[Rating], user : Int, item : Int)  : Double  = {
  if(ratings.filter(_.item==item).length>0){
    val user_avg = compute_user_average(ratings,user)
    val avg_deviation = compute_average_deviation(ratings,item)
    return user_avg + avg_deviation * scale((user_avg+avg_deviation),user_avg)
  }
  else{
    return compute_user_average(ratings,user)
  }
}

//3.2
//predictor based on global average rating
def predictor_global_avg(train: Array[Rating]) : (Int, Int) => Double = {
  val global_avg = compute_global_average(train)
  return (u : Int, i: Int) => global_avg
}

//predictor based on average rating for user
def predictor_user_avg(train: Array[Rating]) : (Int, Int) => Double =  {
  val values = train.groupBy(x=>x.user).map{case (key,value) => (key,mean(value.map(x=>x.rating)))}
  val global_avg = compute_global_average(train)
  return (u : Int, i: Int) => values.getOrElse(u,global_avg)
}

//predictor based on average rating for item
def predictor_item_avg(train: Array[Rating]) : (Int, Int) => Double =  {
  val values = train.groupBy(x=>x.item).map{case (key,value) => (key,mean(value.map(x=>x.rating)))}
  val global_avg = compute_global_average(train)
  return (u : Int,i : Int) => values.getOrElse(i,global_avg)
}

//predictor based on eq. 5
def predictor_baseline(train: Array[Rating]): (Int, Int) => Double = {
  val avg = train.groupBy(x=>x.user).map{case (key,value) => (key,mean(value.map(x=>x.rating)))}
  val global_avg = compute_global_average(train)
  val values = avg.withDefaultValue(global_avg)
  val item_deviation = train.groupBy(x=>x.item).map{case (key,value) => (key, mean(value.map(x=>(x.user,x.rating)).map{case (user,rating)=> (rating - values(user))/scale(rating,values(user))}))}
  return (u: Int,i : Int) => values.getOrElse(u,global_avg) + item_deviation.getOrElse(i, 0.0) * scale((values.getOrElse(u,global_avg) + item_deviation.getOrElse(i, 0.0)), values.getOrElse(u,global_avg))
}

//3.3
//calculate MAE value for the given predictor
def predictor_MAE(test: Array[Rating], prediction_function: (Int,Int) => Double) : Double  = {
  val inter = test.map{case Rating(user,item,rating) => (prediction_function(user,item)-rating).abs}
  return inter.sum/inter.length
}

//---------------------------------------------------
//4
//calculate global average (over all users and items) using spark
def spark_compute_global_average(ratings : org.apache.spark.rdd.RDD[Rating]) : Double = {
    val inter = ratings.map(x => x.rating)
    return inter.sum/inter.count()
}

//calculate average rating for user u over all items using spark
def spark_compute_user_average(ratings : org.apache.spark.rdd.RDD[Rating], user : Int) : Double  = {
    val inter = ratings.filter(x => x.user == user).map(x => x.rating)
    return inter.sum/inter.count()
}

//calculate average rating for item i over all users using spark
def spark_computer_item_average(ratings : org.apache.spark.rdd.RDD[Rating], item : Int) : Double = {
    val inter = ratings.filter(x => x.item == item).map(x => x.rating)
    return inter.sum/inter.count()
}

//calculate global average deviation (eq. 4) using spark
def spark_compute_average_deviation(ratings : org.apache.spark.rdd.RDD[Rating], item : Int) : Double = {
  val item_user = ratings.filter(x => x.item == item).map(x => normalized_deviation(x.rating,spark_compute_user_average(ratings,x.user)))
  val l = item_user.count()
  if(l != 0){
    return item_user.sum/l
  }else{
    return 0.0
  }
}

//predictor based on global average rating using spark
def spark_predictor_global(train : org.apache.spark.rdd.RDD[Rating], spark: SparkSession) : (Int, Int) => Double = {
  val global_avg = spark_compute_global_average(train)
  return (u : Int, i : Int) => global_avg
}

//predictor based on average rating for user using spark
def spark_predictor_user(train : org.apache.spark.rdd.RDD[Rating], spark: SparkSession) : (Int, Int) => Double = {
  val global_avg = spark_compute_global_average(train)
  val values = train.map{case Rating(user, item, rating) => (user,(rating,1))}.reduceByKey((x,y)=>(x._1 + y._1, x._2 + y._2)).map{case (key,value)=> (key, value._1/value._2)}
  val values_broadcast = spark.sparkContext.broadcast(values.collect().toMap.withDefaultValue(global_avg))
  return (u: Int, i: Int) => values_broadcast.value.getOrElse(i,global_avg)
}

//calculate average for all users using spark
def spark_compute_all_user_average(train : org.apache.spark.rdd.RDD[Rating]) : RDD[(Int, Double)] = {
    return train.map{case Rating(user, item, rating) => (user,(rating,1))}.reduceByKey((x,y)=>(x._1 + y._1, x._2 + y._2)).map{case (key,value)=> (key, value._1/value._2)}
}

//predictor based on average rating for item using spark
def spark_predictor_item(train : org.apache.spark.rdd.RDD[Rating], spark: SparkSession) : (Int, Int) => Double = {
  val global_avg = spark_compute_global_average(train)
  val values = train.map{case Rating(user, item, rating) => (item,(rating,1))}.reduceByKey((x,y)=>(x._1 + y._1, x._2 + y._2)).map{case (key,value)=> (key, value._1/value._2)}
  val values_broadcast = spark.sparkContext.broadcast(values.collect().toMap.withDefaultValue(global_avg))
  return (u: Int, i: Int) => values_broadcast.value.getOrElse(i,global_avg)
}


//predictor based on eq. 5 using spark
def spark_predictor_baseline(train : org.apache.spark.rdd.RDD[Rating], spark: SparkSession) : (Int, Int) => Double = {
  val global_avg = spark_compute_global_average(train)
  val values = train.map{case Rating(user, item, rating) => (user,(rating,1))}.reduceByKey((x,y)=>(x._1 + y._1, x._2 + y._2)).map{case (key,value)=> (key, value._1/value._2)}
  val values_broadcast = spark.sparkContext.broadcast(values.collect().toMap.withDefaultValue(global_avg))
  val values_deviation = spark_all_item_deviation(train,values_broadcast,global_avg)
  val values_deviation_broadcast = spark.sparkContext.broadcast(values_deviation.collect().toMap)
  return (u: Int, i: Int) => values_broadcast.value.getOrElse(u,global_avg) + values_deviation_broadcast.value.getOrElse(i, 0.0) * scale((values_broadcast.value.getOrElse(u,global_avg) + values_deviation_broadcast.value.getOrElse(i, 0.0)), values_broadcast.value.getOrElse(u,global_avg))
}

//calculate deviation for all items using spark
def spark_all_item_deviation(test : org.apache.spark.rdd.RDD[Rating], broadcasted_values : Broadcast[Map[Int,Double]], global_avg : Double) : RDD[(Int, Double)] = {
  test.map{case Rating(user,item,rating) => (item, (normalized_deviation(rating,broadcasted_values.value.getOrElse(user,global_avg)),1.0))}.reduceByKey((x,y) => (x._1 + y._2, x._2 + y._2)).map{case (key,value) => (key, value._1/value._2)}
}

def spark_predictor_MAE(test : org.apache.spark.rdd.RDD[Rating], prediction_function : (Int,Int) => Double) : Double  = {
  val inter = test.map(x => (prediction_function(x.user, x.item), x.rating)).map{case (x, y) => (x - y).abs}
  return inter.sum/inter.sum
}

//---------------------------------------------------
//5
//calculate map user->mean rating for all users
def user_to_mean_rating(ratings : Array[Rating]) : Map[Int,Double] = {
    val users = ratings.map{case Rating(u,i,r) => u}.distinct
    return (users.map{user => (user,compute_user_average(ratings, user))}).toMap
  }

//normalizes all the ratings in maps user->rating
def normalized_ratings(ratings: Array[Rating], map : Map[Int,Double]) : Array[Rating] = {
    return ratings.map{case Rating(user,item,rating) => Rating(user,item,(rating-map(user))/scale(rating,map(user)))}
}

//calculate euclidean norm
def euclidean_norm(array : Array[Double]) : Double = {
  return math.sqrt(array.map(x => x*x).sum)
}

//calculate eq.9
def compute_pre_processed_ratings(normalized_ratings : Array[Rating]) : Array[Rating] = {
  val users = normalized_ratings.map{case Rating(user,item,rating) => user}.distinct
  val map = (users.map{user => (user, euclidean_norm(normalized_ratings.filter(x => x.user == user).map{case Rating(user,item,rating) => rating}))}).toMap
  return normalized_ratings.map{case Rating(user,item,rating) => Rating(user,item,rating/map(user))}
}

//calculate map user->(Array[items],item->rating)
def compute_user_item_map(pre_processed_ratings : Array[Rating]) : Map[Int,(Array[Int],Map[Int,Double])] = {
  val users = pre_processed_ratings.map{case Rating(u,i,r) => u}.distinct
  return (users.map{user => (user,(pre_processed_ratings.filter(_.user == user).map{case Rating(user,item,rating) => item},(pre_processed_ratings.filter(_.user == user).map{case Rating(user,item,rating) => (item,rating)}).toMap))}).toMap
}

//calculate cosine similarity
def cosine_similarity(user_item_map : Map[Int,(Array[Int],Map[Int,Double])], user_1:Int, user_2:Int) : Double = {
  if (user_item_map.contains(user_1) || user_item_map.contains(user_2)) {
    var user_1_map = user_item_map(user_1)
    var user_2_map = user_item_map(user_2)
    var intersection = (user_1_map._1).intersect(user_2_map._1)
    return intersection.map(x => user_1_map._2(x) * user_2_map._2(x)).sum
  }
  else return 0.0
}

//calculate weighted ratings for user and item (eq. 7) 
def compute_item_similarity(user_normalized_mean : Array[Rating], u : Int, i : Int, similarity : (Int,Int) => Double) : Double = {
    val item_rating = user_normalized_mean.filter(x => x.item == i).map{case Rating(user,value,rating) => (similarity(u,user),rating)}
    val similarity_sum = (item_rating.map{case (x,y) => x.abs}).sum
    if(similarity_sum != 0){
      return (item_rating.map{case (x,y) => x*y}.sum)/similarity_sum
    }else{
      return 0.0
    }
}

//predictor based on cosine similarity
def predictor_cosine(ratings : Array[Rating]) : (Int,Int) => Double = {
  val user_mean_map = user_to_mean_rating(ratings)
  val user_normalized_mean = normalized_ratings(ratings,user_mean_map)
  val user_preprocessed_ratings = compute_pre_processed_ratings(user_normalized_mean)
  val user_item_map = compute_user_item_map(user_preprocessed_ratings)
  val cosine_similarity_var : ((Int,Int) => Double) = (user_1,user_2) => cosine_similarity(user_item_map,user_1,user_2)
  return (u : Int, i : Int) => {
    val global_avg = compute_global_average(ratings)
    val ru = user_mean_map.getOrElse(u,global_avg)
    val ri = compute_item_similarity(user_normalized_mean,u,i,cosine_similarity_var) 
    ru + ri * scale(ru + ri, ru)
  }
}

//predictor based on uniform similarity
def predictor_one(ratings : Array[Rating]) : (Int,Int) => Double = {
  val user_mean_map = user_to_mean_rating(ratings)
  val user_normalized_mean = normalized_ratings(ratings,user_mean_map)
  val user_preprocessed_ratings = compute_pre_processed_ratings(user_normalized_mean)
  val user_item_map = compute_user_item_map(user_preprocessed_ratings)
  val one_similarity : ((Int,Int) => Double) = (user_1,user_2) => 1
  return (u : Int, i : Int) => {
    val global_avg = compute_global_average(ratings)
    val ru = user_mean_map.getOrElse(u,global_avg)
    val ri = compute_item_similarity(user_normalized_mean,u,i,one_similarity) 
    ru + ri * scale(ru + ri, ru)
  }
}

//calculate jaccard similarity
def jaccard_similarity(user_item_map : Map[Int,(Array[Int],Map[Int,Double])], user_1:Int, user_2:Int) : Double = {
  if (user_item_map.contains(user_1) || user_item_map.contains(user_2)) {
    val user_1_map = user_item_map(user_1)
    val user_2_map = user_item_map(user_2)
    val intersection = (user_1_map._1).toSet.intersect((user_2_map._1).toSet)
    val union = (user_1_map._1).toSet.union((user_2_map._1).toSet)
    return intersection.size.toDouble / union.size.toDouble
  }
  else return 0.0
}

//predictor based on jaccard similarity
def predictor_jaccard(ratings : Array[Rating]) : (Int,Int) => Double = {
  val user_mean_map = user_to_mean_rating(ratings)
  val user_normalized_mean = normalized_ratings(ratings,user_mean_map)
  val user_preprocessed_ratings = compute_pre_processed_ratings(user_normalized_mean)
  val user_item_map = compute_user_item_map(user_preprocessed_ratings)
  val jaccard_similarity_var : ((Int,Int) => Double) = (user_1,user_2) => jaccard_similarity(user_item_map,user_1,user_2)
  return (u : Int, i : Int) => {
    val global_avg = compute_global_average(ratings)
    val ru = user_mean_map.getOrElse(u,global_avg)
    val ri = compute_item_similarity(user_normalized_mean,u,i,jaccard_similarity_var) 
    ru + ri * scale(ru + ri, ru)
  }
}

//---------------------------------------------------
//6

//calculate all similarities for user
def compute_similarities_knn_for_user_u(ratings:Array[Rating],u:Int) : Map[Int,(Double,Double)] = {
    val user_mean_map = user_to_mean_rating(ratings)
    val user_normalized_mean = normalized_ratings(ratings,user_mean_map)
    val user_preprocessed_ratings = compute_pre_processed_ratings(user_normalized_mean)
    val user_item_map = compute_user_item_map(user_preprocessed_ratings)
    val cosine_similarity_var : ((Int,Int) => Double) = (user_1,user_2) => cosine_similarity(user_item_map,user_1,user_2)
    val item_rating = user_normalized_mean.filter(x => x.user!=u).map{case Rating(user,value,rating) => (user,(cosine_similarity_var(u,user),rating))}.toMap
    return item_rating
    }

//calculate similarity between users 
def compute_similarity_knn_with_user_v(u:Int,v:Int,k:Int,item_rating:Map[Int,(Double,Double)]): Double = {
  if (v==u) return 1.0
  else {
    val ordered_similarities = item_rating.toSeq.sortBy(_._2._1)(Ordering[Double].reverse).take(k).toMap
    return ordered_similarities.getOrElse(v,(0.0,0.0))._1
  }
}

//calculate all similarities for all users
def compute_all_similarities_knn(ratings:Array[Rating], k:Int, user_item_map : Map[Int,(Array[Int],Map[Int,Double])]) : Map[Int,Map[Int,Double]] = {
    var knn_map_similarities : Array[(Int,Map[Int,Double])] = Array()
    for(user_1 <- user_item_map.keySet){
      var map_similarities_user_1 : Array[(Int, Double)] = Array()
      for(user_2 <- user_item_map.keySet){
        if(user_1 != user_2){
          //prepend new value for better runtime
          map_similarities_user_1 = (user_2, cosine_similarity(user_item_map,user_1,user_2)) +: map_similarities_user_1 
        }
      }
      map_similarities_user_1 = map_similarities_user_1.sortBy(x => x._2)(Ordering[Double].reverse).take(k)
      knn_map_similarities = (user_1,map_similarities_user_1.toMap) +: knn_map_similarities
    }
    knn_map_similarities.toMap
}

//calculate weighted ratings for user and item (eq. 7) for knn method 
def compute_item_similarity_knn(user_normalized_mean : Array[Rating], u : Int, i : Int, similarity : Map[Int,Map[Int,Double]]) : Double = {
    val item_rating = user_normalized_mean.filter(x => x.item == i && x.user!=u).map{case Rating(user,value,rating) => ((if(similarity(u).contains(user)){similarity(u)(user)} else {0.0}),rating)}
    val similarity_sum = (item_rating.map{case (x,y) => x.abs}).sum
    if(similarity_sum != 0){
      val item_rating_top= (item_rating.map{case (knn,rating) => knn*rating}).sum
      return item_rating_top/similarity_sum
    }else{
      return 0.0
    }
}

//predictor based on knn and cosine similarity
def predictor_knn(ratings : Array[Rating], k:Int) : (Int,Int) => Double = {
  val user_mean_map = user_to_mean_rating(ratings)
  val user_normalized_mean = normalized_ratings(ratings,user_mean_map)
  val user_preprocessed_ratings = compute_pre_processed_ratings(user_normalized_mean)
  val user_item_map = compute_user_item_map(user_preprocessed_ratings)
  val compute_similarities_knn = compute_all_similarities_knn(ratings,k,user_item_map)
  val cosine_similarity_var : ((Int,Int) => Double) = (user_1,user_2) => cosine_similarity(user_item_map,user_1,user_2)
  return (u : Int, i : Int) => {
    val global_avg = compute_global_average(ratings)
    val ru = user_mean_map.getOrElse(u,global_avg)
    val ri = compute_item_similarity_knn(user_normalized_mean,u,i,compute_similarities_knn) 
    ru + ri * scale(ru + ri, ru)
  }
}

//calculate MAE value for the given predictor knn
def knn_mae(train:Array[Rating],test:Array[Rating],k:Int): Double={
  val knn_predictor = predictor_knn(train,k)
  return predictor_MAE(test, knn_predictor)
}

//---------------------------------------------------
//7
//calculate top recommendations for user
def  recommend(ratings:Array[Rating],u:Int,n:Int,predictor:(Int, Int) => Double):Map[Int,Double]={
  val items_to_predict=(ratings.map{case Rating(user, item, rating)=>item}.distinct.toSet.diff(ratings.filter(x=>x.user==u).map{case Rating(user,item,value)=>item}.distinct.toSet)).toArray
  val item_prediction=items_to_predict.map(item => (item,predictor(u,item)))
  return item_prediction.toSeq.sortBy(_._1)(Ordering[Int]).sortBy(_._2)(Ordering[Double].reverse).take(n).toMap
  }
}