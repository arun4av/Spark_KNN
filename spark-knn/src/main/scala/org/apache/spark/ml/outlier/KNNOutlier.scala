package org.apache.spark.ml.classification

import org.apache.spark.SparkException

import breeze.linalg.{Vector => BV}
import scala.collection.Map
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}
import org.apache.spark.ml.{Model, Predictor}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{ArrayType, DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._
import org.apache.spark.mllib.knn.KNNUtils
import org.apache.spark.ml.util.MetadataUtils
import org.apache.spark.ml.clustering.KMeans

import scala.collection.mutable.{ArrayBuffer, LinkedHashMap, PriorityQueue}
import scala.collection.{Map => MAP}
import scala.collection.mutable.{Map => muMap}

class KNNOutlier(override val uid: String) extends Predictor[Vector, KNNOutlier, KNNOutlierModel] {
  def this() = this(Identifiable.randomUID("naiveknnc"))

  override def copy(extra: ParamMap): KNNOutlier = defaultCopy(extra)

  var clusterPoints = muMap[Vector, ArrayBuffer[VectorWithNorm]]()

  var clusterRadius = muMap[Vector, Double]()

  var clusterCenters: Array[Vector] = null

  var k: Int = 1

  var treshold: Int = 1

  def setK(value: Int): this.type = {
    k = value
    this
  }

  def setTreshold(value: Int): this.type = {
    treshold = value
    this
  }

  override def train(dataset: DataFrame): KNNOutlierModel = {
    // Extract columns from data.  If dataset is persisted, do not persist oldDataset.
    val instances = extractLabeledPoints(dataset).map {
      case LabeledPoint(label: Double, features: Vector) => (label, features)
    }
    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE

    if (handlePersistence) instances.persist(StorageLevel.MEMORY_AND_DISK)

    val labelSchema = dataset.schema("label")

    val computeNumClasses: () => Int = () => {
      val Row(maxLabelIndex: Double) = dataset.agg(max(col("label").cast(DoubleType))).head()
      // classes are assumed to be numbered from 0,...,maxLabelIndex
      maxLabelIndex.toInt + 1
    }

    val numClasses = MetadataUtils.getNumClasses(labelSchema).fold(computeNumClasses())(identity)

    val points = instances.map{
      case (label, features) => (label, new VectorWithNorm(features))
    }

    var kVal = Math.min(50, Math.sqrt(dataset.count()/2).toInt)

    if(kVal>1)
    {
      val kMeans = new KMeans().setK(kVal)
      val kMeansModel = kMeans.fit(dataset)
      clusterCenters = kMeansModel.clusterCenters

      for(i <- 0 to clusterCenters.size-1) {
        clusterPoints.put(clusterCenters(i), new ArrayBuffer[VectorWithNorm]())
        clusterRadius.put(clusterCenters(i), Double.MaxValue)
      }
    }
    else {
      val kMeans = new KMeans().setK(2)
      val kMeansModel = kMeans.fit(dataset)
      clusterCenters = kMeansModel.clusterCenters
      clusterPoints.put(clusterCenters(0), new ArrayBuffer[VectorWithNorm]())
      clusterRadius.put(clusterCenters(0), Double.MaxValue)
    }

    // Map every point to thier respective cluster centers
    val intermediateRDD = points.map {
      case (label, point) =>
        var dist:Double = Double.MaxValue
        var curr_dist:Double = 0.0
        var curr_center:Vector = null
        for(value <- clusterCenters) {
          curr_dist = point.fastSquaredDistance(new VectorWithNorm(value))
          if(dist > curr_dist) {
            dist = curr_dist
            curr_center = value
          }
        }
        ((point, label), (curr_center, dist))
    }

    val clusterPointsRDD = intermediateRDD.map {
      case (pointWithLabel, (curr_center, dist)) =>
        (curr_center, pointWithLabel)
    }.groupByKey()

    val clusterRadiusRDD = intermediateRDD.map {
      case (point, value) =>
        value
    }.topByKey(1)
      .map{
        case (curr_center, distArray) =>
          (curr_center, distArray(0))
      }

    val dataPoints = instances.map{
      case (label, features) => (new VectorWithNorm(features), 0)
    }

    new KNNOutlierModel(uid, k, treshold, numClasses, clusterCenters, clusterPointsRDD.asInstanceOf[RDD[(Vector, MAP[VectorWithNorm, Double])]], clusterRadiusRDD, dataPoints.asInstanceOf[RDD[(VectorWithNorm, Int)]])
  }
}

class KNNOutlierModel(override val uid: String,
                      val k: Int,
                      val treshold: Int,
                      val _numClasses: Int,
                      val clusterCenters: Array[Vector],
                      val clusterPointsRDD: RDD[(Vector, MAP[VectorWithNorm, Double])],
                      val clusterRadiusRDD: RDD[(Vector, Double)],
                      val dataPoints: RDD[(VectorWithNorm, Int)])
                      extends ProbabilisticClassificationModel[Vector, KNNOutlierModel] {

  override def numClasses: Int = _numClasses

  val outliers = ArrayBuffer[Any]()

  override def transform(dataset: DataFrame): DataFrame = {
    dataset
  }

  def detect(): ArrayBuffer[Any] = {

    val clusterPoints : MAP[Vector, MAP[VectorWithNorm, Double]] = clusterPointsRDD.collectAsMap()
    val clusterRadius : MAP[Vector, Double] = clusterRadiusRDD.collectAsMap()

    var inputPointsWithClusters = muMap[VectorWithNorm, ArrayBuffer[Vector]]()

    val pointswithCount: Map[Long, Map[VectorWithNorm, Int]] = dataPoints.zipWithUniqueId()
      .map {
        case ((point, count), i) =>

          class HeapData(var dist: Double, var data: VectorWithNorm) extends Ordered[HeapData] {
            def compare(that: HeapData): Int = (this.dist) compare (that.dist)
          }
          var dist:Double = Double.MaxValue
          var curr_dist:Double = 0.0
          var center:Vector = null
          var maxRadius:Double = 0.0

          for(value <- clusterCenters) {
            curr_dist = point.fastSquaredDistance(new VectorWithNorm(value))
            maxRadius = clusterRadius.get(value).get
            if(dist > (curr_dist + maxRadius)) {
              dist = curr_dist + maxRadius
              center = value
            }
          }

          val Q = PriorityQueue[HeapData]()

          for((eachClusterPoint, label) <- clusterPoints.get(center).get) {
            val dist = point.fastSquaredDistance(eachClusterPoint)
            var hd = new HeapData(dist , eachClusterPoint)
            if(Q.size < k+1) {
              Q += hd
            }
            else if (dist < Q.head.dist) {
              Q.dequeue()
              Q += hd
            }
          }

          var points = muMap[VectorWithNorm, Int]()

          for(i <- 0 to Q.size-1) {
            var Data = Q.dequeue().data

            if(points.contains(Data))
              points.put(Data, points.get(Data).get + 1)
            else
              points.put(Data, 1)
          }

          val vector = new Array[Double](numClasses)

          val values = new ArrayBuffer[Any]
          (i, points)
      }.collectAsMap()

    val finalCount = muMap[VectorWithNorm, Int]()

    for(mapValues <- pointswithCount.values) {
      for((k, v) <- mapValues) {
        if(finalCount.contains(k))
          finalCount.put(k, finalCount.get(k).get + 1)
        else
          finalCount.put(k, 1)
      }
    }

    var sum: Int = 0
    var counter: Int = 0

    val detectedOutliersIntermediate = finalCount.map {
      case (point, count) =>
        sum += count
        counter += 1
        (point, count)
    }

    val detectedOutliers = detectedOutliersIntermediate.filter(x => x._2 <= treshold)
      .map {
        case (point, count) =>
          point
      }

    detectedOutliers.foreach(x => outliers.append(x))
    outliers
  }

  override def transformSchema(schema: StructType): StructType = {
    var transformed = schema
    if ($(rawPredictionCol).nonEmpty) {
      transformed = SchemaUtils.appendColumn(transformed, $(rawPredictionCol), new VectorUDT)
    }
    if ($(probabilityCol).nonEmpty) {
      transformed = SchemaUtils.appendColumn(transformed, $(probabilityCol), new VectorUDT)
    }
    if ($(predictionCol).nonEmpty) {
      transformed = SchemaUtils.appendColumn(transformed, $(predictionCol), DoubleType)
    }
    transformed
  }

  override def copy(extra: ParamMap): KNNOutlierModel = {
    val copied = new KNNOutlierModel(uid, k, treshold, numClasses, clusterCenters, clusterPointsRDD, clusterRadiusRDD, dataPoints)
    copyValues(copied, extra).setParent(parent)
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        var i = 0
        val size = dv.size

        var sum = 0.0
        while (i < size) {
          sum += dv.values(i)
          i += 1
        }

        i = 0
        while (i < size) {
          dv.values(i) /= sum
          i += 1
        }

        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in KNNClassificationModel:" +
          " raw2probabilitiesInPlace encountered SparseVector")
    }
  }

  override protected def predictRaw(features: Vector): Vector = {
    throw new SparkException("predictRaw function should not be called directly since kNN prediction is done in distributed fashion. Use transform instead.")
  }

}
