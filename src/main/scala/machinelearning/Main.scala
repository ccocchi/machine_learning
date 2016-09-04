package machinelearning

import scala.language.implicitConversions
import neuralnetwork.Network
import neuralnetwork.Network.Input

import scala.collection.mutable
import scala.io.Source

object Main {
  def main (args: Array[String]) {
//    val matrix  = new Matrix[Double](4, 3, IndexedSeq(1, 3, -3, 0, 2, 7, 1, 3, -3, 0, 2, 7))
//    val mutMatrix  = new MutableMatrix[Double](4, 3, mutable.IndexedSeq(1, 3, -3, 0, 2, 7, 1, 3, -3, 0, 2, 7))
//    val added   = new Matrix[Double](4, 3, IndexedSeq(0.1, 0.3, -0.3, 0, 0.2, 0.7, 0.1, 0.3, -0.3, 0, 0.2, 0.7))
//
//    val start = System.currentTimeMillis()
//
//    for (i <- 0 to 1000000) {
//      matrix + added
//    }
//    val end = System.currentTimeMillis()
//    println(s"Time elaspsed for immutable: ${end - start}")
//
//    val start2 = System.currentTimeMillis()
//
//    for (i <- 0 to 1000000) {
//      mutMatrix + added
//    }
//    val end2 = System.currentTimeMillis()
//    println(s"Time elaspsed for mutable: ${end2 - start2}")


//    val source  = Source.fromFile("/Users/ccocchi/code/machine_learning/data/banana.dat")
//    val input   = sourceToInput(source)
//
//    val normalizedInput  = normalize(input)
//    println(s"Training set size: ${normalizedInput.size}")
//
//    val network = new Network(2, 1, 1, 3)
//
//    var d = 1.0
//    var i = 0
//    while (i < 2000) {
//      i += 1
//      if (i % 100 == 0)
//        println(s"cost: ${network.cost(normalizedInput)}")
//      d = network.train(normalizedInput)
//    }
//
//    println(s"Trained in $i iterations")
//
////    println(network.compute(new ColVector(input.head._1)).values)
////    println(input.head._2)
//
//    var miss = 0
//    var hit  = 0
//
//    normalizedInput.foreach { case (inputs, result) =>
//      val v = Math.round(network.compute(new ColVector(inputs)).values.head)
//      if (v == result.head)
//        hit += 1
//      else
//        miss += 1
//    }
//
//    println(s"Accuracy on training set: ${(hit.toDouble / input.size.toDouble) * 100}%")


    val source = Source.fromFile("/Users/ccocchi/code/machine_learning/data/data.csv")

    val input = source.getLines().map { s =>
      val array = s.split(',')
      val result = IndexedSeq(array.head.toDouble)
      val inputs = array.slice(1, array.length).toIndexedSeq.map(_.toDouble)
      (inputs, result)
    }.toIndexedSeq

    val (sf, nyc) = input.partition(i => i._2.head == 0.0)
    val input3 = intercalate(sf.toList, nyc.toList).toIndexedSeq

    var a = input3.map(c => c._1).transpose
    for(i <- 2 to 6) {
      val seq = a(i)
      val max = seq.max
      val min = seq.min
      val mean = seq.sum / seq.size.toDouble
      val newSeq = seq.map(v => (v - mean)/ (max - min))
      a = a.updated(i, newSeq)
    }

    val normalizedInput: Network.Input = (a.transpose, input3).zipped.map((a, b) => (a, b._2))

    println(s"Training set size: ${normalizedInput.size}")

    val network = new Network(7, 1, 2, 5)

    var d = 1.0
    var i = 0
    while (d > 0.008) {
      i += 1
      if (i > 70000)
        throw new Exception("too long")
      d = network.train(normalizedInput)
    }

    println(s"Trained in $i iterations")

    var miss = 0
    var hit  = 0

    normalizedInput.foreach { case (inputs, result) =>
      val v = Math.round(network.compute(new ColVector(inputs)).values.head)
      if (v == result.head)
        hit += 1
      else
        miss += 1
    }

    println(s"Accuracy on training set: ${(hit.toDouble / input.size.toDouble) * 100}%")

    val source2 = Source.fromFile("/Users/ccocchi/code/machine_learning/data/test.csv")

    val input2 = source2.getLines().map { s =>
      val array = s.split(',')
      val result = IndexedSeq(array.head.toDouble)
      val inputs = array.slice(1, array.length).toIndexedSeq.map(_.toDouble)
      (inputs, result)
    }.toIndexedSeq


    var a2 = input2.map(c => c._1).transpose
    for(i <- 2 to 6) {
      val seq = a2(i)
      val max = seq.max
      val min = seq.min
      val mean = seq.sum / seq.size.toDouble
      val newSeq = seq.map(v => (v - mean)/ (max - min))
      a2 = a2.updated(i, newSeq)
    }

    val normalizedInput2: Network.Input = (a2.transpose, input2).zipped.map((a, b) => (a, b._2))

    println(s"Testing set size: ${normalizedInput2.size}")

    miss = 0
    hit = 0
    normalizedInput2.foreach { case (inputs, result) =>
      val v = Math.round(network.compute(new ColVector(inputs)).values.head)
      if (v == result.head)
        hit += 1
      else
        miss += 1
    }

    println(s"Accuracy on testing set: ${(hit.toDouble / input2.size.toDouble) * 100}%")

    //    val layer = new Layer(2, 2)
    //    layer.weightsMatrix = Matrix(2, 3, IndexedSeq(0.0, 0.0, 0.1, 0.4, 0.8, 0.6))
    //
    //    val outputLayer = new Layer(1, 2)
    //    outputLayer.weightsMatrix = Matrix(1, 3, IndexedSeq(0.0, 0.3, 0.9))
    //
    //    val network = new Network(2, 1, 1, 2) {
    //      override lazy val layers: IndexedSeq[Layer] = IndexedSeq(layer, outputLayer)
    //    }
    //
    //    val inputs = new ColVector(IndexedSeq(0.35, 0.9))
    //
    //    val input = IndexedSeq(
    //      (inputs.values, IndexedSeq(0.5))
    //    )
    //
    //    var d = 1.0
    //    var i = 0
    //    while (d > 0.05) {
    //      i += 1
    //      d = network.train(input)
    //    }
    //
    //    println(s"Trained in $i iterations")
    //    println(s"value = ${network.compute(inputs).values.head}")
    //  }
  }

  def sourceToInput(s: Source): Input = {
    s.getLines().map { s =>
      val array   = s.split(',')
      val inputs  = array.take(2).toIndexedSeq.map(_.toDouble)
      val result  = IndexedSeq((array.last.toDouble + 1.0) / 2.0)
      (inputs, result)
    }.toIndexedSeq
  }

  def normalize(input: Input): Input = {
    val (pos, neg) = input.partition(i => i._2.head == 0.0)
    val input3 = intercalate(pos.toList, neg.toList).toIndexedSeq

//    var a = input3.map(c => c._1).transpose
//    for(i <- 0 to 1) {
//      val seq = a(i)
//      val max = seq.max
//      val min = seq.min
//      val mean = seq.sum / seq.size.toDouble
//      val newSeq = seq.map(v => (v - mean)/ (max - min))
//      a = a.updated(i, newSeq)
//    }
//
//    (a.transpose, input3).zipped.map((a, b) => (a, b._2))

    input3
  }

  def intercalate[V](a : List[V], b : List[V]): List[V] = a match {
    case first :: rest => first :: intercalate(b, rest)
    case _             => b
  }
}
