package machinelearning

import scala.language.implicitConversions
import algebra.RichIndexedSeq
import neuralnetwork.Network
import neuralnetwork.Network.Input

import scala.collection.mutable
import scala.io.Source
import scala.util.Random

object Main {
  def main (args: Array[String]) {
//    val m = new algebra.MVector[Int](IndexedSeq(1)) + IndexedSeq(1).reshape(1, 1)
//    println(m.values)
//
//    val n = m.dot(2)

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

//    val network = new Network(1, 1, 1, 3)
//    val input: Input = IndexedSeq((IndexedSeq(1.0), IndexedSeq(2.0)))
    //println(network.cost(input))
    //network.train(input)

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
      val newSeq = seq.map(v => v - mean)
      val std = newSeq.map(v => v * v).sum / newSeq.size.toDouble
      a = a.updated(i, newSeq.map(v => v / std))
    }

    val aT = a.transpose
    val rT = input3.map(_._2)

    val Ifull = aT.flatten.reshape(7, aT.size)
    val Ofull = rT.flatten.reshape(1, rT.size)

    val inpt = aT.zip(rT).grouped(40).toVector.map(_.unzip).map { case (a, b) =>
      (a.flatten.reshape(7, a.size), b.flatten.reshape(1, a.size))
    }

    val network = new Network(List(7, 15, 1), 1, 0.0)

    var i = 0
    while (i < 10000) {
      if (i == 1000)
        network.learningRate = 0.1
      if (i == 2500)
        network.learningRate = 0.01
      if (i == 9000)
        network.learningRate = 0.001
      inpt.foreach { case(x, y) => network.train(x, y, 251) }
      i += 1
    }

    var hit = 0
    inpt.foreach { case(x, y) =>
      val res = network.compute(x)
      (res.values, y.values).zipped.foreach { case (xx, yy) =>
        if (Math.round(xx) == yy.toLong)
          hit += 1
      }
    }

    println(f"Accuracy on training set: ${(hit.toDouble / 251) * 100}%1.2f %%")


//
//    val normalizedInput: Network.Input = (a.transpose, input3).zipped.map((a, b) => (a, b._2))
//
//    println(s"Training set size: ${normalizedInput.size}")
//
//    //val network = new Network(7, 1, 1, 30)
//
//    var cost = network.cost(normalizedInput)
//    println(s"cost: ${network.cost(normalizedInput)}")
//
//    var i = 0
//    while (i < 10000) {
//      i += 1
//      if (i % 1000 == 0) {
//        cost = network.cost(normalizedInput)
//        println(s"cost: ${network.cost(normalizedInput)}")
//      }
//      network.train(normalizedInput)
//    }
//
//    println(s"Trained for $i epochs")
//    cost = network.cost(normalizedInput)
//    println(s"final cost: ${network.cost(normalizedInput)}")
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
//
//    val source2 = Source.fromFile("/Users/ccocchi/code/machine_learning/data/test.csv")
//
//    val input2 = source2.getLines().map { s =>
//      val array = s.split(',')
//      val result = IndexedSeq(array.head.toDouble)
//      val inputs = array.slice(1, array.length).toIndexedSeq.map(_.toDouble)
//      (inputs, result)
//    }.toIndexedSeq
//
//
//    var a2 = input2.map(c => c._1).transpose
//    for(i <- 2 to 6) {
//      val seq = a2(i)
//      val max = seq.max
//      val min = seq.min
//      val mean = seq.sum / seq.size.toDouble
//      val newSeq = seq.map(v => (v - mean)/ (max - min))
//      a2 = a2.updated(i, newSeq)
//    }
//
//    val normalizedInput2: Network.Input = (a2.transpose, input2).zipped.map((a, b) => (a, b._2))
//
//    println(s"Testing set size: ${normalizedInput2.size}")
//
//    miss = 0
//    hit = 0
//    normalizedInput2.foreach { case (inputs, result) =>
//      val v = Math.round(network.compute(new ColVector(inputs)).values.head)
//      if (v == result.head)
//        hit += 1
//      else
//        miss += 1
//    }
//
//    println(s"Accuracy on testing set: ${(hit.toDouble / input2.size.toDouble) * 100}%")

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

//  val labels = IndexedSeq("MIT", "NUC", "CYT", "ME1", "ME2", "ME3", "EXC", "VAC", "POX", "ERL")
//
//  def sourceToInput(s: Source): Input = {
//    s.getLines().map { s =>
//      val array   = s.split(',')
//      val inputs  = array.take(8).toIndexedSeq.map(_.toDouble)
//      val label   = array.last
//      val result  = IndexedSeq.fill(10)(0.0)
//      (inputs, result.updated(labels.indexOf(label.trim), 1.0))
//    }.toIndexedSeq
//  }
//
//  def normalize(input: Input): Input = {
//    var a = input.map(c => c._1).transpose
//    for(i <- a.indices) {
//      val seq = a(i)
//      val mean = seq.sum / seq.size.toDouble
//      val newSeq = seq.map(v => v - mean)
//      val std = newSeq.map(v => v * v).sum / newSeq.size.toDouble
//      a = a.updated(i, newSeq.map(v => v / std))
//    }
//
//    Random.shuffle((a.transpose, input).zipped.map((a, b) => (a, b._2)))
//  }
//
  def intercalate[V](a : List[V], b : List[V]): List[V] = a match {
    case first :: rest => first :: intercalate(b, rest)
    case _             => b
  }
//
//  val network = new Network(8, 10, 1, 100, 1, 0.1)
//  val ysource = Source.fromFile("/Users/ccocchi/code/machine_learning/data/yeast.dat")
//  val input   = sourceToInput(ysource)
//  val ninput  = normalize(input)
//
//  val validationData = ninput.take(400)
//  val trainingData = ninput.drop(400)
//
//  println(s"cost: ${network.cost(trainingData)}")
//  val batches = trainingData.grouped(100).toSeq
//
//  for (i <- 1 to 900) {
//    if (i == 450)
//      network.learningRate = 0.1
//    if (i == 800)
//      network.learningRate = 0.01
//
//    batches.foreach(b => network.train(b, trainingData.size))
//  }
//  println(s"cost: ${network.cost(trainingData)}")
//
//  var hit = 0
//  trainingData.foreach { case (inputs, result) =>
//    val r = network.compute(new ColVector[Double](inputs))
//    if (r.values.zipWithIndex.maxBy(_._1)._2 == result.zipWithIndex.maxBy(_._1)._2)
//      hit += 1
//  }
//
//  println(f"Accuracy on training set: ${(hit.toDouble / trainingData.size.toDouble) * 100}%1.2f %%")
//
//  hit = 0
//  validationData.foreach { case (inputs, result) =>
//    val r = network.compute(new ColVector[Double](inputs))
//    if (r.values.zipWithIndex.maxBy(_._1)._2 == result.zipWithIndex.maxBy(_._1)._2)
//      hit += 1
//  }
//
//  println(f"Accuracy on validation set: ${(hit.toDouble / validationData.size.toDouble) * 100}%1.2f %%")

}
