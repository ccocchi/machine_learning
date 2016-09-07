package machinelearning

import java.io.{DataInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

import scala.language.implicitConversions
import algebra.{MatrixLike, RichIndexedSeq}
import neuralnetwork.Network
import neuralnetwork.Network.Input
import utils.{MNISTImageLoader, MNISTLabelLoader}


import scala.io.Source
import scala.util.Random

object Main {
  def main (args: Array[String]): Unit = {

//    val stream = new DataInputStream(new GZIPInputStream(new FileInputStream("data/train-images-idx3-ubyte.gz")))
//    val l1 = new Array[Byte](28 * 28 * 60000 + 16)
//    stream.readFully(l1)
//
//    val stream2 = new DataInputStream(new GZIPInputStream(new FileInputStream("data/train-labels-idx1-ubyte.gz")))
//    val l2 = new Array[Byte](60000 + 8)
//    stream2.readFully(l2)
//
//    val normalizedL1 = l1.drop(16).grouped(784 * 10).map(vs => vs.map(v => v / 255.0).toIndexedSeq).toSeq
//    val normalizedL2 = l2.drop(8).flatMap { v =>
//      val a = Array.fill(10)(0.0)
//      a(v) = 1.0
//      a
//    }.toIndexedSeq.grouped(10 * 10).toSeq
//
//    val network = new Network(List(784, 30, 10), 3.0, 0.0)
//
//    println(s"training set size: ${normalizedL1.size}")
//
//    for(epoch <- 1 to 1) {
//      (normalizedL1, normalizedL2).zipped.foreach { case (x, y) =>
//        network.train(x.reshape(784, 10), y.reshape(10, 10), 60000)
//      }
//      println(s"Epoch $epoch done")
//    }

    val network = new Network(List(784, 30, 10), 3.0, 0.0)

    val l1 = new MNISTImageLoader("data/train-images-idx3-ubyte.gz")
    val l2 = new MNISTLabelLoader("data/train-labels-idx1-ubyte.gz")
    for (epoch <- 1 to 2) {
      for (i <- 1 to 6000) {
        val x = l1.images.take(10).toIndexedSeq.flatten.map(v => v /255.0).reshape(784, 10)
        val y = l2.labels.take(10).toIndexedSeq.flatten.reshape(10, 10)
        network.train(x, y, 60000)
      }
      l1.rewind()
      l2.rewind()
      println(s"Epoch $epoch done")
    }

    var hit = 0
    (l1.images, l2.rawLabels).zipped.foreach { case(x, y) =>
      val res = network.compute(x.map(v => v /255.0).reshape(784, 1))
      val i = res.values.zipWithIndex.maxBy(_._1)._2

      if (i == y)
        hit += 1
    }
    println(f"* training accuracy: ${(hit.toDouble / 60000) * 100}%1.2f %%")

    val l1Test = new MNISTImageLoader("data/t10k-images-idx3-ubyte.gz")
    val l2Test = new MNISTLabelLoader("data/t10k-labels-idx1-ubyte.gz")

    hit = 0
    (l1Test.images, l2Test.rawLabels).zipped.foreach { case(x, y) =>
      val res = network.compute(x.map(v => v /255.0).reshape(784, 1))
      val i = res.values.zipWithIndex.maxBy(_._1)._2

      if (i == y)
        hit += 1
    }
    println(f"* test accuracy: ${(hit.toDouble / 10000) * 100}%1.2f %%")


//
//    loader.images.take(2).toIndexedSeq.flatten.reshape(784, 2)


//    val network = new Network(List(1, 3, 1), 0.1, 0.0)
//    network.layers.head.weightsMatrix = IndexedSeq(1.0, 1.0, 1.0).reshape(3, 1)
//    network.layers.last.weightsMatrix = IndexedSeq(1.0, 1.0, 1.0).reshape(1, 3)
//
//
//    val input   = IndexedSeq(1.0, 3.0).reshape(1,2)
//    val output  = IndexedSeq(2.0, 7.0).reshape(1,2)
//
//
//    println(network.cost(input, output))
//    network.train(input, output, 2)
//    println(network.cost(input, output))
//
//    println(network.layers.map(_.biasVector.values))


    //network.train(IndexedSeq(1.0, 3.0).reshape(1,2), IndexedSeq(2.0, -7.0).reshape(1,2), 2)


    //println(network.cost(IndexedSeq(1.0, 3.0).reshape(1,2), IndexedSeq(2.0, -7.0).reshape(1,2)))

    //println(network.layers.last.weightsMatrix.values  )


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

//    val source = Source.fromFile("/Users/ccocchi/code/machine_learning/data/data.csv")
//
//    val input = source.getLines().map { s =>
//      val array = s.split(',')
//      val result = IndexedSeq(array.head.toDouble)
//      val inputs = array.slice(1, array.length).toIndexedSeq.map(_.toDouble)
//      (inputs, result)
//    }.toIndexedSeq
//
//    val (sf, nyc) = input.partition(i => i._2.head == 0.0)
//    val input3 = intercalate(sf.toList, nyc.toList).toIndexedSeq
//
//    var a = input3.map(c => c._1).transpose
//    for(i <- 2 to 6) {
//      val seq = a(i)
//      val max = seq.max
//      val min = seq.min
//      val mean = seq.sum / seq.size.toDouble
//      val newSeq = seq.map(v => v - mean)
//      val std = newSeq.map(v => v * v).sum / newSeq.size.toDouble
//      a = a.updated(i, newSeq.map(v => v / std))
//    }
//
//    val aT = a.transpose
//    val rT = input3.map(_._2)
//
//    val Ifull = aT.flatten.reshape(7, aT.size)
//    val Ofull = rT.flatten.reshape(1, rT.size)
//
//    val inpt = aT.zip(rT).grouped(40).toVector.map(_.unzip).map { case (a, b) =>
//      (a.flatten.reshape(7, a.size), b.flatten.reshape(1, a.size))
//    }
//
//    val network = new Network(List(7, 15, 1), 1, 0.0)
//
//    var i = 0
//    while (i < 10000) {
//      if (i == 1000)
//        network.learningRate = 0.1
//      if (i == 2500)
//        network.learningRate = 0.01
//      if (i == 9000)
//        network.learningRate = 0.001
//      inpt.foreach { case(x, y) => network.train(x, y, 251) }
//      i += 1
//    }
//
//    var hit = 0
//    inpt.foreach { case(x, y) =>
//      val res = network.compute(x)
//      (res.values, y.values).zipped.foreach { case (xx, yy) =>
//        if (Math.round(xx) == yy.toLong)
//          hit += 1
//      }
//    }
//
//    println(f"Accuracy on training set: ${(hit.toDouble / 251) * 100}%1.2f %%")

//  def intercalate[V](a : List[V], b : List[V]): List[V] = a match {
//    case first :: rest => first :: intercalate(b, rest)
//    case _             => b
//  }

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

//    val labels = IndexedSeq("MIT", "NUC", "CYT", "ME1", "ME2", "ME3", "EXC", "VAC", "POX", "ERL")
//
//    def sourceToInput(s: Source) = {
//      s.getLines().map { s =>
//        val array   = s.split(',')
//        val inputs  = array.take(8).toIndexedSeq.map(_.toDouble)
//        val label   = array.last
//        val result  = IndexedSeq.fill(10)(0.0)
//        (inputs, result.updated(labels.indexOf(label.trim), 1.0))
//      }.toIndexedSeq
//    }
//
//    def normalize(input: Input) = {
//      var a = input.map(c => c._1).transpose
//      for(i <- a.indices) {
//        val seq = a(i)
//        val mean = seq.sum / seq.size.toDouble
//        val newSeq = seq.map(v => v - mean)
//        val std = newSeq.map(v => v * v).sum / newSeq.size.toDouble
//        a = a.updated(i, newSeq.map(v => v / std))
//      }
//
//      Random.shuffle((a.transpose, input).zipped.map((a, b) => (a, b._2)))
//    }
//
//    def seqToMat(seq: Seq[(IndexedSeq[Double], IndexedSeq[Double])]): (MatrixLike[Double], MatrixLike[Double]) = {
//      val res1 = IndexedSeq.newBuilder[Double]
//      val res2 = IndexedSeq.newBuilder[Double]
//
//      for(j <- 0 to 7)
//        for(i <- seq.indices) {
//          res1 += seq(i)._1(j)
//        }
//
//      for(j <- 0 to 9)
//        for(i <- seq.indices) {
//          res2 += seq(i)._2(j)
//        }
//
//      (res1.result().reshape(seq.size, 8), res2.result().reshape(seq.size, 10))
//    }
//
//    def batch(ms: (MatrixLike[Double], MatrixLike[Double]), batchSize: Int): Seq[(MatrixLike[Double], MatrixLike[Double])] = {
//      (ms._1.groupByColumns(batchSize), ms._2.groupByColumns(batchSize)).zipped.toList
//    }
//
//    val network = new Network(List(8, 100, 10), 1.0, 0.0)
//    val ysource = Source.fromFile("/Users/ccocchi/code/machine_learning/data/yeast.dat")
//    val input   = normalize(sourceToInput(ysource))
//
//
//    val data = seqToMat(input.drop(400))
//    val trainingData = (data._1.transpose, data._2.transpose)
//    val totalInputs = trainingData._1.colSize
//    val batches = batch(trainingData, 100)
//
////    val totalInputs = input.size - 400
////    val batches = input.drop(400).grouped(100).toSeq
//
//    network.compute(input.head._1.reshape(8, 1))
//
//    println("Starting training:")
//    println(s"* dataset size: $totalInputs")
//    println(s"* initial cost: ${network.cost(trainingData._1, trainingData._2)}")
//
//    for (i <- 1 to 900) {
//      if (i % 100 == 0)
//        println(s"* cost at epoch $i: ${network.cost(trainingData._1, trainingData._2)}")
//      if (i == 450)
//        network.learningRate = 0.1
//      batches.foreach(b => network.train(b._1, b._2, totalInputs))
//      //batches.foreach(b => network.train(b, totalInputs))
//      //network.train(IndexedSeq(input.head), 1)
//    }
//
//    println(s"* final cost: ${network.cost(trainingData._1, trainingData._2)}")
//
//    var hit = 0
//    input.drop(400).foreach { case (i, expected) =>
//      val res = network.compute(i.reshape(8, 1))
//      if (res.values.zipWithIndex.maxBy(_._1)._2 == expected.zipWithIndex.maxBy(_._1)._2)
//        hit += 1
//    }
//
//    println(f"* training accuracy: ${(hit.toDouble / totalInputs) * 100}%1.2f %%")
//
//    hit = 0
//    input.take(400).foreach { case (i, expected) =>
//      val res = network.compute(i.reshape(8, 1))
//      if (res.values.zipWithIndex.maxBy(_._1)._2 == expected.zipWithIndex.maxBy(_._1)._2)
//        hit += 1
//    }
//
//    println(f"* validation accuracy: ${(hit.toDouble / 400) * 100}%1.2f %%")
  }
}