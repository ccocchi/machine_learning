package machinelearning

import scala.language.implicitConversions
import neuralnetwork.{Layer, Network}

import scala.io.Source

object Main {
  def main (args: Array[String]) {
    val source = Source.fromFile("/Users/ccocchi/code/machine_learning/data/data.csv")

    val input: Network.Input = source.getLines().map { s =>
      val array = s.split(',')
      array.update(3, (array(3).toDouble / 1000).toString)
      val result = IndexedSeq(array.head.toDouble)
      val inputs = array.slice(1, array.length).toIndexedSeq.map(_.toDouble)
      (inputs, result)
    }.toIndexedSeq

    println(s"Training set size: ${input.size}")

    val network = new Network(7, 1, 1, 10)

    var d = 1.0
    var i = 0
    while (d > 0.05) {
      i += 1
      d = network.train(input)
    }

    println(s"Trained in $i iterations")

    var miss = 0
    var hit  = 0

    input.foreach { case (inputs, result) =>
      val v = Math.round(network.compute(new ColVector(inputs)).values.head)
      if (v == result.head)
        hit += 1
      else
        miss += 1
    }

    println(hit)
    println(miss)

    println(s"Accuracy on training set: ${(hit.toDouble / input.size.toDouble) * 100}%")


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
}
