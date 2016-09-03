package machinelearning

import scala.language.implicitConversions
import neuralnetwork.{Layer, Network}

object Main {
  def main (args: Array[String]) {
    val layer = new Layer(2, 2)
    layer.weightsMatrix = Matrix(2, 3, IndexedSeq(0.0, 0.0, 0.1, 0.4, 0.8, 0.6))

    val outputLayer = new Layer(1, 2)
    outputLayer.weightsMatrix = Matrix(1, 3, IndexedSeq(0.0, 0.3, 0.9))

    val network = new Network(2, 1, 1, 2) {
      override lazy val layers: IndexedSeq[Layer] = IndexedSeq(layer, outputLayer)
    }

    val inputs = new ColVector(IndexedSeq(0.35, 0.9))

    val input = IndexedSeq(
      (inputs.values, IndexedSeq(0.5))
    )

    println(f"value: ${network.compute(inputs).values.head}%1.2f")
    println(f"cost: ${network.cost(input)}%1.3f")

    var i = 15
    while (i > 0) {
      network.train(input)
      println()
      println(f"value: ${network.compute(inputs).values.head}%1.2f")
      println(f"cost: ${network.cost(input)}%1.3f")
      i -= 1
    }
  }
}
