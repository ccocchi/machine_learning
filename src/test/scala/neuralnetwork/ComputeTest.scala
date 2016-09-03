package neuralnetwork

import machinelearning.{ColVector, Matrix, MatrixOperation}
import org.scalatest.WordSpec

class ComputeTest extends WordSpec {
  val layer = new Layer(3, 2)
  layer.weightsMatrix = Matrix(3, 3, IndexedSeq(0.3, 0.8, 0.1, 0.4, 0.3, 0.6, 0.9, 0.2, 0.4))

  val outputLayer = new Layer(1, 3)
  outputLayer.weightsMatrix = Matrix(1, 4, IndexedSeq(0.3, 0.1, 0.8, 0.1))

  val network = new Network(2, 1, 1, 3) {
    override lazy val layers: IndexedSeq[Layer] = IndexedSeq(layer, outputLayer)
  }

  val inputs = new ColVector(IndexedSeq(1.0, 2.0))

  "A neural network" should {
    "computes final value correctly" in {
      val result = network.compute(inputs)
      assert(result.dimension == 1)
      assertResult(0.7555123275747737)(result.values.head)
    }

    "keeps activation value inside layer after computation" in {
      network.compute(inputs)
      assert(outputLayer.activationValues.isDefined)
    }

    "works with random weights" in {
      val network = new Network(2, 1, 1, 3)
      val result = network.compute(inputs)
      assert(result.dimension == 1)
      assert(result.values.head.isInstanceOf[Double])
    }

    "works with multiple layers" in {
      val otherLayer = new Layer(3, 3)
      otherLayer.weightsMatrix = Matrix(3, 4, IndexedSeq(0.3, 0.8, 0.1, 0.4, 0.3, 0.6, 0.9, 0.2, 0.4, 0.7, 0.7, 0.3))
      val multiLayerNetwork = new Network(2, 1, 2, 3) {
        override lazy val layers: IndexedSeq[Layer] = IndexedSeq(layer, otherLayer, outputLayer)
      }
      val result = multiLayerNetwork.compute(inputs)
      assert(result.dimension == 1)
      assertResult(0.7600502385068484)(result.values.head)
    }
  }
}
