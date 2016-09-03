package neuralnetwork

import machinelearning.{ColVector, Matrix, MatrixOperation}
import org.scalatest.WordSpec

class ComputeTest extends WordSpec {
  val layer = new Layer(3, 2)
  layer.theta = Matrix(3, 3, IndexedSeq(0.3, 0.8, 0.1, 0.4, 0.3, 0.6, 0.9, 0.2, 0.4))

  val outputLayer = new OutputLayer(1, 3)
  outputLayer.theta = Matrix(1, 4, IndexedSeq(0.3, 0.1, 0.8, 0.1))

  val network = new Network(2, 1, 1, 3) {
    override lazy val layers: IndexedSeq[Layer] = IndexedSeq(layer, outputLayer)
  }

  "A neural network" should {
    "computes final value correctly" in {
      val inputs = new ColVector(IndexedSeq(1.0, 2.0))
      val result = network.compute(inputs)
      assert(result.dimension == 1)
      assertResult(1.2635498738785034)(result.values.head)
    }

    "keeps activation value inside layer after computation" in {
      val inputs = new ColVector(IndexedSeq(1.0, 2.0))
      network.compute(inputs)
      assert(outputLayer.activationValues.isDefined)
    }
  }
}
