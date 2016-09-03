package neuralnetwork

import machinelearning.{ColVector, Matrix}
import org.scalatest.WordSpec

class CostTest extends WordSpec {
  val layer = new Layer(2, 2)
  layer.weightsMatrix = Matrix(2, 3, IndexedSeq(0.0, 0.0, 0.1, 0.4, 0.8, 0.6))

  val outputLayer = new OutputLayer(1, 2)
  outputLayer.weightsMatrix = Matrix(1, 3, IndexedSeq(0.0, 0.3, 0.9))

  val network = new Network(2, 1, 1, 2) {
    override lazy val layers: IndexedSeq[Layer] = IndexedSeq(layer, outputLayer)
  }

  val inputs = new ColVector(IndexedSeq(0.35, 0.9))

  val input = IndexedSeq(
    (inputs.values, IndexedSeq(0.5))
  )

  def truncateAt(n: Double, p: Int): Double = { val s = math pow (10, p); (math floor n * s) / s }

  "A neural network" should {
    "compute final value correctly" in {
      val res = network.compute(inputs)
      assertResult(0.69)(truncateAt(res.values.head, 2))
    }
  }
}
