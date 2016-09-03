package neuralnetwork

import machinelearning.ColVector
import neuralnetwork.Network.{Input, LayerWithResult}

object Network {
  type LayerWithResult = (Option[Layer], Option[ColVector[Double]])
  type Input = IndexedSeq[(IndexedSeq[Double], IndexedSeq[Double])]

  val regularizationParameter: Double = 0
}

/**
  * Created by ccocchi on 02/09/16.
  */
class Network( val inputSize: Int,
               val outputSize: Int,
               val hiddenLayers: Int,
               val unitsPerLayer: Int
             )
{
  lazy val layers = buildLayers()

//  def train(input: Input): Unit = {
//    input.foreach { case (inputs, result) => train(inputs, result) }
//  }
//
//  def train(x: IndexedSeq[Double], y: IndexedSeq[Double]): Unit = {
//    val inputs = new ColVector(x)
//    val result = compute(inputs)
//
//    layers.foldRight[LayerWithResult]((None, Some(result))) { case (l1, l2) => l1.computeError(l2); (Some(l1), None) } // maybe l2 here but I don't think so
//    layers.foldLeft(inputs) { case (vs, l) => l.computeDelta(inputs); l.currentValue.get }
//  }

  def compute(colVector: ColVector[Double]): ColVector[Double] = {
    layers.foldLeft(colVector) { (values, l) => l.compute(values) }
  }

  private def buildLayers(): IndexedSeq[Layer] = {
    val seq = IndexedSeq.newBuilder[Layer]
    seq += new Layer(unitsPerLayer, inputSize)

    if (hiddenLayers > 1)
      seq ++= IndexedSeq.fill(hiddenLayers - 1) { new Layer(unitsPerLayer, unitsPerLayer) }

    seq += new OutputLayer(outputSize, unitsPerLayer)
    seq.result()
  }
}
