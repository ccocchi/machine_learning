package neuralnetwork

import algebra.{MVector, MatrixLike}
import algebra.MatrixOperation.print
import scala.util.Random

object Layer {
  final val p = 0.7
}

class Layer(val size: Int, val inputsSize: Int) {
  var weightsMatrix: MatrixLike[Double] = MatrixLike.fill(size, inputsSize) {
    Random.nextGaussian() / Math.sqrt(inputsSize)
  }

  var biasVector: MatrixLike[Double] = MVector.fill(size)(0.0)

  final def compute(inputs: MatrixLike[Double]): MatrixLike[Double] =
    weightsMatrix.dot(inputs).plus(biasVector).map(activationFunction)

  def computeWithDropout(inputs: MatrixLike[Double]): MatrixLike[Double] = {
    val mask = MatrixLike.fill(inputs.rowSize, inputs.colSize) {
      val r = Random.nextDouble()
      if (r < Layer.p)
        1.0
      else
        0.0
    }

    val tmp = inputs * mask * (1 / Layer.p)
    print(mask)
    println()
    print(tmp)

//    println(mask.values.count(v => v == 0.0))
//    println(tmp.values.count(v => v == 0.0))

    val res = weightsMatrix.dot(tmp).plus(biasVector).map(activationFunction)
    println()
    print(res)
    res
  }

  def backprop(x: MatrixLike[Double], sigma: MatrixLike[Double]): MatrixLike[Double] = {
    val left  = weightsMatrix.transpose.dot(sigma)
    val ones  = MatrixLike.fill(x.rowSize, sigma.colSize)(1.0)
    val right = x * (ones - x)
    left * right
  }

  private def activationFunction(z: Double): Double = 1.0 / (1.0 + Math.exp(-z))
}

class OutputLayer(size: Int, inputsSize: Int) extends Layer(size, inputsSize) {
  override def computeWithDropout(inputs: MatrixLike[Double]): MatrixLike[Double] = compute(inputs)
}