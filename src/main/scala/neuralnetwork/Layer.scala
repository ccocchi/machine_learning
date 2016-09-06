package neuralnetwork

import algebra.{MVector, MatrixLike}

object Layer {
  final val epsilon = 0.001
}

class Layer(val size: Int, val inputsSize: Int) {
  var weightsMatrix: MatrixLike[Double] = MatrixLike.fill(size, inputsSize) {
    (Math.random() * (2 * Layer.epsilon) - Layer.epsilon) / Math.sqrt(inputsSize)
  }

  var biasVector: MatrixLike[Double] = MVector.fill(size)(1.0)

  final def compute(inputs: MatrixLike[Double]): MatrixLike[Double] = {
    weightsMatrix.dot(inputs).plus(biasVector).map(activationFunction)
  }

  def backprop(x: MatrixLike[Double], sigma: MatrixLike[Double]): MatrixLike[Double] = {
    val left  = weightsMatrix.transpose.dot(sigma)
    val ones  = MatrixLike.fill(x.rowSize, sigma.colSize)(1.0)
    val right = x * (ones - x)
    left * right
  }

  private def activationFunction(z: Double): Double = 1.0 / (1.0 + Math.exp(-z))
}