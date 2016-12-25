package neuralnetwork

import algebra.MatrixLike

sealed trait ActivationFunction {
  def f(z: Double): Double
  def derivative(x: MatrixLike[Double]): MatrixLike[Double]
}

object SigmoidFunction extends ActivationFunction {
  override def f(z: Double): Double = 1.0 / (1.0 + Math.exp(-z))

  override def derivative(x: MatrixLike[Double]): MatrixLike[Double] = {
    val ones  = MatrixLike.fill(x.rowSize, x.colSize)(1.0)
    x * (ones - x)
  }
}

object ReLUFunction extends ActivationFunction {
  override def f(z: Double): Double = Math.max(0.0, z)
  override def derivative(x: MatrixLike[Double]): MatrixLike[Double] = x.map(v => if (v <= 0) 0 else 1)
}
