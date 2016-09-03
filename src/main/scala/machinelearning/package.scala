import scala.language.implicitConversions

package object machinelearning {
  implicit def intToScalar(i: Int): Scalar[Int] = new Scalar(i)
  implicit def doubleToScalar(i: Double): Scalar[Double] = new Scalar(i)
}
