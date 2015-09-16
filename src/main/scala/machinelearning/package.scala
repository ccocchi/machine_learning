import scala.language.implicitConversions

package object machinelearning {
  implicit def toScalar(i: Int): Scalar[Int] = new Scalar(i)
}
