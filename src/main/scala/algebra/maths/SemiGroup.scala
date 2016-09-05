package algebra.maths

/**
 * A class with an associative plus operation
 */
trait SemiGroup[@specialized(Int, Double) T] extends Serializable {
  def plus(l: T, r: T): T
}

object SemiGroup {
  implicit val intSemiGroup: SemiGroup[Int] = IntRing
  implicit val doubleSemiGroup: SemiGroup[Double] = DoubleField
}
