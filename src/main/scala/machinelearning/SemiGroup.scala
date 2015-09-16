package machinelearning

/**
 * A class with an associative plus operation
 */
trait SemiGroup[@specialized(Int) T] extends Serializable {
  def plus(l: T, r: T): T
}

object SemiGroup {
  implicit val intSemiGroup: SemiGroup[Int] = IntRing
}
