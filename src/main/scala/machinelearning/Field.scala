package machinelearning

/**
 * A ring is a Group with left associativity multiplication
 */
trait Field[@specialized(Double) T] extends Ring[T] {
  def div(l: T, r: T): T
}

object DoubleField extends Field[Double] {
  override def zero: Double = 0
  override def one: Double = 1
  override def negate(v: Double): Double = -v
  override def plus(l: Double, r: Double): Double = l + r
  override def minus(l: Double, r: Double): Double = l - r
  override def times(l: Double, r: Double): Double = l * r
  override def div(l: Double, r: Double): Double = l / r
}

object Field {
  implicit val doubleRing: Field[Double] = DoubleField
}
