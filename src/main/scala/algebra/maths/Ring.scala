package algebra.maths

/**
 * A ring is a Group with left associativity multiplication
 */
trait Ring[@specialized(Int, Double) T] extends Group[T] {
  def one: T // Multiplicative identity
  def times(l: T, r: T): T
}

object IntRing extends Ring[Int] {
  override def zero: Int = 0
  override def one: Int = 1
  override def negate(v: Int): Int = -v
  override def plus(l: Int, r: Int): Int = l + r
  override def minus(l: Int, r: Int): Int = l - r
  override def times(l: Int, r: Int): Int = l * r
}

object Ring {
  implicit val intRing: Ring[Int] = IntRing
  implicit val DoubleRing: Ring[Double] = DoubleField
}
