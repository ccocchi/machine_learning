package machinelearning

/**
 * A group is a monoid that also has substraction
 */
trait Group[@specialized(Int) T] extends Monoid[T] {
  def negate(v: T): T = minus(zero, v)
  def minus(l: T, r: T): T = plus(l, negate(r))
}

object Group {
  implicit val intGroup: Group[Int] = IntRing
}