package algebra.maths

/**
 * A monoid is a semi-group with an additive identity, i.e. (a+0) = (0+a) = a
 */
trait Monoid[@specialized(Int, Double) T] extends SemiGroup[T] {
  def zero: T
}

object Monoid {
  @inline def apply[F](implicit F: Monoid[F]): Monoid[F] = F

  implicit val intMonoid: Monoid[Int] = IntRing
  implicit val doubleMonoid: Monoid[Double] = DoubleField
}
