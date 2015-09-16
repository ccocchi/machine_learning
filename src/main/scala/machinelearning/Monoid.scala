package machinelearning

trait Monoid[T] extends SemiGroup[T] {
  def zero: T
}

object Monoid {
  @inline def apply[F](implicit F: Monoid[F]): Monoid[F] = F

  implicit object IntMonoid extends Monoid[Int] {
    override def zero: Int = 0
  }
}
