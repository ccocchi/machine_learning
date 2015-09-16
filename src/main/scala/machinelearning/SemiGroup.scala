package machinelearning

trait SemiGroup[T] extends Serializable {
  def plus(l: T, r: T): T
}

object SemiGroup {
  implicit object IntSemiGroup extends SemiGroup[Int] {
    override def plus(l: Int, r: Int): Int = l + r
  }
}
