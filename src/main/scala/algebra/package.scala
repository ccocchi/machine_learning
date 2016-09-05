import scala.language.implicitConversions

package object algebra {
  implicit class RichIndexedSeq[V](seq: IndexedSeq[V]) {
    def reshape(r: Int, c: Int): MatrixLike[V] = MatrixLike.apply(r, c, seq)
  }
}
