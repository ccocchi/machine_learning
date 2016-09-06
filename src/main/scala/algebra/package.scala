import scala.language.implicitConversions

package object algebra {
  implicit class RichIndexedSeq[V](seq: IndexedSeq[V]) {
    def reshape(r: Int, c: Int): MatrixLike[V] = {
      assert(seq.size == r * c, s"${seq.size} seq cannot be shaped into ${r}x${c} matrix")
      MatrixLike.apply(r, c, seq)
    }
    def toMVector: MVector[V] = MVector(seq)
  }
}
