package algebra

import maths.Ring

trait MatrixProduct[Left, Right, Res] {
  def apply(l: Left, r: Right): Res
}

object MatrixProduct {
  implicit def matrixLikeProduct[V](implicit ring: Ring[V]):
    MatrixProduct[MatrixLike[V], MatrixLike[V], MatrixLike[V]] =
  {
    new MatrixProduct[MatrixLike[V], MatrixLike[V], MatrixLike[V]] {
      override def apply(l: MatrixLike[V], r: MatrixLike[V]): MatrixLike[V] = {
        assert(l.colSize == r.rowSize)

        val res = IndexedSeq.newBuilder[V]
        var i = 0

        while (i < r.colSize) {
          var j = 0

          while(j < l.rowSize) {
            var k = 0
            var tmp = ring.zero

            while(k < l.colSize) {
              tmp = ring.plus(tmp, ring.times(l(j, k), r(k, i)))
              k += 1
            }

            res += tmp
            j += 1
          }
          i += 1
        }

        Matrix(l.rowSize, r.colSize, res.result())
      }
    }
  }


  implicit def matrixScalarProduct[V](implicit ring: Ring[V]): MatrixProduct[MatrixLike[V], V, MatrixLike[V]] = {
    new MatrixProduct[MatrixLike[V], V, MatrixLike[V]] {
      override def apply(l: MatrixLike[V], r: V): MatrixLike[V] = l.dup(l.values.map(v => ring.times(v, r)))
    }
  }
}
