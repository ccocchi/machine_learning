package machinelearning

trait MatrixProduct[Left, Right, Res] {
  def apply(l: Left, r: Right): Res
}

object MatrixProduct {
  implicit def scalarRightProduct[V](implicit ring: Ring[V]): MatrixProduct[Matrix[V], Scalar[V], Matrix[V]] =
    new MatrixProduct[Matrix[V], Scalar[V], Matrix[V]] {
      override def apply(l: Matrix[V], r: Scalar[V]): Matrix[V] = l.map(v => ring.times(v, r.value))
    }

  implicit def scalarLeftProduct[V](implicit ring: Ring[V]): MatrixProduct[Scalar[V], Matrix[V], Matrix[V]] =
    new MatrixProduct[Scalar[V], Matrix[V], Matrix[V]] {
      override def apply(l: Scalar[V], r: Matrix[V]): Matrix[V] = r.map(v => ring.times(v, l.value))
    }

  implicit def colVectorRightProduct[V](implicit ring: Ring[V]): MatrixProduct[RowVector[V], ColVector[V], V] =
    new MatrixProduct[RowVector[V], ColVector[V], V] {
      override def apply(l: RowVector[V], r: ColVector[V]): V = (l.values, r.values).zipped.foldLeft(ring.zero) {
        case (acc, (lv, rv)) => ring.plus(acc, ring.times(lv, rv))
      }
    }

  implicit def matrixProduct[V](implicit ring: Ring[V]): MatrixProduct[Matrix[V], Matrix[V], Matrix[V]] =
    new MatrixProduct[Matrix[V], Matrix[V], Matrix[V]] {
      override def apply(l: Matrix[V], r: Matrix[V]): Matrix[V] = {
        val res = IndexedSeq.newBuilder[V]

        var i = 0
        while (i < r.cols) {
          var j = 0

          while(j < l.rows) {
            var k = 0
            var tmp = ring.zero

            while(k < l.cols) {
              tmp = ring.plus(tmp, ring.times(l.getValue((k, j)), r.getValue((i, k))))
              k += 1
            }

            res += tmp
            j += 1
          }
          i += 1
        }

        Matrix(l.rows, r.cols, res.result())
      }
    }

  implicit def matrixWithColVectorProduct[V](implicit ring: Ring[V]): MatrixProduct[Matrix[V], ColVector[V], Matrix[V]] =
    new MatrixProduct[Matrix[V], ColVector[V], Matrix[V]] {
      override def apply(l: Matrix[V], r: ColVector[V]): Matrix[V] = {
        val res = IndexedSeq.newBuilder[V]

        var i = 0
        while(i < l.rows) {
          res += l.row(i) * r
          i += 1
        }

        Matrix(l.rows, 1, res.result())
      }
    }
}
