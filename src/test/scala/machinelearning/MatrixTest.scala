package machinelearning

import org.scalatest.WordSpec

class MatrixTest extends WordSpec {
  /*
    [ 1   0
      3   2
      -3  7 ]
   */
  val matrix = new Matrix(3, 2, IndexedSeq(1, 3, -3, 0, 2, 7))

  /*
    [ 2  -1
      0   4
      1   1 ]
   */
  val matrix2 = new Matrix(3, 2, IndexedSeq(2, 0, 1, -1, 4, 1))


  /*
    [ 1   3
      3   2 ]
   */
  val floatMatrix = new Matrix(2, 2, IndexedSeq(1.0, 3.0, 3.0, 2.0))

  "A matrix" can {
    "have a size" in {
      assert(matrix.size == 6)
    }

    "be accessed via elements position" in {
      assert(matrix.getValue((0, 0)) == 1)
      assert(matrix.getValue((1, 0)) == 0)
    }

    "be updated" in {
      val m = matrix.updated((0, 0), 7)

      assert(m.size == 6)
      assert(m.getValue((0, 0)) == 7)
    }

    "be additionned with another matrix" in {
      val m = matrix + matrix2
      assertResult(IndexedSeq(3, 3, -2, -1, 6, 8))(m.rowsByColumns)
    }

    "be multiplied by scalars" in {
      val m = 2 * matrix

      assert(m.size == 6)
      assert(m.getValue(1, 1) == 4)
    }

    "be multiplied by a vector" in {
      val vector = new ColVector[Int](IndexedSeq(1, 0))
      val m = matrix * vector

      assertResult(IndexedSeq(1, 3, -3))(m.rowsByColumns)
    }

    "be multiplied by a matrix" in {
      val m   = Matrix(2, 2, IndexedSeq(1, 0, 1, 2))
      val res = matrix * m
      assertResult(IndexedSeq(1, 3, -3, 1, 7, 11))(res.rowsByColumns)
    }

    "be transposed" in {
      val m = matrix.transpose

      assertResult(2)(m.rows)
      assertResult(3)(m.cols)
      assertResult(IndexedSeq(1, 0, 3, 2, -3, 7))(m.rowsByColumns)
    }

    "swap rows" in {
      val m = matrix.swapRows(1, 2)
      assertResult(IndexedSeq(1, -3, 3, 0, 7, 2))(m.rowsByColumns)
    }

    "apply a function to a row" in {
      val m = matrix.mapRow(1)((v, _) => v * 2)
      assertResult(IndexedSeq(1, 6, -3, 0, 4, 7))(m.rowsByColumns)
    }

    "combine rows" in {
      val values = floatMatrix.row(0)
      val m = floatMatrix.combineRows(1, values, -3.0)
      assertResult(IndexedSeq(1.0, 0.0, 3.0, -7.0))(m.rowsByColumns)
    }

    "be inversed" in {
      val matrix  = new Matrix(2, 2, IndexedSeq(4.0, 3.0, 3.0, 2.0))
      val inverse = matrix.inverse
      assertResult(IndexedSeq(-2.0, 3.0, 3.0, -4.0))(inverse.rowsByColumns)
    }
  }

  "An identity matrix" can {
    "be created" in {
      val i = Matrix.identity[Int](3)
      assertResult(IndexedSeq(1, 0, 0, 0, 1, 0, 0, 0, 1))(i.rowsByColumns)
    }

    "be multiplied with another matrix" in {
      val m = Matrix(2, 2, IndexedSeq(3, 0, 1, -2))
      val i = Matrix.identity[Int](2)

      assert(m * i == m)
    }
  }
}
