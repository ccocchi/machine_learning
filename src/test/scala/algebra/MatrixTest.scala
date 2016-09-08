package algebra

import org.scalatest.{FunSpec, WordSpec}

class MatrixTest extends FunSpec {
  /*
    [ 1   0
      3   2
      -3  7 ]
   */
  val m1 = IndexedSeq(1, 3, -3, 0, 2, 7).reshape(3, 2)

  describe("A matrix") {
    describe("when transposing") {
      val t = m1.transpose

      it("switches dimensions") {
        assertResult(2)(t.rowSize)
        assertResult(3)(t.colSize)
      }

      it("is still randomly accessible") {
        assertResult(3)(t(0, 1))
        assertResult(7)(t(1, 2))
      }

      it("can be chained with dot operation") {
        val other = IndexedSeq(1, -1, 2).reshape(3, 1)
        val res   = t.dot(other)

        assertResult(2)(res.rowSize)
        assertResult(1)(res.colSize)

        assertResult(-8)(res(0, 0))
        assertResult(12)(res(1, 0))
      }
    }

    it("can be multiplied by another matrix") {
      val other = IndexedSeq(1, -1, 0, 2).reshape(2, 2)
      val res   = m1.dot(other)

      assertResult(3)(res.rowSize)
      assertResult(2)(res.colSize)

      assertResult(1)(res(0, 0))
      assertResult(4)(res(1, 1))
      assertResult(14)(res(2, 1))
    }

    it("can be mapped with a function") {
      val other = m1.map(v => v * 2)

      assertResult(2)(other(0, 0))
      assertResult(4)(other(1, 1))
    }

    it("can be mapped with another matrix and a function") {
      val other = IndexedSeq(1, 3, -3, 0, 2, 7).reshape(3, 2)
      val res = m1.mapWith(other)((x, y) => x + y)

      assertResult(2)(res(0, 0))
      assertResult(-6)(res(2, 0))
    }

    //    "be accessed randomly" in {
    //      assertResult(7)(m1(2, 1))
    //    }
    //
    //    "be transposed" in {
    //      val v   = m1(0, 1)
    //      val res = m1.transpose
    //      //assert(res.isTranspose)
    //
    //      assertResult(2)(res.rowSize)
    //      assertResult(3)(res.colSize)
    //
    //      assertResult(v)(res(1, 0))
    //      assertResult(3)(res(0, 1))
    //    }
    //
    //    "be summed by its columns" in {
    //      val res = m1.sumColumns
    //      assertResult(3)(res.rowSize)
    //      assertResult(1)(res.colSize)
    //      assertResult(IndexedSeq(1, 5, 4))(res.values)
    //    }
    //  }
    //
    //  "A vector" can {
    //
    //  }
  }
}
