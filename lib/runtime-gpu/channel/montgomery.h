#pragma once

#include "bigint.h"
#include "runtime-gpu/core/abi.h"

namespace gpumpc {
template <class Field> class MontNum {
  public:
    using Number = typename Field::Number;
    GPUMPC_HOST_AND_DEVICE static constexpr MontNum One();
    GPUMPC_HOST_AND_DEVICE static MontNum Reduce(const Number &bn);
    GPUMPC_HOST_AND_DEVICE static void
    Multiply(MontNum *res, const MontNum &lhs, const MontNum &rhs);
    GPUMPC_HOST_AND_DEVICE Number Recover() const;
    explicit constexpr MontNum(const Number &bn) : data_(bn) {}
    MontNum() = default;
    template <unsigned kExpBits>
    GPUMPC_HOST_AND_DEVICE MontNum Pow(const BigInt<kExpBits> &exp) const;

    void Dump() const { data_.Dump(); }
    GPUMPC_HOST_AND_DEVICE const Number &Data() const { return data_; }

  private:
    Number data_;
};

template <class Field>
GPUMPC_HOST_AND_DEVICE inline constexpr MontNum<Field> MontNum<Field>::One() {
    return MontNum(Field::R());
}

template <class Field>
GPUMPC_HOST_AND_DEVICE void
MontNum<Field>::Multiply(MontNum *res, const MontNum &lhs, const MontNum &rhs) {
    Number lo, hi, m, r, t;
    Number::MulLo(&lo, lhs.data_, rhs.data_);
    Number::MulHi(&hi, lhs.data_, rhs.data_);
    Number::MulLo(&m, lo, Field::NPrime());
    Number::MulHi(&r, m, Field::N());
    Number::AddWithCC(&t, hi, r, 1);
    if (t >= Field::N()) {
        Number::AddWithCC(&res->data_, t, Field::MinusN(), 0);
    } else {
        res->data_ = t;
    }
}

template <class Field>
GPUMPC_HOST_AND_DEVICE MontNum<Field> MontNum<Field>::Reduce(const Number &bn) {
    MontNum t;
    MontNum r(bn);
    MontNum::Multiply(&t, r, Field::MontRawR2());
    return t;
}

template <class Field>
GPUMPC_HOST_AND_DEVICE typename MontNum<Field>::Number
MontNum<Field>::Recover() const {
    MontNum o(Number::One());
    MontNum t;
    MontNum::Multiply(&t, *this, o);
    return t.data_;
}

template <class Field>
template <unsigned kExpBits>
GPUMPC_HOST_AND_DEVICE inline MontNum<Field>
MontNum<Field>::Pow(const BigInt<kExpBits> &exp) const {
    using ExpNum = BigInt<kExpBits>;
    MontNum u(*this);
    MontNum ret = MontNum::One();
    const auto &d = exp.Digits();
    for (unsigned i = 0; i < ExpNum::kWords; i++) {
        unsigned v = d[i];
        for (unsigned j = 0; j < 32; j++) {
            if (v & (1 << j)) {
                MontNum::Multiply(&ret, ret, u);
            }
            MontNum::Multiply(&u, u, u);
        }
    }
    return ret;
}

} // namespace gpumpc