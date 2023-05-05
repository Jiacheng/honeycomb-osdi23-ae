/**
 * An overestimation to estimate the range of variables as a supplement of the polyhedral analysis.
 */
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct Interval {
    pub min: isize,
    pub max: isize,
}

impl Interval {
    pub(crate) const fn new(min: isize, max: isize) -> Interval {
        Self { min, max }
    }

    pub(crate) const fn singleton(x: isize) -> Interval {
        Self::new(x, x)
    }

    pub(crate) fn mul_constant(&self, x: isize) -> Interval {
        if x > 0 {
            Self::new(self.min * x, self.max * x)
        } else {
            Self::new(self.max * x, self.min * x)
        }
    }

    pub(crate) fn union(&self, other: &Interval) -> Interval {
        Self::new(self.min.min(other.min), self.max.max(other.max))
    }

    pub(crate) fn add(lhs: &Interval, rhs: &Interval) -> Interval {
        Self::new(lhs.min + rhs.min, lhs.max + rhs.max)
    }

    #[allow(dead_code)]
    pub(crate) fn sub(lhs: &Interval, rhs: &Interval) -> Interval {
        Self::new(lhs.min - rhs.min, lhs.max - rhs.max)
    }

    /*
     * Multiplication significantly overestimates the range. Should be used judiciously.
     */
    pub(crate) fn mul(lhs: &Interval, rhs: &Interval) -> Interval {
        let mut values = [
            lhs.min * rhs.min,
            lhs.min * rhs.max,
            lhs.max * rhs.min,
            lhs.max * rhs.max,
        ];
        values.sort();
        Self::new(values[0], values[3])
    }

    pub(crate) fn div(lhs: &Interval, rhs: &Interval) -> Interval {
        assert!(rhs.min > 0 || rhs.max < 0);
        let mut values = [
            lhs.min / rhs.min,
            lhs.min / rhs.max,
            lhs.max / rhs.min,
            lhs.max / rhs.max,
        ];
        values.sort();
        Self::new(values[0], values[3])
    }

    pub(crate) fn single_value(&self) -> Option<isize> {
        (self.min == self.max).then_some(self.min)
    }
}
