use crate::adt::interval::Interval;
use crate::analysis::{
    LoopAnalysis, LoopBoundAnalysis, PolyRepr, PolyhedralAnalysis, SCEVExpr, SCEVExprRef,
    ScalarEvolution, VirtualUse,
};
use crate::ir::constraints::ValueDescriptor;
use crate::ir::instruction::IRInstruction;
use crate::ir::{APConstant, Value};
use crate::isa::rdna2::Instruction;
use crate::prover::global_memory::SIZE_OF_DWORD;
use crate::prover::symbolic_heap::{AccessDescriptor, SymbolicHeap, SymbolicLocation};
use std::cell::RefCell;
use std::collections::HashMap;
use std::mem::size_of_val;

/**
 * Given a scalar expression, the scalar range analysis returns the range of the
 * values of the expression. This is used to show that the memory access are inbound.
 **/
pub(crate) struct ScalarRangeAnalysis<'a, 'b> {
    li: &'b LoopAnalysis<'a>,
    lba: &'b LoopBoundAnalysis<'a, 'b>,
    scev: &'b ScalarEvolution<'a, 'b>,
    poly: &'b PolyhedralAnalysis<'a, 'b>,
    ranges: RefCell<RangeMap>,
}

#[derive(Debug, Clone)]
enum RangeResult {
    Resolving,
    Resolved(Option<Interval>),
}
type RangeMap = HashMap<SCEVExprRef, RangeResult>;

impl<'a, 'b> ScalarRangeAnalysis<'a, 'b> {
    pub(crate) fn new(
        li: &'b LoopAnalysis<'a>,
        lba: &'b LoopBoundAnalysis<'a, 'b>,
        scev: &'b ScalarEvolution<'a, 'b>,
        poly: &'b PolyhedralAnalysis<'a, 'b>,
    ) -> ScalarRangeAnalysis<'a, 'b> {
        Self {
            li,
            lba,
            scev,
            poly,
            ranges: RefCell::new(RangeMap::default()),
        }
    }

    pub(crate) fn get(&self, scev: &SCEVExprRef) -> Option<Interval> {
        // Note that an unknown SCEVExprRef might have been resolved thus the analysis has the interval.
        match scev {
            SCEVExprRef::KernelArgumentBase
            | SCEVExprRef::DispatchPacketBase
            | SCEVExprRef::Constant(APConstant::ConstantFloat(_)) => None,
            SCEVExprRef::Constant(APConstant::ConstantInt(val)) => Some(Interval::singleton(*val)),
            _ => {
                let mut range = self.ranges.borrow_mut();
                self.get_internal(&mut range, scev)
            }
        }
    }

    fn get_internal(&self, range: &mut RangeMap, scev: &SCEVExprRef) -> Option<Interval> {
        match range.get(scev) {
            None => {
                range.insert(scev.clone(), RangeResult::Resolving);
                let r = self.analyze_scev(range, scev);
                range.insert(scev.clone(), RangeResult::Resolved(r));
                r
            }
            // The polyhedral analysis should take care the common cases of recursive references.
            // return none if it is beyond the capability of analysis.
            Some(RangeResult::Resolving) => None,
            Some(RangeResult::Resolved(x)) => *x,
        }
    }

    /**
     * Record the range of read-only heap variables given the preconditions.
     **/
    pub(crate) fn run(&self, heap: &SymbolicHeap) {
        let mut range = self.ranges.borrow_mut();
        for (inst_idx, li) in
            self.li
                .func
                .instructions
                .iter()
                .enumerate()
                .filter_map(|(inst_idx, inst)| match Instruction::wrap(inst.clone())? {
                    IRInstruction::LoadInst(li) => Some((inst_idx, li)),
                    _ => None,
                })
        {
            let addr_expr = self.scev.get_scev(VirtualUse::Address(inst_idx));
            if let Some(loc) = SymbolicLocation::trivial_ro_region_access(&addr_expr) {
                if let Some(sv) = heap.get(&loc, li.get_dst_dwords() * SIZE_OF_DWORD) {
                    sv.into_iter().for_each(|(desc, off_bytes, sv)| {
                        assert_eq!(0, off_bytes % 4);
                        let off = off_bytes / 4;
                        let r = match sv.value {
                            ValueDescriptor::SystemPointer(x) => x,
                            ValueDescriptor::HeapPointer(x) => x,
                            ValueDescriptor::Value(x) => x,
                        };
                        let range_result = RangeResult::Resolved(Some(Interval::new(r.min, r.max)));
                        let maybe_32bit_range = r.max < (1 << 31) && r.min >= -(1 << 31);
                        match desc {
                            AccessDescriptor::Bit32 | AccessDescriptor::Low32Of64 => {
                                if maybe_32bit_range {
                                    let uses = VirtualUse::Value(Value::Instruction(inst_idx, off));
                                    let res = self.scev.get_scev(uses);
                                    range.insert(res, range_result);
                                }
                            }
                            AccessDescriptor::Bit64 => {
                                if maybe_32bit_range {
                                    // Sometimes, only the low-32-bits are used
                                    let uses = VirtualUse::Value(Value::Instruction(inst_idx, off));
                                    let res = self.scev.get_scev(uses);
                                    range.insert(res, range_result.clone());
                                }
                                let uses = vec![
                                    Value::Instruction(inst_idx, off),
                                    Value::Instruction(inst_idx, off + 1),
                                ];
                                let uses = VirtualUse::Group(uses);
                                let res = self.scev.get_scev(uses);
                                range.insert(res, range_result);
                            }
                        }
                    })
                }
            }
        }
    }

    fn analyze_scev(&self, range: &mut RangeMap, scev: &SCEVExprRef) -> Option<Interval> {
        match scev {
            SCEVExprRef::KernelArgumentBase
            | SCEVExprRef::DispatchPacketBase
            | SCEVExprRef::Constant(APConstant::ConstantFloat(_))
            | SCEVExprRef::Unknown(_) => None,
            SCEVExprRef::Constant(APConstant::ConstantInt(val)) => Some(Interval::singleton(*val)),
            SCEVExprRef::Expr(expr) => {
                let maybe_poly = if let Some(p) = self.poly.repr.get(scev) {
                    self.analyze_poly_expr(range, p)
                } else {
                    None
                };
                if maybe_poly.is_some() {
                    return maybe_poly;
                }

                match expr {
                    SCEVExpr::SExt(val) | SCEVExpr::ZExt(val) => {
                        self.get_internal(range, val.as_ref())
                    }
                    SCEVExpr::Add(v) => v.iter().try_fold(Interval::singleton(0), |acc, sc| {
                        Some(Interval::add(&acc, &self.get_internal(range, sc)?))
                    }),
                    SCEVExpr::AddRec(addrec) if addrec.is_affine() => {
                        if let Some(poly) = PolyRepr::from_scev(scev, self.scev) {
                            return self.analyze_poly_expr(range, &poly);
                        }
                        if addrec.is_affine() {
                            let base = self.get_internal(range, &addrec.operands[0])?;
                            let mult = self
                                .get_internal(range, &addrec.operands[1])?
                                .single_value()?;
                            let l = &self.li.loops[addrec.loop_info_id];
                            let bound = self.lba.analyze(l)?;
                            let ranges = [bound.start?, bound.final_iv]
                                .into_iter()
                                .filter_map(|x| {
                                    let scev = self.scev.get_scev(VirtualUse::Value(x));
                                    self.get_internal(range, &scev)
                                })
                                .reduce(|lhs, rhs| Interval::union(&lhs, &rhs))?;
                            return Some(Interval::add(&base, &ranges.mul_constant(mult)));
                        }
                        None
                    }
                    SCEVExpr::Mul(v) => v.iter().try_fold(Interval::singleton(1), |acc, sc| {
                        Some(Interval::mul(&acc, &self.get_internal(range, sc)?))
                    }),
                    SCEVExpr::Or(v) => {
                        Self::try_map_reduce(
                            v.iter(),
                            |x| self.get_internal(range, x),
                            |lhs, rhs| {
                                let max_bits = (size_of_val(&rhs.max) * 8) as u32;
                                let mask = match (rhs.max as usize).leading_zeros() {
                                    0 => !0isize,
                                    v => (1isize << (max_bits - v)) - 1,
                                };
                                // A conservative approximation to not enlarging the minimum bound.
                                Some(Interval::new(lhs.min, lhs.max | mask))
                            },
                        )
                    }
                    SCEVExpr::And(v) => {
                        if let Some(result) = self.try_match_bitand_mask(range, v.as_slice()) {
                            return Some(result);
                        }
                        Self::try_map_reduce(
                            v.iter(),
                            |x| self.get_internal(range, x),
                            |lhs, rhs| {
                                if let Some(l) = lhs.single_value() {
                                    Some(Interval::new(rhs.min & l, rhs.max & l))
                                } else {
                                    rhs.single_value()
                                        .map(|r| Interval::new(lhs.min & r, lhs.max & r))
                                }
                            },
                        )
                    }
                    SCEVExpr::AShr(v) => {
                        let lhs = self.get_internal(range, &v[0])?;
                        let r = self
                            .get_internal(range, &v[1])
                            .map(|x| x.single_value())??;
                        (r >= 0).then_some(Interval::new(lhs.min >> r, lhs.max >> r))
                    }
                    SCEVExpr::Shl(v) => {
                        let lhs = self.get_internal(range, &v[0])?;
                        let rhs = self.get_internal(range, &v[1])?;
                        if let Some(lhs) = lhs.single_value() {
                            if lhs >= 0 && rhs.min >= 0 {
                                // c << x
                                return Some(Interval::new(lhs << rhs.min, lhs << rhs.max));
                            }
                        }
                        if let Some(rhs) = rhs.single_value() {
                            if rhs >= 0 {
                                // x << c
                                return Some(lhs.mul_constant(1 << rhs));
                            }
                        }
                        None
                    }
                    SCEVExpr::Div(v) => {
                        let lhs = self.get_internal(range, &v[0])?;
                        let rhs = self.get_internal(range, &v[1])?;
                        Some(Interval::div(&lhs, &rhs))
                    }
                    SCEVExpr::Mod(v) => {
                        Some(Interval::new(0, self.get_internal(range, &v[1])?.max - 1))
                    }
                    SCEVExpr::Select(select) => select
                        .operands
                        .iter()
                        .map(|val| self.get_internal(range, val))
                        .reduce(|lhs, rhs| Some(lhs?.union(&rhs?)))?,
                    SCEVExpr::Xor(v) => self.try_match_abs(range, v),
                    _ => None,
                }
            }
        }
    }

    // match pattern: (-1 + (1 << x)) & y ==> y % (1 << x)
    // conservatively:
    // y % (1 << x) ==> [0, (1 << x) - 1] ==> [0, (1 << x.max) - 1]
    fn try_match_bitand_mask(&self, range: &mut RangeMap, ops: &[SCEVExprRef]) -> Option<Interval> {
        let check_mask_pattern = |x: &SCEVExprRef| {
            if let SCEVExprRef::Expr(SCEVExpr::Add(ops)) = x {
                if ops.len() == 2 && ops[0].as_int() == Some(-1) {
                    if let SCEVExprRef::Expr(SCEVExpr::Shl(ops)) = &ops[1] {
                        if ops.len() == 2 && ops[0].as_int() == Some(1) {
                            if let Some(val) = self.get_internal(range, &ops[1]) {
                                if val.min >= 0 && val.max <= 63 {
                                    // matched pattern -1 + (1 << x)
                                    // return (1 << x.max) - 1
                                    return Some((1 << val.max) - 1);
                                }
                            }
                        }
                    }
                }
            }
            None
        };
        let mask_result = ops
            .iter()
            .filter_map(check_mask_pattern)
            .reduce(|lhs, rhs| lhs & rhs);
        if let Some(mask_result) = mask_result {
            // return [0, (1 << x.max) - 1]
            return Some(Interval::new(0, mask_result));
        }
        None
    }

    // (x >> 31) ^ (x + (x >> 31)) ==> abs(x)
    fn try_match_abs(&self, range: &mut RangeMap, ops: &Vec<SCEVExprRef>) -> Option<Interval> {
        if ops.len() != 2 {
            return None;
        }
        for (lhs, rhs) in [(&ops[0], &ops[1]), (&ops[1], &ops[0])] {
            match lhs {
                SCEVExprRef::Expr(SCEVExpr::AShr(l))
                    if l[1].as_int() == Some(31)
                        && *rhs == self.scev.create_add_expr(&l[0], lhs) =>
                {
                    let val = self.get_internal(range, &l[0])?;
                    return if val.min >= 0 {
                        Some(val)
                    } else if val.max <= 0 {
                        Some(val.mul_constant(-1))
                    } else {
                        Some(Interval::new(0, isize::max(val.max, -val.min)))
                    };
                }
                _ => (),
            }
        }
        None
    }

    fn try_map_reduce<IT: Iterator<Item = R>, M, F, R, S>(
        mut it: IT,
        mut m: M,
        mut f: F,
    ) -> Option<S>
    where
        M: FnMut(R) -> Option<S>,
        F: FnMut(S, S) -> Option<S>,
    {
        let mut first = match it.next() {
            Some(i) => m(i),
            None => return None,
        };

        for v in it.map(m) {
            first = f(first?, v?)
        }
        first
    }

    fn analyze_poly_expr(&self, range: &mut RangeMap, poly: &PolyRepr) -> Option<Interval> {
        let (li, lba) = (self.li, self.lba);
        poly.v
            .iter()
            .try_fold(Interval::singleton(0), |interval, (lid, expr)| {
                if *lid == PolyRepr::CONSTANT_LOOP_ID {
                    let r = self.get_internal(range, expr)?;
                    Some(Interval::add(&interval, &r))
                } else {
                    let mult = self.get_internal(range, expr)?.single_value()?;
                    let l = &li.loops[*lid];
                    let bound = lba.analyze(l)?;
                    let ranges = [bound.start?, bound.final_iv]
                        .into_iter()
                        .filter_map(|x| {
                            let scev = self.scev.get_scev(VirtualUse::Value(x));
                            self.get_internal(range, &scev)
                        })
                        .reduce(|lhs, rhs| Interval::union(&lhs, &rhs))?;
                    Some(Interval::add(&interval, &ranges.mul_constant(mult)))
                }
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{ConstantPropagation, DomFrontier, PHIAnalysis};
    use crate::fileformat::KernelInfo;
    use crate::ir::{DomTree, Function, ImplicitLoopInfo};

    fn test_scalar_range(
        ki: &KernelInfo,
        func: &Function,
        block_size: [usize; 3],
        grid_size: [usize; 3],
        f: fn(&ScalarEvolution, ScalarRangeAnalysis),
    ) {
        let dom = DomTree::analyze(func);
        let df = DomFrontier::new(&dom);
        let def_use = PHIAnalysis::analyze(func, &dom, &df, ki).unwrap();
        let def_use = ConstantPropagation::run(def_use);
        let mut la = LoopAnalysis::new(func);
        la.analyze(&dom);
        la.augment_with_implicit_loops();
        let implicit_loop_info = ImplicitLoopInfo::new(block_size, grid_size);
        let scev = ScalarEvolution::new(&dom, &def_use, &la, Some(implicit_loop_info));
        let mut poly = PolyhedralAnalysis::new(func, &scev);
        poly.analyze();
        let lba = LoopBoundAnalysis::new(&def_use, &scev, Some(implicit_loop_info));
        let range = ScalarRangeAnalysis::new(&la, &lba, &scev, &poly);
        let mut heap = SymbolicHeap::new();
        heap.register_implicit_loop_bounds(&implicit_loop_info);
        range.run(&heap);
        f(&scev, range)
    }

    #[test]
    fn test_grid_size_range() {
        let (ki, func) = &crate::tests::cfg::store_at_grid_size();
        test_scalar_range(ki, func, [256, 1, 1], [64, 1, 1], |scev, range| {
            let expr = scev.get_scev(VirtualUse::Value(Value::Instruction(0, 0)));
            let r = range.get(&expr);
            assert_eq!(r, Some(Interval::singleton(256 * 64)));

            // Propagate the constraints of values loading from the RO region
            let expr = scev.get_scev(VirtualUse::Group(vec![
                Value::Instruction(6, 0),
                Value::Instruction(6, 1),
            ]));
            let r = range.get(&expr);
            assert_eq!(r, Some(Interval::singleton(256 * 64 * 4)));
        });
    }

    #[test]
    fn test_bit_manipulation() {
        let (ki, func) = &crate::tests::scev::bit_operations();
        const EXPECTED_RESULTS: &[Interval] = &[
            Interval::singleton(16),
            Interval::singleton(512),
            Interval::new(0, 512 * 64 - 1),
            Interval::new(0, 512 * 64 / 4096 - 1),
            Interval::new(0, 4096 - 1),
        ];
        test_scalar_range(ki, func, [512, 16, 1], [64, 1, 1], |scev, range| {
            for (idx, v) in EXPECTED_RESULTS.iter().enumerate() {
                let r =
                    range.get(&scev.get_scev(VirtualUse::Value(Value::Instruction(idx + 1, 0))));
                assert_eq!(r, Some(*v));
            }
        });
    }

    #[test]
    fn test_argument_loop_step() {
        let (ki, func) = &crate::tests::scev::argument_loop_step();
        test_scalar_range(ki, func, [1, 1, 1], [1, 1, 1], |scev, range| {
            let r = range.get(&scev.get_scev(VirtualUse::Value(Value::Instruction(3, 0))));
            assert_eq!(r, Some(Interval::new(1, 17)));
        });
    }

    #[test]
    fn test_abs_pattern() {
        let (ki, func) = &crate::tests::scev::abs_pattern();
        test_scalar_range(ki, func, [1, 1, 1], [20, 1, 1], |scev, range| {
            let r = range.get(&scev.get_scev(VirtualUse::Value(Value::Instruction(3, 0))));
            // abs([0, 19] - 12) = abs([-12, 7]) = [0, 12]
            assert_eq!(r, Some(Interval::new(0, 12)));
        });
    }

    #[test]
    fn test_poly_range() {
        let (ki, func) = &crate::tests::scev::nested_loop();
        test_scalar_range(ki, func, [1, 1, 1], [1, 1, 1], |scev, range| {
            let base = scev.get_scev(VirtualUse::Group(vec![
                Value::Instruction(0, 0),
                Value::Instruction(0, 1),
            ]));
            range
                .ranges
                .borrow_mut()
                .insert(base, RangeResult::Resolved(Some(Interval::new(4096, 8192))));
            let r = range.get(&scev.get_scev(VirtualUse::Address(12)));
            assert_eq!(r, Some(Interval::new(4096, 8192 + 24 * 5 + 4 * 6)));
        });
    }

    #[test]
    fn test_load_argument_value() {
        let t = &crate::tests::memory_analysis::load_argument_value();
        let func = &t.func;
        let ki = &t.kernel;
        let dom = DomTree::analyze(func);
        let df = DomFrontier::new(&dom);
        let def_use = PHIAnalysis::analyze(func, &dom, &df, ki).unwrap();
        let def_use = ConstantPropagation::run(def_use);
        let mut la = LoopAnalysis::new(func);
        la.analyze(&dom);
        la.augment_with_implicit_loops();
        let implicit_loop_info = ImplicitLoopInfo::new([512, 16, 1], [64, 1, 1]);
        let scev = ScalarEvolution::new(&dom, &def_use, &la, Some(implicit_loop_info));
        let mut poly = PolyhedralAnalysis::new(func, &scev);
        poly.analyze();
        let lba = LoopBoundAnalysis::new(&def_use, &scev, Some(implicit_loop_info));
        let range = ScalarRangeAnalysis::new(&la, &lba, &scev, &poly);
        let mut heap = SymbolicHeap::new();
        t.register_constraints(&mut heap);
        heap.register_implicit_loop_bounds(&implicit_loop_info);
        range.run(&heap);
        // coalesced access
        assert_eq!(
            Some(Interval::new(100, 200)),
            range.get(&scev.get_scev(VirtualUse::Value(Value::Instruction(0, 0))))
        );
        assert_eq!(
            Some(Interval::new(300, 400)),
            range.get(&scev.get_scev(VirtualUse::Group(vec![
                Value::Instruction(0, 2),
                Value::Instruction(0, 3),
            ])))
        );
        // access low-32-bits of an 64-bit argument (suppose the argument to be no more than 2^32 - 1)
        assert_eq!(
            Some(Interval::new(300, 400)),
            range.get(&scev.get_scev(VirtualUse::Value(Value::Instruction(1, 0))))
        );
        assert_eq!(
            Some(Interval::new(300, 400)),
            range.get(&scev.get_scev(VirtualUse::Value(Value::Instruction(0, 2))))
        );
    }

    #[test]
    fn test_shl() {
        let (ki, func) = &crate::tests::scev::shl_operations();
        test_scalar_range(ki, func, [60000, 1, 1], [8, 1, 1], |scev, range| {
            let r = range.get(&scev.get_scev(VirtualUse::Value(Value::Instruction(1, 0))));
            assert_eq!(r, Some(Interval::new(1 << 4, 1 << 11)));
            let r = range.get(&scev.get_scev(VirtualUse::Value(Value::Instruction(3, 0))));
            assert_eq!(r, Some(Interval::new(0, (1 << 11) - 1)));
        });
    }

    #[test]
    fn test_complex_poly_range() {
        let (ki, func) = &crate::tests::scev::complex_nested_loop();
        test_scalar_range(ki, func, [1, 1, 1], [1, 1, 1], |scev, range| {
            let r = range.get(&scev.get_scev(VirtualUse::Value(Value::Instruction(5, 0))));
            assert_eq!(r, Some(Interval::new(4 + 48, 4 + 48 + 24 * 5 + 4 * 6)));
        });
    }

    #[test]
    fn test_integet_division() {
        let (ki, func) = &crate::tests::scev::integer_division();
        test_scalar_range(ki, func, [1, 1, 1], [1, 1, 1], |scev, range| {
            let dividend = scev.get_scev(VirtualUse::Value(Value::Instruction(0, 0)));
            let divisor = scev.get_scev(VirtualUse::Value(Value::Instruction(1, 0)));
            range.ranges.borrow_mut().insert(
                dividend,
                RangeResult::Resolved(Some(Interval::new(4096, 8192))),
            );
            range.ranges.borrow_mut().insert(
                divisor,
                RangeResult::Resolved(Some(Interval::new(128, 256))),
            );
            let div = range.get(&scev.get_scev(VirtualUse::Value(Value::Instruction(20, 0))));
            let rem = range.get(&scev.get_scev(VirtualUse::Value(Value::Instruction(22, 0))));
            let div56 = range.get(&scev.get_scev(VirtualUse::Value(Value::Instruction(27, 0))));
            let div87 = range.get(&scev.get_scev(VirtualUse::Value(Value::Instruction(29, 0))));
            assert_eq!(div, Some(Interval::new(16, 64)));
            assert_eq!(rem, Some(Interval::new(0, 255)));
            assert_eq!(div56, Some(Interval::new(2, 4)));
            assert_eq!(div87, Some(Interval::new(2, 4)));
        });
    }
}
