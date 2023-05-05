use smallvec::SmallVec;

use crate::analysis::scalar_evolution::VirtualUse;
use crate::analysis::{SCEVExpr, SCEVExprRef, ScalarEvolution};
use crate::ir::Function;
use crate::isa::rdna2::isa::Opcode;
use crate::isa::rdna2::opcodes::SMEMOpcode;
use std::collections::{BTreeMap, HashMap};

/**
 * Basic polyhedral analysis. Ignore all conditions for now.
 **/
#[derive(Clone)]
pub struct PolyhedralAnalysis<'a, 'b> {
    func: &'a Function<'a>,
    scev: &'b ScalarEvolution<'a, 'b>,
    // the instruction index of the load store instructions -> poly representation
    pub(crate) repr: HashMap<SCEVExprRef, PolyRepr>,
}

/**
 * The polyhedral representation of a value.
 **/
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolyRepr {
    // Sparse map of loop_id -> SCEV expression.
    pub v: Vec<(usize, SCEVExprRef)>,
}

impl PolyRepr {
    // The synthetic loop id that represents the constants
    pub const CONSTANT_LOOP_ID: usize = u32::MAX as usize;

    fn sum(ops: &[PolyRepr], se: &ScalarEvolution) -> Self {
        let mut m: BTreeMap<usize, SCEVExprRef> = BTreeMap::new();
        for op in ops.iter() {
            for (lid, scev) in op.v.iter() {
                if let Some(x) = m.remove(lid) {
                    m.insert(*lid, se.create_add_expr_impl([x, scev.clone()].as_slice()));
                } else {
                    m.insert(*lid, scev.clone());
                }
            }
        }
        let v = m
            .into_iter()
            .filter(|(_, scev)| scev.as_int() != Some(0))
            .collect();
        Self { v }
    }

    fn mul_scev(&self, scev: &SCEVExprRef, se: &ScalarEvolution) -> Self {
        let v = self
            .v
            .iter()
            .filter_map(|(lid, x)| {
                let scev = se.create_mul_expr(x, scev);
                if scev.as_int() != Some(0) {
                    Some((*lid, scev))
                } else {
                    None
                }
            })
            .collect();
        Self { v }
    }

    fn as_constant(&self) -> Option<&SCEVExprRef> {
        if self.v.len() == 1 {
            let (lid, scev) = &self.v[0];
            (*lid == PolyRepr::CONSTANT_LOOP_ID).then_some(scev)
        } else {
            None
        }
    }

    fn mul(lhs: &Self, rhs: &Self, se: &ScalarEvolution) -> Option<Self> {
        if lhs.v.is_empty() || rhs.v.is_empty() {
            Some(PolyRepr { v: vec![] })
        } else if let Some(c) = lhs.as_constant() {
            Some(rhs.mul_scev(c, se))
        } else {
            rhs.as_constant().map(|c| lhs.mul_scev(c, se))
        }
    }

    pub(crate) fn from_scev(scev: &SCEVExprRef, se: &ScalarEvolution) -> Option<PolyRepr> {
        use SCEVExpr::*;
        match scev {
            SCEVExprRef::Constant(_)
            | SCEVExprRef::Unknown(_)
            | SCEVExprRef::KernelArgumentBase
            | SCEVExprRef::DispatchPacketBase => Some(PolyRepr {
                v: vec![(PolyRepr::CONSTANT_LOOP_ID, scev.clone())],
            }),
            SCEVExprRef::Expr(expr) => match expr {
                Add(ops) => {
                    let mut addens: SmallVec<[_; 4]> = SmallVec::new();
                    for scev in ops.iter() {
                        addens.push(Self::from_scev(scev, se)?);
                    }
                    Some(PolyRepr::sum(addens.as_slice(), se))
                }
                Mul(ops) if ops.len() == 2 => {
                    let lhs = Self::from_scev(&ops[0], se)?;
                    let rhs = Self::from_scev(&ops[1], se)?;
                    PolyRepr::mul(&lhs, &rhs, se)
                }
                AddRec(addrec) if addrec.is_affine() => {
                    let start = &addrec.operands[0];
                    let step = &addrec.operands[1];
                    // start + loop_idx * step
                    let start = Self::from_scev(start, se)?;
                    let steps = PolyRepr {
                        v: vec![(addrec.loop_info_id, step.clone())],
                    };
                    Some(PolyRepr::sum(&[start, steps], se))
                }
                ZExt(sub_secv) | SExt(sub_secv) => Self::from_scev(sub_secv.as_ref(), se),
                // If all sub-expressions are in constant-loop, the parent expression is also in constant-loop.
                And(ops) | Or(ops) | AShr(ops)
                    if ops.iter().all(|op| {
                        let p = Self::from_scev(op, se);
                        if let Some(p) = p {
                            p.as_constant().is_some()
                        } else {
                            false
                        }
                    }) =>
                {
                    Some(PolyRepr {
                        v: vec![(PolyRepr::CONSTANT_LOOP_ID, scev.clone())],
                    })
                }
                Select(select)
                    if select.operands.iter().all(|op| {
                        let p = Self::from_scev(op, se);
                        if let Some(p) = p {
                            p.as_constant().is_some()
                        } else {
                            false
                        }
                    }) =>
                {
                    Some(PolyRepr {
                        v: vec![(PolyRepr::CONSTANT_LOOP_ID, scev.clone())],
                    })
                }
                _ => None,
            },
        }
    }
}

impl<'a, 'b> PolyhedralAnalysis<'a, 'b> {
    pub fn new(func: &'a Function<'a>, scev: &'b ScalarEvolution<'a, 'b>) -> Self {
        Self {
            func,
            scev,
            repr: Default::default(),
        }
    }

    pub fn analyze(&mut self) {
        let mut repr = HashMap::new();
        let scev = self.scev;
        for (inst_idx, inst) in self.func.instructions.iter().enumerate() {
            match inst.op {
                Opcode::SMEM(SMEMOpcode::S_LOAD_DWORDX2)
                | Opcode::SMEM(SMEMOpcode::S_LOAD_DWORDX4)
                | Opcode::SMEM(SMEMOpcode::S_LOAD_DWORDX8)
                | Opcode::SMEM(SMEMOpcode::S_LOAD_DWORDX16)
                | Opcode::SMEM(SMEMOpcode::S_BUFFER_LOAD_DWORDX2)
                | Opcode::SMEM(SMEMOpcode::S_BUFFER_LOAD_DWORDX4)
                | Opcode::SMEM(SMEMOpcode::S_BUFFER_LOAD_DWORDX8)
                | Opcode::SMEM(SMEMOpcode::S_BUFFER_LOAD_DWORDX16)
                | Opcode::VMEM(_) => {
                    let expr = scev.get_scev(VirtualUse::Address(inst_idx));
                    if let Some(x) = self.construct_poly_repr(&expr) {
                        repr.insert(expr, x);
                    };
                }
                _ => {}
            }
        }
        self.repr = repr;
    }

    fn construct_poly_repr(&self, scev: &SCEVExprRef) -> Option<PolyRepr> {
        PolyRepr::from_scev(scev, self.scev)
    }
}

#[cfg(test)]
mod tests {
    use crate::analysis::polyhedral::PolyhedralAnalysis;
    use crate::analysis::scalar_evolution::ScalarEvolution;
    use crate::analysis::{
        ConstantPropagation, DomFrontier, LoopAnalysis, PHIAnalysis, SCEVExpr, SCEVExprRef,
        VirtualUse,
    };
    use crate::ir::DomTree;
    use crate::tests::scev::test_with_dummy_scev;

    use super::PolyRepr;

    #[test]
    fn test_polyrepr() {
        let (ki, func) = crate::tests::cfg::nested_loop();
        let dom = DomTree::analyze(&func);
        let df = DomFrontier::new(&dom);
        let def_use = PHIAnalysis::analyze(&func, &dom, &df, &ki).expect("Cannot analyze PHI");
        let def_use = ConstantPropagation::run(def_use);
        let dom = DomTree::analyze(&func);
        let mut li = LoopAnalysis::new(&func);
        li.analyze(&dom);
        let scev = ScalarEvolution::new(&dom, &def_use, &li, None);
        let mut poly = PolyhedralAnalysis::new(&func, &scev);
        poly.analyze();
        assert_eq!(2, poly.repr.len());
        let k = scev.get_scev(VirtualUse::Address(11));
        let first = poly
            .repr
            .get(&k)
            .unwrap()
            .v
            .iter()
            .map(|(loop_id, _)| *loop_id)
            .collect::<Vec<usize>>();
        assert_eq!(vec![0, 1, PolyRepr::CONSTANT_LOOP_ID], first); // there is a constant "unknown" base address
    }

    fn create_poly_repr(v: &[isize], const_value: Option<isize>) -> PolyRepr {
        let mut v: Vec<(usize, SCEVExprRef)> = v
            .iter()
            .enumerate()
            .filter(|(_, x)| **x != 0)
            .map(|(i, x)| (i, SCEVExpr::create_const_int(*x)))
            .collect();
        if let Some(c) = const_value {
            v.push((PolyRepr::CONSTANT_LOOP_ID, SCEVExpr::create_const_int(c)));
        }
        PolyRepr { v }
    }

    #[test]
    fn test_poly_add() {
        test_with_dummy_scev(|se| {
            let lhs = create_poly_repr(&[1, 2, 0], Some(3));
            let rhs = create_poly_repr(&[0, 1, 2], Some(-3));
            let expected = create_poly_repr(&[1, 3, 2], None);
            assert_eq!(PolyRepr::sum(&[lhs, rhs], se), expected);
        })
    }

    #[test]
    fn test_poly_mul() {
        test_with_dummy_scev(|se| {
            let lhs = create_poly_repr(&[1, 2, 0], Some(3));
            let rhs = create_poly_repr(&[0, 0, 0], Some(-3));
            let expected = create_poly_repr(&[-3, -6, 0], Some(-9));
            assert_eq!(PolyRepr::mul(&lhs, &rhs, se), Some(expected));
        })
    }

    #[test]
    fn test_poly_mul_zero() {
        test_with_dummy_scev(|se| {
            let lhs = create_poly_repr(&[1, 2, 0], Some(3));
            let rhs = create_poly_repr(&[0, 0, 0], Some(0));
            let expected = create_poly_repr(&[0, 0, 0], None);
            assert_eq!(PolyRepr::mul(&lhs, &rhs, se), Some(expected));
        })
    }

    #[test]
    fn test_poly_mul_fail() {
        test_with_dummy_scev(|se| {
            let lhs = create_poly_repr(&[1, 2, 0], Some(3));
            let rhs = create_poly_repr(&[0, 0, -3], Some(0));
            assert_eq!(PolyRepr::mul(&lhs, &rhs, se), None);
        })
    }

    #[test]
    fn test_poly_from_constant() {
        test_with_dummy_scev(|se| {
            let scev = SCEVExpr::create_const_int(42);
            let poly = create_poly_repr(&[0, 0, 0], Some(42));
            assert_eq!(PolyRepr::from_scev(&scev, se), Some(poly));
        })
    }

    #[test]
    fn test_poly_from_addrec() {
        test_with_dummy_scev(|se| {
            let start = SCEVExpr::create_const_int(42);
            let step = SCEVExpr::create_const_int(-1);
            let scev = se.create_add_rec_expr(&start, &step, 0);
            let poly = create_poly_repr(&[-1, 0, 0], Some(42));
            assert_eq!(PolyRepr::from_scev(&scev, se), Some(poly));
        })
    }

    #[test]
    fn test_poly_from_multi_addrec() {
        test_with_dummy_scev(|se| {
            let start = SCEVExpr::create_const_int(42);
            let step = SCEVExpr::create_const_int(-1);
            let start = se.create_add_rec_expr(&start, &step, 1);
            let step = SCEVExpr::create_const_int(16);
            let scev = se.create_add_rec_expr(&start, &step, 0);
            let poly = create_poly_repr(&[16, -1, 0], Some(42));
            assert_eq!(PolyRepr::from_scev(&scev, se), Some(poly));
        })
    }

    #[test]
    fn test_poly_from_add() {
        test_with_dummy_scev(|se| {
            let start = SCEVExpr::create_const_int(42);
            let step = SCEVExpr::create_const_int(-1);
            let lhs = se.create_add_rec_expr(&start, &step, 0);
            let rhs = SCEVExpr::create_const_int(7);
            let scev = se.create_add_expr(&lhs, &rhs);
            let poly = create_poly_repr(&[-1, 0, 0], Some(49));
            assert_eq!(PolyRepr::from_scev(&scev, se), Some(poly));
        })
    }

    #[test]
    fn test_poly_from_mul() {
        test_with_dummy_scev(|se| {
            let start = SCEVExpr::create_const_int(6);
            let step = SCEVExpr::create_const_int(-1);
            let lhs = se.create_add_rec_expr(&start, &step, 0);
            let rhs = SCEVExpr::create_const_int(7);
            let scev = se.create_mul_expr(&lhs, &rhs);
            let poly = create_poly_repr(&[-7, 0, 0], Some(42));
            assert_eq!(PolyRepr::from_scev(&scev, se), Some(poly));
        })
    }

    #[test]
    fn test_poly_from_zext() {
        test_with_dummy_scev(|se| {
            let scev = SCEVExpr::create_const_int(42);
            let scev = se.create_extend_expr(false, &scev);
            let poly = create_poly_repr(&[0, 0, 0], Some(42));
            assert_eq!(PolyRepr::from_scev(&scev, se), Some(poly));
        })
    }

    #[test]
    fn test_constant_loop() {
        test_with_dummy_scev(|se| {
            let s1 = se.create_unknown();
            let s2 = se.create_unknown();
            let s5 = se.create_and_expr(&s1, &s2);
            let s6 = se.create_or_expr(&s1, &s2);
            let s7 = se.create_ashr_expr(&s5, &SCEVExpr::create_const_int(2));
            let s8 = ScalarEvolution::create_select_expr(&s6, &s7);
            if let Some(poly) = PolyRepr::from_scev(&s8, se) {
                assert_eq!(Some(&s8), poly.as_constant())
            } else {
                assert!(false);
            }
        })
    }
}
