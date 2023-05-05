use super::match_div::DivPattern;
use crate::analysis::group_pattern::WideOperand;
use crate::analysis::phi::InstructionUse;
use crate::analysis::{GroupPattern, Loop, LoopAnalysis, PHIAnalysis};
use crate::ir::instruction::{BinaryOperator, IRInstruction};
use crate::ir::{APConstant, Value, PHI};
use crate::ir::{DomTree, ImplicitLoopInfo};
use crate::isa::rdna2::Instruction;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

//
// TODO: Optimize for the excessive copies during analysis
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum SCEVExpr {
    Add(Vec<SCEVExprRef>),
    Mul(Vec<SCEVExprRef>),
    Shl(Vec<SCEVExprRef>),
    AShr(Vec<SCEVExprRef>),
    And(Vec<SCEVExprRef>),
    Or(Vec<SCEVExprRef>),
    Xor(Vec<SCEVExprRef>),
    Div(Vec<SCEVExprRef>),
    Mod(Vec<SCEVExprRef>),
    AddRec(AddRec),
    ZExt(Box<SCEVExprRef>), // zero-extend
    SExt(Box<SCEVExprRef>), // signed-extend
    Select(Select),
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum SCEVExprRef {
    Expr(SCEVExpr),
    Constant(APConstant),
    KernelArgumentBase,
    DispatchPacketBase,
    // Unknown can also represent a symbolic variable, uniquely identified by its ID
    Unknown(usize),
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct AddRec {
    pub(crate) operands: Vec<SCEVExprRef>,
    pub loop_info_id: usize,
}

/*
 * The select expression captures the condition and all arms of possible outcomes to enable
 * subsequent analysis to have the flexibility to compute path-sensitive results.
 *
 * TODO: Potentially lower the select expression to min / max for more precise analysis
 */
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Select {
    // TODO: Actually fill it up
    pub(crate) selector: Option<Value>,
    pub(crate) operands: Vec<SCEVExprRef>,
}

pub struct ScalarEvolution<'a, 'b> {
    dom: &'b DomTree<'a>,
    la: &'b LoopAnalysis<'a>,
    def_use: &'b PHIAnalysis<'a>,
    implicit_loop_info: Option<ImplicitLoopInfo>,
    ctx: RefCell<Context>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum VirtualUse {
    Value(Value),
    // Represent a group of registers (e.g. 2 registers that represent a 64-bit integer).
    Group(Vec<Value>),
    // The address of memory instructions
    Address(usize),
}

#[derive(Default)]
struct ContextFrame {
    expr_value_map: HashMap<SCEVExprRef, VirtualUse>,
    value_expr_map: HashMap<VirtualUse, SCEVExprRef>,
}

#[derive(Default)]
struct Context {
    stack: Vec<ContextFrame>,
    // use global counter to avoid alias
    symbol_counter: usize,
}

impl Context {
    fn new() -> Self {
        let mut r = Self {
            stack: vec![],
            symbol_counter: 0,
        };
        r.push();
        r
    }

    fn insert(&mut self, vuse: VirtualUse, scev: SCEVExprRef) {
        let frame = self.stack.last_mut().unwrap();
        frame.expr_value_map.insert(scev.clone(), vuse.clone());
        frame.value_expr_map.insert(vuse, scev);
    }

    fn get_scev(&self, vuse: &VirtualUse) -> Option<SCEVExprRef> {
        self.stack
            .iter()
            .rev()
            .find_map(|x| x.value_expr_map.get(vuse))
            .cloned()
    }

    fn push(&mut self) {
        self.stack.push(Default::default());
    }

    fn pop(&mut self) {
        assert!(self.stack.len() > 1);
        self.stack.pop();
    }
}

impl SCEVExpr {
    fn scev_type_id(&self) -> u32 {
        use SCEVExpr::*;
        match self {
            Add(_) => 1,
            Mul(_) => 2,
            Shl(_) => 3,
            AShr(_) => 4,
            And(_) => 5,
            Or(_) => 6,
            Xor(_) => 7,
            Div(_) => 8,
            Mod(_) => 9,
            AddRec(_) => 10,
            ZExt(_) => 11,
            SExt(_) => 12,
            Select(_) => 13,
        }
    }

    pub(crate) fn create_const_int(v: isize) -> SCEVExprRef {
        SCEVExprRef::new_constant(APConstant::ConstantInt(v))
    }
}

impl SCEVExprRef {
    fn new(expr: SCEVExpr) -> Self {
        SCEVExprRef::Expr(expr)
    }

    fn new_constant(expr: APConstant) -> SCEVExprRef {
        SCEVExprRef::Constant(expr)
    }

    fn scev_type_id(&self) -> u32 {
        match self {
            SCEVExprRef::Expr(e) => e.scev_type_id(),
            SCEVExprRef::Constant(_) => 0,
            SCEVExprRef::KernelArgumentBase => 14,
            SCEVExprRef::DispatchPacketBase => 15,
            SCEVExprRef::Unknown(_) => u32::MAX,
        }
    }

    pub fn get(&self, _se: &ScalarEvolution) -> Option<SCEVExpr> {
        match self {
            SCEVExprRef::Expr(e) => Some(e.clone()),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<isize> {
        match self {
            SCEVExprRef::Constant(APConstant::ConstantInt(x)) => Some(*x),
            _ => None,
        }
    }

    fn is_loop_invariant(&self, _l: &Loop) -> bool {
        match self {
            SCEVExprRef::DispatchPacketBase
            | SCEVExprRef::KernelArgumentBase
            | SCEVExprRef::Constant(_) => true,
            SCEVExprRef::Unknown(_) => false,
            SCEVExprRef::Expr(expr) => match expr {
                SCEVExpr::Add(ops)
                | SCEVExpr::Mul(ops)
                | SCEVExpr::Shl(ops)
                | SCEVExpr::AShr(ops)
                | SCEVExpr::And(ops)
                | SCEVExpr::Or(ops)
                | SCEVExpr::Xor(ops)
                | SCEVExpr::Div(ops)
                | SCEVExpr::Mod(ops) => ops.iter().all(|op| op.is_loop_invariant(_l)),
                SCEVExpr::ZExt(op) | SCEVExpr::SExt(op) => op.is_loop_invariant(_l),
                // A conservative judgment, treat all select and addrec expressions as non-invariant
                SCEVExpr::Select(_) => false,
                SCEVExpr::AddRec(_) => false,
            },
        }
    }
}

impl PartialOrd for SCEVExprRef {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.scev_type_id().partial_cmp(&other.scev_type_id())
    }
}

impl AddRec {
    pub(crate) fn is_affine(&self) -> bool {
        self.operands.len() == 2
    }

    pub(crate) fn get_step_recurrence(&self) -> Option<SCEVExprRef> {
        self.is_affine().then(|| self.operands[1].clone())
    }
}

impl<'a, 'b> ScalarEvolution<'a, 'b> {
    pub fn new(
        dom: &'b DomTree<'a>,
        def_use: &'b PHIAnalysis<'a>,
        la: &'b LoopAnalysis<'a>,
        implicit_loop_info: Option<ImplicitLoopInfo>,
    ) -> ScalarEvolution<'a, 'b> {
        ScalarEvolution {
            dom,
            la,
            def_use,
            implicit_loop_info,
            ctx: RefCell::new(Context::new()),
        }
    }

    pub fn get_scev(&self, u: VirtualUse) -> SCEVExprRef {
        if let Some(cached) = self.ctx.borrow().get_scev(&u) {
            return cached;
        }

        let r = match &u {
            VirtualUse::Value(v) => self.create_scev_value(v),
            VirtualUse::Group(v) => self.create_scev_value_group(v.as_slice()),
            VirtualUse::Address(v) => self.create_scev_for_vmem_addr(*v),
        }
        .unwrap_or_else(|| self.create_unknown());

        self.ctx.borrow_mut().insert(u, r.clone());
        r
    }

    pub(crate) fn create_unknown(&self) -> SCEVExprRef {
        let mut ctx = self.ctx.borrow_mut();
        let id = ctx.symbol_counter;
        ctx.symbol_counter += 1;
        SCEVExprRef::Unknown(id)
    }

    fn maybe_binary_op_scev(
        &self,
        operator: BinaryOperator,
        lhs: &SCEVExprRef,
        rhs: &SCEVExprRef,
    ) -> Option<SCEVExprRef> {
        match operator {
            BinaryOperator::Add => Some(self.create_add_expr(lhs, rhs)),
            BinaryOperator::Sub => Some(self.create_sub_expr(lhs, rhs)),
            BinaryOperator::Mul => Some(self.create_mul_expr(lhs, rhs)),
            BinaryOperator::LShift => Some(self.create_shl_expr(lhs, rhs)),
            BinaryOperator::AShr => Some(self.create_ashr_expr(lhs, rhs)),
            BinaryOperator::Min | BinaryOperator::Max => Some(Self::create_select_expr(lhs, rhs)),
            BinaryOperator::And => Some(self.create_and_expr(lhs, rhs)),
            BinaryOperator::Or => Some(self.create_or_expr(lhs, rhs)),
            BinaryOperator::Xor => Some(self.create_xor_expr(lhs, rhs)),
        }
    }

    fn create_scev_value(&self, v: &Value) -> Option<SCEVExprRef> {
        match DivPattern::try_match(*v, self.def_use) {
            Some(DivPattern::Div(x, y)) => {
                let x = self.get_scev(VirtualUse::Value(x));
                let y = self.get_scev(VirtualUse::Value(y));
                return Some(self.create_div_expr(&x, &y));
            }
            Some(DivPattern::Rem(x, y)) => {
                let x = self.get_scev(VirtualUse::Value(x));
                let y = self.get_scev(VirtualUse::Value(y));
                return Some(self.create_mod_expr(&x, &y));
            }
            Some(DivPattern::DivConst(x, y)) => {
                let x = self.get_scev(VirtualUse::Value(x));
                let y = SCEVExprRef::Constant(APConstant::ConstantInt(y));
                return Some(self.create_div_expr(&x, &y));
            }
            _ => {}
        }
        let du = self.def_use;
        match v {
            Value::Argument(arg_idx) if self.implicit_loop_info.is_some() => {
                self.create_scev_for_implicit_loop_argument(*arg_idx)
            }
            Value::Instruction(inst_idx, op_idx) => {
                let ir_inst = self.def_use.func.instructions[*inst_idx].clone().wrap()?;
                match ir_inst {
                    IRInstruction::BinOpInst(binst) => {
                        let desc = binst.get_desc();
                        if *op_idx != desc.dst_idx {
                            return None;
                        }
                        let lhs =
                            du.get_def_use(InstructionUse::op(*inst_idx, desc.lhs_op_idx, 0))?;
                        let rhs =
                            du.get_def_use(InstructionUse::op(*inst_idx, desc.rhs_op_idx, 0))?;
                        let l = self.get_scev(VirtualUse::Value(lhs));
                        let r = self.get_scev(VirtualUse::Value(rhs));
                        self.maybe_binary_op_scev(desc.op, &l, &r)
                    }
                    IRInstruction::TernaryOpInst(ternary_inst) => {
                        let desc = ternary_inst.get_desc();
                        if *op_idx != desc.dst_idx {
                            return None;
                        }
                        let u0 = du.get_def_use(InstructionUse::op(*inst_idx, desc.src0, 0))?;
                        let src0 = self.get_scev(VirtualUse::Value(u0));
                        [(desc.op0, desc.src1), (desc.op1, desc.src2)]
                            .into_iter()
                            .try_fold(src0, |r, (op, src)| {
                                let u = du.get_def_use(InstructionUse::op(*inst_idx, src, 0))?;
                                let scev = self.get_scev(VirtualUse::Value(u));
                                self.maybe_binary_op_scev(op, &r, &scev)
                            })
                    }
                    IRInstruction::SelectInst(select) => {
                        let desc = select.get_desc();
                        if *op_idx != desc.dst_idx {
                            return None;
                        }
                        let src0 =
                            du.get_def_use(InstructionUse::op(*inst_idx, desc.src0_idx, 0))?;
                        let src1 =
                            du.get_def_use(InstructionUse::op(*inst_idx, desc.src1_idx, 0))?;
                        let src0 = self.get_scev(VirtualUse::Value(src0));
                        let src1 = self.get_scev(VirtualUse::Value(src1));
                        Some(Self::create_select_expr(&src0, &src1))
                    }
                    _ => None,
                }
            }
            Value::Phi(phi_idx) => self.create_node_for_phi(*phi_idx),
            Value::Constant(x) => Some(SCEVExprRef::Constant(*x)),
            _ => None,
        }
    }

    fn create_scev_value_group(&self, v: &[Value]) -> Option<SCEVExprRef> {
        if v.len() != 2 {
            return None;
        }
        let func = self.def_use.func;
        let du = self.def_use;
        match GroupPattern::detect_pattern(func, self.def_use, v)? {
            GroupPattern::AddSubU64(
                op,
                [WideOperand {
                    inst_idx: i0_idx,
                    lhs_op_idx: lo0,
                    rhs_op_idx: lo1,
                }, WideOperand {
                    inst_idx: i1_idx,
                    lhs_op_idx: hi0,
                    rhs_op_idx: hi1,
                }],
            ) => {
                // 64-bit add
                let lhs = vec![
                    du.get_def_use(InstructionUse::op(i0_idx, lo0, 0))?,
                    du.get_def_use(InstructionUse::op(i1_idx, hi0, 0))?,
                ];
                let rhs = vec![
                    du.get_def_use(InstructionUse::op(i0_idx, lo1, 0))?,
                    du.get_def_use(InstructionUse::op(i1_idx, hi1, 0))?,
                ];
                let l = self.get_scev(VirtualUse::Group(lhs));
                let r = self.get_scev(VirtualUse::Group(rhs));
                if op == BinaryOperator::Add {
                    Some(self.create_add_expr(&l, &r))
                } else {
                    Some(self.create_sub_expr(&l, &r))
                }
            }
            GroupPattern::Mul32U64(WideOperand {
                inst_idx: i0_idx,
                lhs_op_idx: lo0,
                rhs_op_idx: lo1,
            }) => {
                let (lhs, rhs) = (
                    du.get_def_use(InstructionUse::op(i0_idx, lo0, 0))?,
                    du.get_def_use(InstructionUse::op(i0_idx, lo1, 0))?,
                );
                let (l, r) = (
                    self.get_scev(VirtualUse::Value(lhs)),
                    self.get_scev(VirtualUse::Value(rhs)),
                );
                Some(self.create_mul_expr(&l, &r))
            }
            GroupPattern::Mad32U64(
                WideOperand {
                    inst_idx: i_idx,
                    lhs_op_idx: src0_idx,
                    rhs_op_idx: src1_idx,
                },
                src2_idx,
            ) => {
                let a = du.get_def_use(InstructionUse::op(i_idx, src0_idx, 0))?;
                let b = du.get_def_use(InstructionUse::op(i_idx, src1_idx, 0))?;
                let a = self.get_scev(VirtualUse::Value(a));
                let b = self.get_scev(VirtualUse::Value(b));
                let mul = self.create_mul_expr(&a, &b);

                let c_lo = du.get_def_use(InstructionUse::op(i_idx, src2_idx, 0))?;
                let c_hi = du.get_def_use(InstructionUse::op(i_idx, src2_idx, 1))?;
                let c = self.get_scev(VirtualUse::Group(vec![c_lo, c_hi]));
                Some(self.create_add_expr(&mul, &c))
            }
            GroupPattern::ShiftU64(op, i_idx, lhs_idx, rhs_idx) => {
                let lhs = vec![
                    du.get_def_use(InstructionUse::op(i_idx, lhs_idx, 0))?,
                    du.get_def_use(InstructionUse::op(i_idx, lhs_idx, 1))?,
                ];
                let rhs = du.get_def_use(InstructionUse::op(i_idx, rhs_idx, 0))?;
                let l = self.get_scev(VirtualUse::Group(lhs));
                let r = self.get_scev(VirtualUse::Value(rhs));
                if op == BinaryOperator::LShift {
                    Some(self.create_shl_expr(&l, &r))
                } else {
                    // XXX: Not exactly right for signedness?
                    Some(self.create_ashr_expr(&l, &r))
                }
            }
            GroupPattern::ZExt(value_lo) => {
                let scev = self.get_scev(VirtualUse::Value(value_lo));
                Some(self.create_extend_expr(false, &scev))
            }
            GroupPattern::SExt(value_lo) => {
                let scev = self.get_scev(VirtualUse::Value(value_lo));
                Some(self.create_extend_expr(true, &scev))
            }
            GroupPattern::Shl32(value_lo) => {
                let scev = self.get_scev(VirtualUse::Value(value_lo));
                Some(
                    self.create_shl_expr(
                        &scev,
                        &SCEVExprRef::Constant(APConstant::ConstantInt(32)),
                    ),
                )
            }
            GroupPattern::PtrMask(v_lo, v_hi) => Some(self.create_and_expr(
                &SCEVExprRef::Constant(APConstant::ConstantInt((1 << 48) - 1)),
                &self.get_scev(VirtualUse::Group(vec![v_lo, v_hi])),
            )),
            GroupPattern::Phi(p0_idx, p1_idx) => self.create_node_for_phi_group(&[p0_idx, p1_idx]),
            GroupPattern::Constant(c) => Some(SCEVExprRef::new_constant(c)),
            GroupPattern::KernelArgumentBase => Some(SCEVExprRef::KernelArgumentBase),
            GroupPattern::DispatchPacketBase => Some(SCEVExprRef::DispatchPacketBase),
        }
    }

    fn create_add_rec_from_phi(&self, phi_idx: usize) -> Option<SCEVExprRef> {
        let p = &self.def_use.phis[phi_idx];
        let l = self.la.get_loop_for(p.bb_idx)?;
        if l.get_header() != Some(p.bb_idx) {
            return None;
        }

        let (s, e) = Self::is_representable_by_addrec(l, p)?;
        self.create_simple_affine_add_rec(l, p, &s, &e)
            .or_else(|| self.create_affine_add_rec(l, phi_idx, &s, &e))
    }

    fn create_node_from_select_like_phi(&self, phi_idx: usize) -> Option<SCEVExprRef> {
        let p = &self.def_use.phis[phi_idx];
        if p.values.len() != 2
            || !p
                .values
                .iter()
                .all(|(bb_idx, _)| self.dom.is_reachable_from_entry(*bb_idx))
        {
            return None;
        }

        let l = self.la.get_loop_for(p.bb_idx);

        // We don't want to break LCSSA, even in a SCEV expression tree.
        if p.values
            .iter()
            .any(|(bb_idx, _)| self.la.get_loop_for(*bb_idx).map(|l| l.id) != l.map(|l| l.id))
        {
            return None;
        }

        if !p
            .values
            .iter()
            .any(|(bb_idx, _)| self.dom.dominates(*bb_idx, p.bb_idx))
        {
            return None;
        }

        let lhs = self.get_scev(VirtualUse::Value(*p.values[0].1));
        let rhs = self.get_scev(VirtualUse::Value(*p.values[1].1));
        Some(Self::create_select_expr(&lhs, &rhs))
    }

    fn create_node_from_select_like_phi_group(&self, phi_idxs: &[usize; 2]) -> Option<SCEVExprRef> {
        let phis = [
            &self.def_use.phis[phi_idxs[0]],
            &self.def_use.phis[phi_idxs[1]],
        ];
        if phis[0].bb_idx != phis[1].bb_idx
            || phis[0].values.len() != phis[1].values.len()
            || phis[0].values.len() != 2
        {
            return None;
        }

        let l = self.la.get_loop_for(phis[0].bb_idx);
        // We don't want to break LCSSA, even in a SCEV expression tree.
        for p in phis.iter() {
            if p.values
                .iter()
                .any(|(bb_idx, _)| self.la.get_loop_for(*bb_idx).map(|l| l.id) != l.map(|l| l.id))
            {
                return None;
            }

            if !p
                .values
                .iter()
                .any(|(bb_idx, _)| self.dom.dominates(*bb_idx, p.bb_idx))
            {
                return None;
            }
        }

        let lhs = self.get_scev(VirtualUse::Group(vec![
            *phis[0].values[0].1,
            *phis[1].values[0].1,
        ]));
        let rhs = self.get_scev(VirtualUse::Group(vec![
            *phis[0].values[1].1,
            *phis[1].values[1].1,
        ]));
        Some(Self::create_select_expr(&lhs, &rhs))
    }

    fn create_node_for_phi(&self, phi_idx: usize) -> Option<SCEVExprRef> {
        if let Some(add_rec) = self.create_add_rec_from_phi(phi_idx) {
            return Some(add_rec);
        }
        if let Some(select) = self.create_node_from_select_like_phi(phi_idx) {
            return Some(select);
        }
        None
    }

    fn create_add_rec_from_phi_group(&self, phi_idxs: &[usize; 2]) -> Option<SCEVExprRef> {
        let p = &[
            &self.def_use.phis[phi_idxs[0]],
            &self.def_use.phis[phi_idxs[1]],
        ];
        if p[0].bb_idx != p[1].bb_idx
            || p[0].values.len() != p[1].values.len()
            || p[0].values.len() != 2
        {
            return None;
        }
        let l = self.la.get_loop_for(p[0].bb_idx)?;
        if l.get_header() != Some(p[0].bb_idx) {
            return None;
        }

        // For now we only support the case of create_simple_affine_addrec
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        p.iter().try_for_each(|p| {
            let (s, e) = Self::is_representable_by_addrec(l, p)?;
            starts.push(s);
            ends.push(e);
            Some(())
        })?;
        self.create_simple_affine_add_rec_group(l, p, starts.as_slice(), ends.as_slice())
            .or_else(|| {
                self.create_affine_add_rec_group(l, phi_idxs, starts.as_slice(), ends.as_slice())
            })
    }

    fn create_node_for_phi_group(&self, phi_idxs: &[usize; 2]) -> Option<SCEVExprRef> {
        // according to https://github.com/llvm/llvm-project/blob/f49e0c8d818a863ab70d02eb172ee79c58cf8f8a/llvm/lib/Analysis/ScalarEvolution.cpp#L6032
        if let Some(add_rec) = self.create_add_rec_from_phi_group(phi_idxs) {
            return Some(add_rec);
        }
        if let Some(select) = self.create_node_from_select_like_phi_group(phi_idxs) {
            return Some(select);
        }
        None
    }

    fn is_representable_by_addrec(l: &Loop, p: &PHI) -> Option<(Value, Value)> {
        // For now we only support the case of create_simple_affine_addrec
        let mut start_set = HashSet::new();
        let mut end_set = HashSet::new();
        for (bb_idx, v) in &p.values {
            if l.contains(*bb_idx) {
                end_set.insert(*v.as_ref());
            } else {
                start_set.insert(*v.as_ref());
            }
        }
        if start_set.len() != 1 || end_set.len() != 1 {
            return None;
        }
        let start = start_set.into_iter().next().unwrap();
        let end = end_set.into_iter().next().unwrap();
        Some((start, end))
    }

    /// For the simplest case: PN = PHI(Start, OP(Self, LoopInvariant))
    fn create_simple_affine_add_rec(
        &self,
        l: &Loop,
        p: &PHI,
        start: &Value,
        end: &Value,
    ) -> Option<SCEVExprRef> {
        if p.values.len() != 2 {
            return None;
        }

        let is_matched_phi =
            |v| matches!(v, &Value::Phi(phi_idx) if self.def_use.phis[phi_idx] == *p);
        let get_acc = |phi, acc| {
            (is_matched_phi(phi) && l.is_loop_invariant(acc, self.def_use))
                .then(|| self.get_scev(VirtualUse::Value(*acc)))
        };

        let func = self.la.func;
        let du = self.def_use;
        let accum = match end {
            Value::Instruction(inst_idx, op_idx) => {
                let inst = func.instructions[*inst_idx].clone().wrap()?;
                let binst = inst.as_binary_op()?;
                let desc = binst.get_desc();
                if *op_idx != desc.dst_idx {
                    return None;
                }
                let lhs = du.get_def_use(InstructionUse::op(*inst_idx, desc.lhs_op_idx, 0))?;
                let rhs = du.get_def_use(InstructionUse::op(*inst_idx, desc.rhs_op_idx, 0))?;

                match desc.op {
                    BinaryOperator::Add => [(&lhs, &rhs), (&rhs, &lhs)]
                        .into_iter()
                        .filter_map(|(l, r)| get_acc(l, r))
                        .next(),
                    BinaryOperator::Sub => Some(self.create_neg_expr(&get_acc(&lhs, &rhs)?)),
                    _ => None,
                }
            }
            _ => None,
        }?;

        let start_scev = self.get_scev(VirtualUse::Value(*start));
        Some(self.create_add_rec_expr(&start_scev, &accum, l.id))
    }

    // This function tries to find an AddRec expression for the simplest (yet most
    // common) cases: PN = PHI(Start, OP(Self, LoopInvariant)).
    fn create_simple_affine_add_rec_group(
        &self,
        l: &Loop,
        phis: &[&PHI],
        starts: &[Value],
        ends: &[Value],
    ) -> Option<SCEVExprRef> {
        if phis.len() != 2 {
            return None;
        }
        let num_legs = phis[0].values.len();
        if num_legs != 2 || starts.len() != 2 || ends.len() != 2 {
            return None;
        }

        let func = self.la.func;
        let du = self.def_use;
        let accum = match GroupPattern::detect_pattern(func, du, ends)? {
            GroupPattern::AddSubU64(
                op,
                [WideOperand {
                    inst_idx: i0_idx,
                    lhs_op_idx: lhs_idx0,
                    rhs_op_idx: rhs_idx0,
                }, WideOperand {
                    inst_idx: i1_idx,
                    lhs_op_idx: lhs_idx1,
                    rhs_op_idx: rhs_idx1,
                }],
            ) if op == BinaryOperator::Add => {
                let (lo0, lo1, hi0, hi1) = (
                    du.get_def_use(InstructionUse::op(i0_idx, lhs_idx0, 0))?,
                    du.get_def_use(InstructionUse::op(i0_idx, rhs_idx0, 0))?,
                    du.get_def_use(InstructionUse::op(i1_idx, lhs_idx1, 0))?,
                    du.get_def_use(InstructionUse::op(i1_idx, rhs_idx1, 0))?,
                );
                let lhs = GroupPattern::detect_pattern(func, du, &[lo0, hi0]);
                let li_value = if let Some(GroupPattern::Phi(_, _)) = lhs {
                    [lo1, hi1]
                } else {
                    [lo0, hi0]
                };

                let mut invariant_args = Vec::new();
                for v in li_value {
                    if !l.is_loop_invariant(&v, self.def_use) {
                        return None;
                    }
                    invariant_args.push(v);
                }
                Some(self.get_scev(VirtualUse::Group(invariant_args)))
            }
            _ => None,
        }?;

        let start_scev = self.get_scev(VirtualUse::Group(starts.to_vec()));
        Some(self.create_add_rec_expr(&start_scev, &accum, l.id))
    }

    // General but slow method to create affine accum for AddRec
    fn create_affine_add_rec(
        &self,
        l: &Loop,
        phi_idx: usize,
        start: &Value,
        end: &Value,
    ) -> Option<SCEVExprRef> {
        self.create_affine_add_rec_group_impl(
            l,
            || VirtualUse::Value(Value::Phi(phi_idx)),
            || VirtualUse::Value(*start),
            || VirtualUse::Value(*end),
        )
    }

    fn create_affine_add_rec_group(
        &self,
        l: &Loop,
        phi_idxs: &[usize; 2],
        starts: &[Value],
        ends: &[Value],
    ) -> Option<SCEVExprRef> {
        self.create_affine_add_rec_group_impl(
            l,
            || VirtualUse::Group(vec![Value::Phi(phi_idxs[0]), Value::Phi(phi_idxs[1])]),
            || VirtualUse::Group(vec![starts[0], starts[1]]),
            || VirtualUse::Group(vec![ends[0], ends[1]]),
        )
    }

    fn find_accum(
        &self,
        l: &Loop,
        place_holder: &SCEVExprRef,
        get_start: impl Fn() -> VirtualUse,
        end_scev: &SCEVExprRef,
    ) -> Option<SCEVExprRef> {
        match end_scev {
            SCEVExprRef::Expr(SCEVExpr::Add(ops)) => {
                let addens = ops
                    .iter()
                    .filter(|v| *v != place_holder)
                    .cloned()
                    .collect::<Vec<SCEVExprRef>>();
                if addens.len() + 1 != ops.len()
                    || !addens.iter().all(|scev| scev.is_loop_invariant(l))
                {
                    return None;
                }
                let start_val = self.get_scev(get_start());
                let phi_scev = self.create_add_rec_expr(
                    &start_val,
                    &self.create_add_expr_impl(addens.as_slice()),
                    l.id,
                );
                Some(phi_scev)
            }
            _ => None,
        }
    }

    fn create_affine_add_rec_group_impl(
        &self,
        l: &Loop,
        phis: impl Fn() -> VirtualUse,
        starts: impl Fn() -> VirtualUse,
        ends: impl Fn() -> VirtualUse,
    ) -> Option<SCEVExprRef> {
        // stash current context
        self.ctx.borrow_mut().push();
        // Handle PHI node value symbolically.
        let place_holder = self.create_unknown();
        self.ctx.borrow_mut().insert(phis(), place_holder.clone());

        // Using this symbolic name for the PHI, analyze the value coming around
        // the back-edge.
        let end_scev = self.get_scev(ends());

        // If the value coming around the backedge is an add with the symbolic
        // value we just inserted, then we found a simple induction variable!
        let accum = self.find_accum(l, &place_holder, starts, &end_scev);

        // Remove the temporary PHI node SCEV that has been inserted while intending
        // to create an AddRecExpr for this PHI node. We can not keep this temporary
        // as it will prevent later (possibly simpler) SCEV expressions to be added
        // to the ValueExprMap.

        // restore the cache from the stashed context.
        self.ctx.borrow_mut().pop();
        accum
    }

    fn create_scev_for_implicit_loop_argument(&self, arg_idx: usize) -> Option<SCEVExprRef> {
        ImplicitLoopInfo::ARGUMENT_DESCRIPTORS
            .iter()
            .find(|(arg_id, _)| *arg_id == arg_idx)
            .map(|(_, lid)| {
                self.create_add_rec_expr(
                    &SCEVExpr::create_const_int(0),
                    &SCEVExpr::create_const_int(1),
                    *lid,
                )
            })
    }

    fn create_scev_for_vmem_addr(&self, inst_idx: usize) -> Option<SCEVExprRef> {
        let func = self.la.func;
        let inst = &func.instructions[inst_idx];
        let du = self.def_use;
        let addr_info = match Instruction::wrap(inst.clone())? {
            IRInstruction::StoreInst(si) => si.get_addr(inst_idx, du),
            IRInstruction::LoadInst(li) => li.get_addr(inst_idx, du),
            IRInstruction::AtomicInst(ai) => ai.get_addr(inst_idx, du),
            _ => None,
        }?;

        let addr_scev = self.get_scev(addr_info.base);
        let real_addr = [addr_info.reg_offset, addr_info.imm_offset]
            .into_iter()
            .fold(addr_scev, |l, r| {
                if let Some(v) = r {
                    // Address = sdata (64-bit) + addr (32-bit)
                    let off = self.get_scev(v);
                    let zext = self.create_extend_expr(false, &off);
                    self.create_add_expr(&l, &zext)
                } else {
                    l
                }
            });
        Some(real_addr)
    }

    fn fold_expressions<'c>(
        &self,
        ops: impl Iterator<Item = &'c SCEVExprRef>,
        should_recurse: fn(&ScalarEvolution, &SCEVExprRef) -> Option<Vec<SCEVExprRef>>,
    ) -> Vec<SCEVExprRef> {
        // fold add expressions
        let mut ret = Vec::new();
        ops.for_each(|o| {
            if let Some(x) = should_recurse(self, o) {
                ret.extend(x.iter().cloned());
            } else {
                ret.push(o.clone())
            }
        });
        // sort by complexity to group similar expression types together
        ret.sort_by(|a, b| SCEVExprRef::partial_cmp(a, b).unwrap_or(Ordering::Equal));
        ret
    }

    pub(crate) fn create_add_expr_impl(&self, addens: &[SCEVExprRef]) -> SCEVExprRef {
        assert!(!addens.is_empty());
        let mut ops = self.fold_expressions(addens.iter(), |se, expr| match expr.get(se) {
            Some(SCEVExpr::Add(x)) => Some(x),
            _ => None,
        });

        if let Some(x) = ops.get(0).and_then(|x| x.as_int()) {
            while let Some(y) = ops.get(1).and_then(|y| y.as_int()) {
                // found two constants, fold them together
                ops.remove(1);
                ops[0] = SCEVExpr::create_const_int(x + y);
            }
            // if the constant is 0, strip it out
            if ops[0].as_int() == Some(0) {
                ops.remove(0);
            }
        }
        match ops.len() {
            0 => SCEVExpr::create_const_int(0),
            1 => ops.remove(0),
            _ => SCEVExprRef::new(SCEVExpr::Add(ops.into_iter().collect())),
        }
    }

    pub fn create_add_expr(&self, lhs: &SCEVExprRef, rhs: &SCEVExprRef) -> SCEVExprRef {
        self.create_add_expr_impl([lhs.clone(), rhs.clone()].as_slice())
    }

    pub(crate) fn create_mul_expr(&self, lhs: &SCEVExprRef, rhs: &SCEVExprRef) -> SCEVExprRef {
        if [lhs, rhs].iter().any(|op| op.as_int() == Some(0)) {
            return SCEVExpr::create_const_int(0);
        }
        let mut ops =
            self.fold_expressions([lhs, rhs].into_iter(), |se, expr| match expr.get(se) {
                Some(SCEVExpr::Mul(x)) => Some(x),
                _ => None,
            });

        // fold constants
        if let Some(x) = ops.get(0).and_then(|x| x.as_int()) {
            while let Some(y) = ops.get(1).and_then(|y| y.as_int()) {
                // found two constants, fold them together
                ops.remove(1);
                ops[0] = SCEVExpr::create_const_int(x * y);
            }
            // if the constant is 1, strip it out
            if ops[0].as_int() == Some(1) {
                ops.remove(0);
            }
        }
        match ops.len() {
            0 => SCEVExpr::create_const_int(1),
            1 => ops.remove(0),
            _ => SCEVExprRef::new(SCEVExpr::Mul(ops.into_iter().collect())),
        }
    }

    fn create_div_expr(&self, lhs: &SCEVExprRef, rhs: &SCEVExprRef) -> SCEVExprRef {
        SCEVExprRef::Expr(SCEVExpr::Div(vec![lhs.clone(), rhs.clone()]))
    }

    fn create_mod_expr(&self, lhs: &SCEVExprRef, rhs: &SCEVExprRef) -> SCEVExprRef {
        SCEVExprRef::Expr(SCEVExpr::Mod(vec![lhs.clone(), rhs.clone()]))
    }

    fn create_shl_expr(&self, lhs: &SCEVExprRef, rhs: &SCEVExprRef) -> SCEVExprRef {
        if let Some(shift) = rhs.as_int() {
            return self.create_mul_expr(&SCEVExpr::create_const_int(1 << shift), lhs);
        }
        SCEVExprRef::Expr(SCEVExpr::Shl(vec![lhs.clone(), rhs.clone()]))
    }

    pub(crate) fn create_ashr_expr(&self, lhs: &SCEVExprRef, rhs: &SCEVExprRef) -> SCEVExprRef {
        SCEVExprRef::Expr(SCEVExpr::AShr(vec![lhs.clone(), rhs.clone()]))
    }

    fn create_sub_expr(&self, lhs: &SCEVExprRef, rhs: &SCEVExprRef) -> SCEVExprRef {
        if let SCEVExprRef::Expr(SCEVExpr::Mul(rhs)) = rhs {
            if rhs.len() == 2 {
                if let SCEVExprRef::Expr(SCEVExpr::Div(v)) = &rhs[0] {
                    if v.len() == 2 && v[0] == *lhs && v[1] == rhs[1] {
                        // x - (x / y) * y ==> x % y
                        return self.create_mod_expr(lhs, &rhs[1]);
                    }
                }
                if let SCEVExprRef::Expr(SCEVExpr::Div(v)) = &rhs[1] {
                    if v.len() == 2 && v[0] == *lhs && v[1] == rhs[0] {
                        // x - y * (x / y) ==> x % y
                        return self.create_mod_expr(lhs, &rhs[0]);
                    }
                }
            }
        }
        let rhs_opp = self.create_neg_expr(rhs);
        self.create_add_expr(lhs, &rhs_opp)
    }

    fn create_neg_expr(&self, expr: &SCEVExprRef) -> SCEVExprRef {
        match expr {
            SCEVExprRef::Constant(x) => SCEVExprRef::Constant(x.opposite()),
            _ => self.create_mul_expr(&SCEVExpr::create_const_int(-1), expr),
        }
    }

    pub(crate) fn create_and_expr(&self, lhs: &SCEVExprRef, rhs: &SCEVExprRef) -> SCEVExprRef {
        if [lhs, rhs].iter().any(|op| op.as_int() == Some(0)) {
            return SCEVExpr::create_const_int(0);
        }
        if let (Some(l), Some(r)) = (lhs.as_int(), rhs.as_int()) {
            return SCEVExpr::create_const_int(l & r);
        }
        let ops = self.fold_expressions([lhs, rhs].into_iter(), |se, expr| match expr.get(se) {
            Some(SCEVExpr::And(x)) => Some(x),
            _ => None,
        });
        SCEVExprRef::new(SCEVExpr::And(ops.into_iter().collect()))
    }

    pub(crate) fn create_or_expr(&self, lhs: &SCEVExprRef, rhs: &SCEVExprRef) -> SCEVExprRef {
        if let (Some(l), Some(r)) = (lhs.as_int(), rhs.as_int()) {
            return SCEVExpr::create_const_int(l | r);
        }
        let ops = self.fold_expressions([lhs, rhs].into_iter(), |se, expr| match expr.get(se) {
            Some(SCEVExpr::Or(x)) => Some(x),
            _ => None,
        });
        SCEVExprRef::new(SCEVExpr::Or(ops.into_iter().collect()))
    }

    fn create_xor_expr(&self, lhs: &SCEVExprRef, rhs: &SCEVExprRef) -> SCEVExprRef {
        if let (Some(l), Some(r)) = (lhs.as_int(), rhs.as_int()) {
            return SCEVExpr::create_const_int(l ^ r);
        }
        let ops = self.fold_expressions([lhs, rhs].into_iter(), |se, expr| match expr.get(se) {
            Some(SCEVExpr::Xor(x)) => Some(x),
            _ => None,
        });
        SCEVExprRef::new(SCEVExpr::Xor(ops.into_iter().collect()))
    }

    pub(crate) fn create_extend_expr(&self, signed: bool, expr: &SCEVExprRef) -> SCEVExprRef {
        if let Some(x) = expr.as_int() {
            SCEVExpr::create_const_int(x)
        } else if signed {
            SCEVExprRef::new(SCEVExpr::SExt(Box::new(expr.clone())))
        } else {
            SCEVExprRef::new(SCEVExpr::ZExt(Box::new(expr.clone())))
        }
    }

    pub(crate) fn create_select_expr(lhs: &SCEVExprRef, rhs: &SCEVExprRef) -> SCEVExprRef {
        SCEVExprRef::new(SCEVExpr::Select(Select {
            selector: None,
            operands: vec![lhs.clone(), rhs.clone()],
        }))
    }

    pub(crate) fn create_add_rec_expr(
        &self,
        start: &SCEVExprRef,
        accum: &SCEVExprRef,
        loop_id: usize,
    ) -> SCEVExprRef {
        SCEVExprRef::new(SCEVExpr::AddRec(AddRec {
            operands: vec![start.clone(), accum.clone()],
            loop_info_id: loop_id,
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::analysis::scalar_evolution::{SCEVExpr, ScalarEvolution, VirtualUse};
    use crate::analysis::{
        ConstantPropagation, DomFrontier, InstructionUse, LoopAnalysis, PHIAnalysis, SCEVExprRef,
    };
    use crate::fileformat::{Disassembler, KernelInfo};
    use crate::ir::{APConstant, DomTree, Function, Value};
    use crate::tests::cfg::simple_kernel_info;
    use crate::tests::scev::test_with_dummy_scev;

    fn test_expr_in_code(code: &[u32], f: fn(&ScalarEvolution) -> ()) {
        let ki = simple_kernel_info("", code);
        let func = Disassembler::parse_kernel(&ki).expect("Failed to parse the function");
        test_expr_in_kernel(&ki, &func, f)
    }

    fn test_expr_in_kernel(ki: &KernelInfo, func: &Function, f: fn(&ScalarEvolution) -> ()) {
        let dom = DomTree::analyze(func);
        let df = DomFrontier::new(&dom);
        let def_use = PHIAnalysis::analyze(&func, &dom, &df, &ki).expect("Cannot analyze PHI");
        let def_use = ConstantPropagation::run(def_use);
        let dom = DomTree::analyze(&func);
        let mut li = LoopAnalysis::new(&func);
        li.analyze(&dom);
        let scev = ScalarEvolution::new(&dom, &def_use, &li, None);
        f(&scev)
    }

    #[test]
    fn test_scev() {
        use SCEVExpr::*;
        let (ki, func) = crate::tests::cfg::nested_loop();
        test_expr_in_kernel(ki, &func, |se| {
            let expr = se.get_scev(VirtualUse::Address(11));
            if let Some(Add(x)) = expr.get(se) {
                assert!(matches!(x[0].get(se), Some(AddRec(_))));
                assert!(matches!(x[1].get(se), Some(AddRec(_))));
            } else {
                assert!(false);
            }
        });
    }

    #[test]
    fn test_mul() {
        use SCEVExpr::*;
        const CODE: &[u32] = &[
            0xBE800380, // s_mov_b32 s0, 0
            0x93018200, // s_mul_i32 s1, s0, 2
            0x80008100, // s_add_u32 s0, s0, 1
            0xBF078800, // s_cmp_lg_u32 s0, 8
            0xBF85FFFC, // s_cbranch_scc1 65532
            0x9A811104, // s_mul_hi_u32 s1, s4, s17
            0x93021104, // s_mul_i32 s2, s4, s17
            0xBF810000, // s_endpgm
        ];

        test_expr_in_code(CODE, |se| {
            let expr = se.get_scev(VirtualUse::Value(Value::Instruction(1, 0)));
            if let Some(Mul(x)) = expr.get(se) {
                assert!(matches!(
                    x[0],
                    SCEVExprRef::Constant(APConstant::ConstantInt(2))
                ));
                assert!(matches!(x[1].get(se), Some(AddRec(_))));
            } else {
                assert!(false);
            }

            let expr = se.get_scev(VirtualUse::Group(vec![
                Value::Instruction(6, 0),
                Value::Instruction(5, 0),
            ]));
            assert!(matches!(expr.get(se), Some(Mul(_))));
        });
    }

    #[test]
    fn test_fold_constant_mul() {
        test_with_dummy_scev(|se| {
            assert_eq!(
                se.create_mul_expr(
                    &SCEVExpr::create_const_int(2),
                    &SCEVExpr::create_const_int(3)
                ),
                SCEVExpr::create_const_int(6)
            );
        })
    }

    #[test]
    fn test_shift() {
        use SCEVExpr::*;
        const CODE: &[u32] = &[
            0xBE800380, // s_mov_b32 s0, 0
            0x8F018200, // s_lshl_b32 s1, s0, 2
            0x91028200, // s_ashr_i32 s2, s0, 2
            0x80008100, // s_add_u32 s0, s0, 1
            0xBF078800, // s_cmp_lg_u32 s0, 8
            0xBF85FFFB, // s_cbranch_scc1 65531
            0x8F828204, // s_lshl_b64 s[2:3], s[4:5], 2
            0x908A8204, // s_lshr_b64 s[10:11], s[4:5], 2
            0xBF810000, // s_endpgm
        ];
        test_expr_in_code(CODE, |se| {
            let expr = se.get_scev(VirtualUse::Value(Value::Instruction(1, 0)));
            if let Some(Mul(x)) = expr.get(se) {
                assert!(matches!(
                    x[0],
                    SCEVExprRef::Constant(APConstant::ConstantInt(4))
                ));
                assert!(matches!(x[1].get(se), Some(AddRec(_))));
            } else {
                assert!(false);
            }

            let expr = se.get_scev(VirtualUse::Value(Value::Instruction(2, 0)));
            if let Some(AShr(x)) = expr.get(se) {
                assert!(matches!(x[0].get(se), Some(AddRec(_))));
                assert!(matches!(
                    x[1],
                    SCEVExprRef::Constant(APConstant::ConstantInt(2))
                ));
            } else {
                assert!(false);
            }

            let expr = se.get_scev(VirtualUse::Group(vec![
                Value::Instruction(6, 0),
                Value::Instruction(6, 1),
            ]));
            if let Some(Mul(x)) = expr.get(se) {
                assert!(matches!(
                    x[0],
                    SCEVExprRef::Constant(APConstant::ConstantInt(4))
                ));
                assert!(matches!(x[1], SCEVExprRef::KernelArgumentBase));
            } else {
                assert!(false);
            }

            let expr = se.get_scev(VirtualUse::Group(vec![
                Value::Instruction(7, 0),
                Value::Instruction(7, 1),
            ]));
            if let Some(AShr(x)) = expr.get(se) {
                assert!(matches!(x[0], SCEVExprRef::KernelArgumentBase));
                assert!(matches!(
                    x[1],
                    SCEVExprRef::Constant(APConstant::ConstantInt(2))
                ));
            } else {
                assert!(false);
            }
        });
    }

    #[test]
    fn test_lshl_add() {
        use SCEVExpr::*;
        let (ki, func) = crate::tests::cfg::clear_cp();
        test_expr_in_kernel(ki, &func, |se| {
            let expr = se.get_scev(VirtualUse::Value(Value::Instruction(2, 0)));
            if let Some(Add(x)) = expr.get(se) {
                assert_eq!(x.len(), 2);
                if let Some(Mul(x)) = x[0].get(&se) {
                    assert_eq!(x.len(), 2);
                    assert!(matches!(
                        x[0],
                        SCEVExprRef::Constant(APConstant::ConstantInt(256))
                    ));
                } else {
                    assert!(false);
                }
            } else {
                assert!(false);
            }
        });
    }

    #[test]
    fn test_fold_constant_addition() {
        use SCEVExpr::*;
        let (ki, func) = crate::tests::cfg::double_step_loop();
        test_expr_in_kernel(ki, &func, |se| {
            let expr = se.get_scev(VirtualUse::Value(Value::Instruction(2, 0)));
            if let Some(Add(x)) = expr.get(se) {
                assert!(matches!(
                    x[0],
                    SCEVExprRef::Constant(APConstant::ConstantInt(2))
                ));
            } else {
                assert!(false);
            }
        });
    }

    #[test]
    fn test_addr_scev() {
        // v_mov_b32_e32 v0, 18                                       // 000000001000: 7E000292
        // v_mov_b32_e32 v1, 52                                       // 000000001004: 7E0202B4
        // s_mov_b32 s0, 0x56                                         // 000000001008: BE8003FF 00000056
        // s_mov_b32 s1, 0x78                                         // 000000001010: BE8103FF 00000078
        // global_load_dword v3, v0, s[0:1]                           // 000000001018: DC308000 03000000
        // global_load_dword v3, v[0:1], off offset:4                 // 000000001020: DC308004 037D0000
        // global_store_dword v0, v3, s[0:1] offset:8                 // 000000001028: DC708008 00000300
        // global_store_dword v[0:1], v3, off offset:12               // 000000001030: DC70800C 007D0300
        // global_atomic_add v19, v0, v19, s[0:1] glc                 // 000000001038: DCC98000 13001300
        // global_atomic_cmpswap_x2 v[6:7], v[0:1], v[2:5], off offset:24 glc// 000000001040: DD458018 067D0200
        // s_endpgm                                                   // 000000001048: BF810000
        const CODE: &[u32] = &[
            0x7E000292, 0x7E0202B4, 0xBE8003FF, 0x00000056, 0xBE8103FF, 0x00000078, 0xDC308000,
            0x03000000, 0xDC308004, 0x037D0000, 0xDC708008, 0x00000300, 0xDC70800C, 0x007D0300,
            0xDCC98000, 0x13001300, 0xDD458018, 0x067D0200, 0xBF810000,
        ];
        const ADDRS: &[(usize, isize)] = &[
            (4, 0x7800000068),
            (5, 0x3400000012 + 4),
            (6, 0x7800000068 + 8),
            (7, 0x3400000012 + 12),
            (8, 0x7800000068),
            (9, 0x3400000012 + 24),
        ];
        test_expr_in_code(CODE, |scev| {
            for (idx, addr) in ADDRS {
                let expr = scev.get_scev(VirtualUse::Address(*idx));
                assert_eq!(expr, SCEVExprRef::Constant(APConstant::ConstantInt(*addr)));
            }
        });
    }

    #[test]
    fn test_cselect() {
        use SCEVExpr::*;
        // s_mov_b32 s0, 0                                            // 000000001000: BE800380
        // s_cselect_b32 s1, s0, 0                                    // 000000001004: 85018000
        // s_add_u32 s0, s0, 1                                        // 000000001008: 80008100
        // s_cmp_lg_u32 s0, 8                                         // 00000000100C: BF078800
        // s_cbranch_scc1 65532                                       // 000000001010: BF85FFFC <f+0x4>
        // s_endpgm                                                   // 000000001014: BF810000
        const CODE: &[u32] = &[
            0xBE800380, 0x85018000, 0x80008100, 0xBF078800, 0xBF85FFFC, 0xBF810000,
        ];
        test_expr_in_code(CODE, |scev| {
            let expr = scev.get_scev(VirtualUse::Value(Value::Instruction(1, 0)));
            if let Some(Select(x)) = expr.get(scev) {
                assert_eq!(x.operands.len(), 2);
                assert!(matches!(x.operands[0].get(scev), Some(AddRec(_))));
                assert!(matches!(
                    x.operands[1],
                    SCEVExprRef::Constant(APConstant::ConstantInt(0))
                ));
            } else {
                assert!(false);
            }
        });
    }

    #[test]
    fn test_lshl_or() {
        use SCEVExpr::*;
        const CODE: &[u32] = &[
            0xD76F0000, 0x04011006, // v_lshl_or_u32 v0, s6, 8, v0
            0xBF810000, // s_endpgm
        ];
        test_expr_in_code(CODE, |se| {
            let expr = se.get_scev(VirtualUse::Value(Value::Instruction(0, 0)));
            if let Some(Or(x)) = expr.get(se) {
                assert_eq!(x.len(), 2);
                if let Some(Mul(x)) = x[0].get(se) {
                    assert_eq!(x.len(), 2);
                    assert!(matches!(
                        x[0],
                        SCEVExprRef::Constant(APConstant::ConstantInt(256))
                    ));
                } else {
                    assert!(false);
                }
            } else {
                assert!(false);
            }
        });
    }

    #[test]
    fn test_sext() {
        // v_add_nc_u32_e32 v0, v0, 1
        // v_ashrrev_i32_e32 v1, 31, v0
        // s_endpgm
        const CODE: &[u32] = &[0x4A000081, 0x3002009F, 0xBF810000];
        test_expr_in_code(CODE, |scev| {
            let expr = scev.get_scev(VirtualUse::Group(vec![
                Value::Instruction(0, 0),
                Value::Instruction(1, 0),
            ]));
            assert!(matches!(expr.get(scev), Some(SCEVExpr::SExt(_))));
        });
    }

    #[test]
    fn test_simple_add64() {
        const CODE: &[u32] = &[
            0x80028404, // s_add_u32 s2, s4, 4
            0x82038005, // s_addc_u32 s3, s5, 0
            0xD70F6A00, 0x00020004, // v_add_co_u32 v0, vcc_lo, s4, v0
            0x50020205, // v_add_co_ci_u32_e32 v1, vcc_lo, s5, v1, vcc_lo
            0x80860080, // s_sub_u32 s6, 0, s0
            0x82878080, // s_subb_u32 s7, 0, 0
            0xBF810000, // s_endpgm
        ];
        test_expr_in_code(CODE, |scev| {
            match scev
                .get_scev(VirtualUse::Group(vec![
                    Value::Instruction(0, 0),
                    Value::Instruction(1, 0),
                ]))
                .get(scev)
            {
                Some(SCEVExpr::Add(ops)) => {
                    assert_eq!(ops.len(), 2);
                    assert!(matches!(
                        &ops[0],
                        SCEVExprRef::Constant(APConstant::ConstantInt(4))
                    ));
                    assert!(matches!(&ops[1], SCEVExprRef::KernelArgumentBase));
                }
                _ => assert!(false),
            }

            match scev
                .get_scev(VirtualUse::Group(vec![
                    Value::Instruction(2, 0),
                    Value::Instruction(3, 0),
                ]))
                .get(scev)
            {
                Some(SCEVExpr::Add(ops)) => {
                    assert_eq!(ops.len(), 2);
                    assert!(matches!(&ops[0], SCEVExprRef::KernelArgumentBase));
                }
                _ => assert!(false),
            }

            match scev
                .get_scev(VirtualUse::Group(vec![
                    Value::Instruction(4, 0),
                    Value::Instruction(5, 0),
                ]))
                .get(scev)
            {
                Some(SCEVExpr::Mul(v)) => {
                    assert!(matches!(
                        &v[0],
                        SCEVExprRef::Constant(APConstant::ConstantInt(-1))
                    ));
                }
                _ => assert!(false),
            }
        });
    }

    #[test]
    fn test_add64_swapped() {
        const CODE: &[u32] = &[
            0x80020484, // s_add_u32 s2, 4, s4
            0x82038005, // s_addc_u32 s3, s5, 0
            0x91059F04, // s_ashr_i32 s5, s4, 31
            0x80020484, // s_add_u32 s2, 4, s4
            0x82038005, // s_addc_u32 s3, s5, 0
            0x90848204, // s_lshr_b64 s[4:5], s[4:5], 2
            0x80020484, // s_add_u32 s2, 4, s4
            0x82038005, // s_addc_u32 s3, s5, 0
            0x93040302, // s_mul_i32 s4, s2, s3
            0x9A850302, // s_mul_hi_u32 s5, s2, s3
            0x80020484, // s_add_u32 s2, 4, s4
            0x82038005, // s_addc_u32 s3, s5, 0
            0xBF810000, //s_endpgm
        ];
        test_expr_in_code(CODE, |scev| {
            match scev
                .get_scev(VirtualUse::Group(vec![
                    Value::Instruction(0, 0),
                    Value::Instruction(1, 0),
                ]))
                .get(scev)
            {
                Some(SCEVExpr::Add(ops)) => {
                    assert_eq!(ops.len(), 2);
                    assert!(matches!(
                        &ops[0],
                        SCEVExprRef::Constant(APConstant::ConstantInt(4))
                    ));
                    assert!(matches!(&ops[1], SCEVExprRef::KernelArgumentBase));
                }
                _ => assert!(false),
            };
            match scev
                .get_scev(VirtualUse::Group(vec![
                    Value::Instruction(3, 0),
                    Value::Instruction(4, 0),
                ]))
                .get(scev)
            {
                Some(SCEVExpr::Add(ops)) => {
                    assert_eq!(ops.len(), 2);
                    assert!(matches!(
                        &ops[0],
                        SCEVExprRef::Constant(APConstant::ConstantInt(4))
                    ));
                    assert!(matches!(&ops[1], SCEVExprRef::Expr(SCEVExpr::SExt(_))));
                }
                _ => assert!(false),
            };
            match scev
                .get_scev(VirtualUse::Group(vec![
                    Value::Instruction(6, 0),
                    Value::Instruction(7, 0),
                ]))
                .get(scev)
            {
                Some(SCEVExpr::Add(ops)) => {
                    assert_eq!(ops.len(), 2);
                    assert!(matches!(
                        &ops[0],
                        SCEVExprRef::Constant(APConstant::ConstantInt(4))
                    ));
                    assert!(matches!(&ops[1], SCEVExprRef::Expr(SCEVExpr::AShr(_))));
                }
                _ => assert!(false),
            };
            match scev
                .get_scev(VirtualUse::Group(vec![
                    Value::Instruction(10, 0),
                    Value::Instruction(11, 0),
                ]))
                .get(scev)
            {
                Some(SCEVExpr::Add(ops)) => {
                    assert_eq!(ops.len(), 2);
                    assert!(matches!(
                        &ops[0],
                        SCEVExprRef::Constant(APConstant::ConstantInt(4))
                    ));
                    assert!(matches!(&ops[1], SCEVExprRef::Expr(SCEVExpr::Mul(_))));
                }
                _ => assert!(false),
            }
        });
    }

    #[test]
    fn test_add64_rec() {
        const CODE: &[u32] = &[
            0xBF800000, // s_nop 0
            0x80048404, // s_add_u32 s4, s4, 4
            0x82058005, // s_addc_u32 s5, s5, 0
            0xBF85FFFD, // s_cbranch_scc1 65533
            0xBF810000, // s_endpgm
        ];
        test_expr_in_code(CODE, |scev| {
            match scev
                .get_scev(VirtualUse::Group(vec![
                    Value::Instruction(1, 0),
                    Value::Instruction(2, 0),
                ]))
                .get(scev)
            {
                Some(SCEVExpr::Add(ops)) => {
                    assert_eq!(ops.len(), 2);
                    assert!(matches!(
                        &ops[0],
                        SCEVExprRef::Constant(APConstant::ConstantInt(4))
                    ));
                    assert!(matches!(ops[1].get(scev), Some(SCEVExpr::AddRec(_))));
                }
                _ => assert!(false),
            }
        });
    }

    #[test]
    fn test_add64_rec_swapped() {
        const CODE: &[u32] = &[
            0xBF800000, // s_nop 0
            0x80040484, // s_add_u32 s4, 4, s4
            0x82058005, // s_addc_u32 s5, s5, 0
            0xBF85FFFD, // s_cbranch_scc1 65533
            0xBF810000, // s_endpgm
        ];
        test_expr_in_code(CODE, |scev| {
            match scev
                .get_scev(VirtualUse::Group(vec![
                    Value::Instruction(1, 0),
                    Value::Instruction(2, 0),
                ]))
                .get(scev)
            {
                Some(SCEVExpr::Add(ops)) => {
                    assert_eq!(ops.len(), 2);
                    assert!(matches!(
                        &ops[0],
                        SCEVExprRef::Constant(APConstant::ConstantInt(4))
                    ));
                    assert!(matches!(ops[1].get(scev), Some(SCEVExpr::AddRec(_))));
                }
                _ => assert!(false),
            }
        });
    }

    #[test]
    fn test_min_max() {
        const CODE: &[u32] = &[
            0xBE800380, // s_mov_b32 s0 0
            0xBE810381, // s_mov_b32 s1 1
            0x83020100, // s_min_i32 s2 s0 s1
            0xBF810000, // s_endpgm
        ];
        test_expr_in_code(CODE, |scev| {
            match scev
                .get_scev(VirtualUse::Value(Value::Instruction(2, 0)))
                .get(scev)
            {
                Some(SCEVExpr::Select(ops)) => {
                    assert_eq!(
                        ops.operands,
                        vec![
                            SCEVExprRef::Constant(APConstant::ConstantInt(0)),
                            SCEVExprRef::Constant(APConstant::ConstantInt(1)),
                        ]
                    );
                }
                _ => assert!(false),
            };
        });
    }

    #[test]
    fn test_shl32() {
        const CODE: &[u32] = &[
            0x7e000280, // v_mov_b32_e32 v0, 0
            0x7e020301, // v_mov_b32_e32 v1, v1
            0xd7010000, 0x0002009e, // v_ashrrev_i64 v[0:1], 30, v[0:1]
            0xdc308000, 0x007d0000, // global_load_dword v0, v[0:1], off
            0xbf810000, // s_endpgm
        ];
        test_expr_in_code(CODE, |scev| {
            let group = (0..2)
                .filter_map(|offset| scev.def_use.get_def_use(InstructionUse::op(2, 2, offset)))
                .collect::<Vec<Value>>();
            assert_eq!(group.len(), 2);
            if let Some(SCEVExpr::Mul(x)) = scev.get_scev(VirtualUse::Group(group)).get(scev) {
                assert_eq!(x.len(), 2);
                assert_eq!(x[0].as_int(), Some(1 << 32));
            } else {
                assert!(false);
            }
            if let Some(SCEVExpr::AShr(x)) = scev.get_scev(VirtualUse::Address(3)).get(scev) {
                assert_eq!(x.len(), 2);
                assert_eq!(x[1].as_int(), Some(30));
                if let Some(SCEVExpr::Mul(x)) = x[0].get(scev) {
                    assert_eq!(x.len(), 2);
                    assert_eq!(x[0].as_int(), Some(1 << 32));
                } else {
                    assert!(false);
                }
            } else {
                assert!(false);
            }
        });
    }

    #[test]
    fn test_phi_to_select_expression() {
        const CODE: &[u32] = &[
            // BB0:
            0x7e000280, // v_mov_b32_e32 v0, 0
            0x7e020280, // v_mov_b32_e32 v1, 0
            0xbe800380, // s_mov_b32 s0, 0
            // BB1:
            0x80008100, // s_add_u32 s0, s0, 1
            0xbf078400, // s_cmp_lg_u32 s0, 4
            0xbf850002, // s_cbranch_scc1 BB3
            // BB2:
            0x7e000281, // v_mov_b32_e32 v0, 1
            0x7e020280, // v_mov_b32_e32 v1, 0
            // BB3:
            0xdc708000, 0x007d0000, // global_store_dword v[0:1], v0, off
            0xbf078800, // s_cmp_lg_u32 s0, 8
            0xbf85fff7, // s_cbranch_scc1 BB1
            // BB4:
            0xbf810000, // s_endpgm
        ];
        test_expr_in_code(CODE, |scev| {
            if let Some(SCEVExpr::Select(select)) = scev.get_scev(VirtualUse::Address(8)).get(scev)
            {
                assert_eq!(select.operands.len(), 2);
                assert!(matches!(select.operands[0], SCEVExprRef::Unknown(_)));
                assert_eq!(select.operands[1].as_int(), Some(1));
            } else {
                assert!(false);
            }
        });
    }

    #[test]
    fn test_mad() {
        // TODO: Need to update SCEV to bound the operands as 24-bit or 16-bit integers
        const CODE: &[u32] = &[
            0xbe800384, // s_mov_b32 s0, 4
            0xbe810385, // s_mov_b32 s1, 5
            0x7e020286, // v_mov_b32 v1, 6
            0xd5430000, 0x04040200, // v_mad_u32_u24 v0, s0, s1, v1
            0x16000200, // v_mul_u32_u24 v0, s0, v1
            0xbf810000, // s_endpgm
        ];
        test_expr_in_code(CODE, |se| {
            let expr = se.get_scev(VirtualUse::Value(Value::Instruction(3, 0)));
            assert_eq!(expr.as_int(), Some(26));
            let expr = se.get_scev(VirtualUse::Value(Value::Instruction(4, 0)));
            assert_eq!(expr.as_int(), Some(24));
        });
    }

    #[test]
    fn test_mad_32_64() {
        const CODE: &[u32] = &[
            0xbe8003ff, 0x00010001, // s_mov_b32 s0, 0x10001
            0xbe8103ff, 0x00010001, // s_mov_b32 s1, 0x10001
            0x7e000281, // v_mov_b32 v0, 1
            0x7e020281, // v_mov_b32 v1, 1
            0xd5766a00, 0x04000200, // v_mad_u64_u32 v[0:1], vcc_lo, s0, s1, v[0:1]
            0xbf810000, // s_endpgm
        ];
        test_expr_in_code(CODE, |se| {
            let expr = se.get_scev(VirtualUse::Group(vec![
                Value::Instruction(4, 0),
                Value::Instruction(4, 1),
            ]));
            assert_eq!(expr.as_int(), Some(0x200020002));
        });
    }

    #[test]
    fn test_generic_add_rec() {
        let (ki, func) = crate::tests::cfg::double_step_loop();
        test_expr_in_kernel(ki, &func, |se| {
            let expr = se.get_scev(VirtualUse::Value(Value::Instruction(2, 0)));
            if let Some(SCEVExpr::Add(x)) = expr.get(se) {
                assert_eq!(x.len(), 2);
                assert!(matches!(
                    x[0],
                    SCEVExprRef::Constant(APConstant::ConstantInt(2))
                ));
                if let Some(SCEVExpr::AddRec(addrec)) = x[1].get(se) {
                    assert_eq!(
                        addrec.operands,
                        vec![
                            SCEVExprRef::Constant(APConstant::ConstantInt(0)),
                            SCEVExprRef::Constant(APConstant::ConstantInt(2)),
                        ]
                    );
                } else {
                    assert!(false);
                }
            } else {
                assert!(false);
            }
        });
    }

    #[test]
    fn test_mad_32_32() {
        const CODE: &[u32] = &[
            0xbe8003ff, 0x00001001, // s_mov_b32 s0, 0x1001
            0xbe8103ff, 0x00001001, // s_mov_b32 s1, 0x1001
            0x7e000282, // v_mov_b32 v0, 2
            0x7e020282, // v_mov_b32 v1, 2
            0xd5766a00, 0x04000200, // v_mad_u64_u32 v[0:1], vcc_lo, s0, s1, v[0:1]
            0x7e040300, // v_mov_b32 v2, v0
            0x7e040301, // v_mov_b32 v2, v1
            0xbf810000, // s_endpgm
        ];
        test_expr_in_code(CODE, |se| {
            if let Some(val) = se.def_use.get_def_use(InstructionUse::op(5, 1, 0)) {
                let expr = se.get_scev(VirtualUse::Value(val));
                assert_eq!(expr.as_int(), Some(0x1002003));
            }
            if let Some(val) = se.def_use.get_def_use(InstructionUse::op(6, 1, 0)) {
                let expr = se.get_scev(VirtualUse::Value(val));
                assert!(matches!(expr, SCEVExprRef::Unknown(_)));
            }
        });
    }

    #[test]
    fn test_xor() {
        const CODE: &[u32] = &[
            0x8900bf00, // s_xor_b32 s0, s0, 63
            0xbf810000, // s_endpgm
        ];
        test_expr_in_code(CODE, |se| {
            let expr = se.get_scev(VirtualUse::Value(Value::Instruction(0, 0)));
            if let SCEVExprRef::Expr(SCEVExpr::Xor(ops)) = expr {
                assert_eq!(ops.len(), 2);
                assert_eq!(ops[0].as_int(), Some(63));
            } else {
                assert!(false);
            }
        });
    }

    #[test]
    fn test_pointer_mask() {
        const CODE: &[u32] = &[
            0xf4040002, 0xfa000000, // s_load_dwordx2 s[0:1], s[4:5], 0x0
            0xf4040080, 0xfa000000, // s_load_dwordx2 s[2:3], s[0:1], 0x0
            0x8701ff01, 0x0000ffff, // s_and_b32 s1, s1, 0xffff
            0xf4040080, 0xfa000000, // s_load_dwordx2 s[2:3], s[0:1], 0x0
            0xbf810000, // s_endpgm
        ];
        test_expr_in_code(CODE, |se| {
            let expr1 = se.get_scev(VirtualUse::Address(1));
            let expr2 = se.get_scev(VirtualUse::Address(3));
            assert_eq!(
                expr2,
                SCEVExprRef::Expr(SCEVExpr::And(vec![
                    SCEVExprRef::Constant(APConstant::ConstantInt((1 << 48) - 1)),
                    expr1
                ]))
            );
        });
    }
}
