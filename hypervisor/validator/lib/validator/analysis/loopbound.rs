use super::scalar_evolution::{AddRec, SCEVExprRef};
use crate::analysis::phi::InstructionUse;
use crate::analysis::scalar_evolution::VirtualUse;
use crate::analysis::{Loop, PHIAnalysis, SCEVExpr, ScalarEvolution};
use crate::ir::instruction::CmpInst;
use crate::ir::{APConstant, Value, PHI};
use crate::ir::{Function, ImplicitLoopInfo};
use crate::isa::rdna2::isa::Opcode;
use crate::isa::rdna2::opcodes::SOP2Opcode;

#[derive(Copy, Clone, Debug)]
#[allow(dead_code)]
pub struct LoopBound {
    pub(crate) start: Option<Value>,
    // Start with the case where the step value is constant
    pub(crate) step_value: Option<isize>,
    pub(crate) final_iv: Value,
}

pub(crate) struct LoopBoundAnalysis<'a, 'b> {
    use_def: &'b PHIAnalysis<'a>,
    se: &'b ScalarEvolution<'a, 'b>,
    implicit_loop_info: Option<ImplicitLoopInfo>,
}

impl<'a, 'b> LoopBoundAnalysis<'a, 'b> {
    pub(crate) fn new(
        use_def: &'b PHIAnalysis<'a>,
        se: &'b ScalarEvolution<'a, 'b>,
        implicit_loop_info: Option<ImplicitLoopInfo>,
    ) -> LoopBoundAnalysis<'a, 'b> {
        Self {
            use_def,
            se,
            implicit_loop_info,
        }
    }

    pub(crate) fn analyze(&self, l: &Loop) -> Option<LoopBound> {
        if l.is_implicit {
            let ili = self
                .implicit_loop_info
                .expect("No information on dimensions for implicit loops");
            let dim_id = ImplicitLoopInfo::IMPLICIT_LOOP_NUM / 2 - l.id / 2 - 1;
            let dim = &ili.dimensions[dim_id];
            let size = if l.id % 2 == 0 {
                dim.grid_size
            } else {
                dim.workgroup_size
            } - 1;
            return Some(LoopBound {
                start: Some(Value::Constant(APConstant::ConstantInt(0))),
                step_value: Some(1),
                final_iv: Value::Constant(APConstant::ConstantInt(size as isize)),
            });
        }

        let use_def = self.use_def;
        let header = l.get_header()?;
        let (lhs, rhs) = self.get_cmp_condition_operands(l)?;
        let lhs_scev = self.se.get_scev(VirtualUse::Value(lhs));
        let rhs_scev = self.se.get_scev(VirtualUse::Value(rhs));
        // iterate each PHI node to find the induction variable
        for (phi_idx, phi) in use_def
            .phis
            .iter()
            .enumerate()
            .filter(|(_, phi)| phi.bb_idx == header)
        {
            let scev = self.se.get_scev(VirtualUse::Value(Value::Phi(phi_idx)));
            if let Some(SCEVExpr::AddRec(addrec)) = scev.get(self.se) {
                let is_induction_variable = |op_scev: &SCEVExprRef| {
                    // case 1: cmp = step_inst < final_value
                    // we further consider the case where there are multiple intermediate step values when loop unrolling,
                    // it should be like `op = phi + some-steps`
                    if let Some(SCEVExpr::Add(x)) = op_scev.get(self.se) {
                        if x.iter().any(|e| e == &scev) {
                            return true;
                        }
                    }
                    // case 2: cmp = ind_var < final_value
                    if *op_scev == scev {
                        return true;
                    }
                    false
                };
                let candidate = [(&lhs_scev, &rhs), (&rhs_scev, &lhs)];
                if let Some((_, final_iv)) = candidate
                    .iter()
                    .find(|(op_scev, _)| is_induction_variable(op_scev))
                {
                    return self.get_bound(l, phi, &addrec, **final_iv);
                }
            }
        }
        None
    }

    fn get_bound(
        &self,
        l: &Loop,
        phi: &PHI,
        addrec: &AddRec,
        final_iv: Value,
    ) -> Option<LoopBound> {
        // We can use predecessor instead of preheader to get the incoming value.
        // The PHI node can be recognized as the induction value
        // no matter how many successors the predecessor has,
        // because we will neither hoist instructions nor modify the incoming value.
        let predecessor = l.get_loop_predecessor(self.use_def.func)?;
        let start = phi.get_incoming_value_for_block(predecessor).copied();
        let step_scev = addrec.get_step_recurrence();
        let step_value = step_scev.and_then(|s| s.as_int());
        Some(LoopBound {
            start,
            step_value,
            final_iv,
        })
    }

    fn get_cmp_condition_operands(&self, l: &Loop) -> Option<(Value, Value)> {
        let use_def = self.use_def;
        let func = use_def.func;
        let condition = l.get_latch_condition(func, use_def)?;
        let (idx, cmp_inst) = self.get_cmp_condition(&condition, func, l)?;
        if !cmp_inst.get_type().is_integer_ty() {
            return None;
        }
        let lhs = use_def.get_def_use(InstructionUse::op(idx, 1, 0))?;
        let rhs = use_def.get_def_use(InstructionUse::op(idx, 2, 0))?;
        Some((lhs, rhs))
    }

    fn get_cmp_condition(
        &self,
        condition: &Value,
        func: &Function,
        l: &Loop,
    ) -> Option<(usize, CmpInst)> {
        if let &Value::Instruction(idx, op_idx) = condition {
            let inst = &func.instructions[idx];
            match (inst.op, op_idx) {
                (Opcode::SOP2(o), _) => {
                    use SOP2Opcode::*;
                    match (o, op_idx) {
                        (S_AND_B32, 0) | (S_ANDN2_B32, 0) | (S_OR_B32, 0) => {
                            let lhs = self.use_def.get_def_use(InstructionUse::op(idx, 2, 0))?;
                            let rhs = self.use_def.get_def_use(InstructionUse::op(idx, 3, 0))?;
                            [lhs, rhs]
                                .into_iter()
                                .filter(|v| !l.is_loop_invariant(v, self.use_def))
                                .filter_map(|v| self.get_cmp_condition(&v, func, l))
                                .next()
                        }
                        _ => None,
                    }
                }
                (_, 0) => {
                    let ir_inst = func.instructions[idx].clone().wrap()?;
                    let cmp_inst = ir_inst.as_cmp_inst()?.clone();
                    Some((idx, cmp_inst))
                }
                (_, _) => None,
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::analysis::loopbound::LoopBoundAnalysis;
    use crate::analysis::{
        ConstantPropagation, DomFrontier, InstructionUse, LoopAnalysis, PHIAnalysis,
        PolyhedralAnalysis, SCEVExpr, ScalarEvolution, VirtualUse,
    };
    use crate::fileformat::{Disassembler, KernelInfo, SGPRSetup};
    use crate::ir::ImplicitLoopInfo;
    use crate::ir::{APConstant, DomTree, Function, Value};
    use crate::tests::cfg::simple_kernel_info;

    fn test_loop(
        ki: &KernelInfo,
        func: &Function,
        loop_idx: usize,
        step_value: isize,
        start: isize,
        end: isize,
    ) {
        let dom = DomTree::analyze(&func);
        let df = DomFrontier::new(&dom);
        let def_use = PHIAnalysis::analyze(&func, &dom, &df, &ki).expect("Cannot analyze Phi");
        let def_use = ConstantPropagation::run(def_use);
        let mut li = LoopAnalysis::new(&func);
        li.analyze(&dom);
        let se = ScalarEvolution::new(&dom, &def_use, &li, None);
        let lba = LoopBoundAnalysis::new(&def_use, &se, None);
        let lb = lba.analyze(&li.loops[loop_idx]).expect("Cannot find bound");
        assert_eq!(Some(step_value), lb.step_value);
        assert_eq!(
            Some(Value::Constant(APConstant::ConstantInt(start))),
            lb.start
        );
        assert_eq!(Value::Constant(APConstant::ConstantInt(end)), lb.final_iv);
    }

    #[test]
    fn test_nested_loop() {
        let (ki, func) = crate::tests::cfg::nested_loop();
        test_loop(ki, &func, 0, 4, 0, 24);
    }

    #[test]
    fn test_final_iv_sreg() {
        // s_mov_b32 s0, 0                                            // 000000001000: BE800380
        // s_mov_b32 s1, 16                                           // 000000001004: BE810390
        // s_add_u32 s0, s0, 1                                        // 000000001008: 80008100
        // s_nop 0                                                    // 00000000100C: BF800000
        // s_cmp_lg_u32 s0, s1                                        // 000000001010: BF070100
        // s_cbranch_scc1 65532                                       // 000000001014: BF85FFFC <f+0x8>
        // s_endpgm                                                   // 000000001018: BF810000
        const CODE: &[u32] = &[
            0xBE800380, 0xBE810390, 0x80008100, 0xBF800000, 0xBF070100, 0xBF85FFFC, 0xBF810000,
        ];
        let ki = simple_kernel_info("f", &CODE);
        let func = Disassembler::parse_kernel(&ki).expect("Failed to parse the function");
        test_loop(&ki, &func, 0, 1, 0, 16);
    }

    #[test]
    fn test_cmp_operand_position() {
        // s_mov_b32 s0, 0                                            // 000000001000: BE800380
        // s_add_u32 s0, s0, 1                                        // 000000001004: 80008100
        // s_cmp_lg_u32 8, s0                                         // 000000001008: BF070088
        // s_cbranch_scc1 65533                                       // 00000000100C: BF85FFFD <f+0x4>
        // s_endpgm                                                   // 000000001010: BF810000
        const CODE: &[u32] = &[0xBE800380, 0x80008100, 0xBF070088, 0xBF85FFFD, 0xBF810000];
        const KERNEL_INFO: KernelInfo = simple_kernel_info("f", &CODE);
        let func = Disassembler::parse_kernel(&KERNEL_INFO).expect("Failed to parse the function");
        test_loop(&KERNEL_INFO, &func, 0, 1, 0, 8);
    }

    #[test]
    fn test_cmp_inst_position() {
        // v_mov_b32_e32 v0, 0                                        // 000000001000: 7E000280
        // v_cmp_ne_u32_e32 vcc_lo, 8, v0                             // 000000001004: 7D8A0088
        // v_add_nc_u32_e32 v0, 1, v0                                 // 000000001008: 4A000081
        // s_cbranch_vccnz 65533                                      // 00000000100C: BF87FFFD <f+0x4>
        // s_endpgm                                                   // 000000001010: BF810000
        const CODE: &[u32] = &[0x7E000280, 0x7D8A0088, 0x4A000081, 0xBF87FFFD, 0xBF810000];
        const KERNEL_INFO: &KernelInfo = &simple_kernel_info("f", &CODE);
        let func = Disassembler::parse_kernel(KERNEL_INFO).expect("Failed to parse the function");
        test_loop(KERNEL_INFO, &func, 0, 1, 0, 8);
    }

    #[test]
    fn test_step_add_i32() {
        // s_mov_b32 s0, 0                                            // 000000001000: BE800380
        // s_add_i32 s0, s0, 1                                        // 000000001000: 81008100
        // s_cmp_lg_u32 s0, 8                                         // 000000001004: BF078800
        // s_cbranch_scc1 65533                                       // 000000001008: BF85FFFD <f>
        // s_endpgm                                                   // 00000000100C: BF810000
        const CODE: &[u32] = &[0xBE800380, 0x81008100, 0xBF078800, 0xBF85FFFD, 0xBF810000];
        let ki = simple_kernel_info("f", &CODE);
        let func = Disassembler::parse_kernel(&ki).expect("Failed to parse the function");
        test_loop(&ki, &func, 0, 1, 0, 8);
    }

    #[test]
    fn test_step_sub_u32() {
        // s_mov_b32 s0, 8                                            // 000000001000: BE800388
        // s_sub_u32 s0, s0, 1                                        // 000000001004: 80808100
        // s_cmp_lg_u32 s0, 0                                         // 000000001008: BF078000
        // s_cbranch_scc1 65533                                       // 00000000100C: BF85FFFD <f+0x4>
        // s_endpgm                                                   // 000000001010: BF810000
        const CODE: &[u32] = &[0xBE800388, 0x80808100, 0xBF078000, 0xBF85FFFD, 0xBF810000];
        let ki = simple_kernel_info("f", &CODE);
        let func = Disassembler::parse_kernel(&ki).expect("Failed to parse the function");
        test_loop(&ki, &func, 0, -1, 8, 0);
    }

    #[test]
    fn test_step_sub_i32() {
        // s_mov_b32 s0, 8                                            // 000000001000: BE800388
        // s_sub_i32 s0, s0, 1                                        // 000000001004: 81808100
        // s_cmp_lg_u32 s0, 0                                         // 000000001008: BF078000
        // s_cbranch_scc1 65533                                       // 00000000100C: BF85FFFD <f+0x4>
        // s_endpgm                                                   // 000000001010: BF810000
        const CODE: &[u32] = &[0xBE800388, 0x81808100, 0xBF078000, 0xBF85FFFD, 0xBF810000];
        let ki = simple_kernel_info("f", &CODE);
        let func = Disassembler::parse_kernel(&ki).expect("Failed to parse the function");
        test_loop(&ki, &func, 0, -1, 8, 0);
    }

    #[test]
    fn test_implicit_loop_bound() {
        // s_mov_b32 s0, s0                                           // 000000001000: BE800300
        // s_mov_b32 s1, s1                                           // 000000001004: BE810301
        // s_mov_b32 s2, s2                                           // 000000001008: BE820302
        // v_mov_b32_e32 v0, v0                                       // 00000000100C: 7E000300
        // v_mov_b32_e32 v1, v1                                       // 000000001010: 7E020301
        // v_mov_b32_e32 v2, v2                                       // 000000001014: 7E040302
        // s_endpgm                                                   // 000000001018: BF810000
        const CODE: &[u32] = &[
            0xBE800300, 0xBE810301, 0xBE820302, 0x7E000300, 0x7E020301, 0x7E040302, 0xBF810000,
            0xBF810000,
        ];
        const KERNEL_INFO: &KernelInfo = &KernelInfo {
            name: "",
            code: CODE,
            pgmrsrcs: [0, 0x1380, 0], // enable workgroup id x/y/z, workitem id x/y/z
            setup: SGPRSetup::from_bits_truncate(0),
            kern_arg_size: 8,
            kern_arg_segment_align: 0,
            group_segment_fixed_size: 0,
            private_segment_fixed_size: 0,
            arguments: vec![],
        };
        let func = &Disassembler::parse_kernel(KERNEL_INFO).expect("Failed to parse the function");
        let dom = DomTree::analyze(func);
        let df = DomFrontier::new(&dom);
        let def_use =
            PHIAnalysis::analyze(func, &dom, &df, KERNEL_INFO).expect("Failed to analysis PHI");
        let mut li = LoopAnalysis::new(func);
        li.analyze(&dom);
        let implicit_loop = ImplicitLoopInfo::new([64, 32, 16], [8, 4, 2]);
        li.augment_with_implicit_loops();
        assert_eq!(li.loops.len(), 6);
        let scev = ScalarEvolution::new(&dom, &def_use, &li, Some(implicit_loop.clone()));
        let mut poly = PolyhedralAnalysis::new(func, &scev);
        poly.analyze();
        let lba = LoopBoundAnalysis::new(&def_use, &scev, Some(implicit_loop));

        [
            // loop_idx, inst_idx, start, final_iv
            (0, 2, 0, 2 - 1),  // group_id_z
            (1, 5, 0, 16 - 1), // local_id_z
            (2, 1, 0, 4 - 1),  // group_id_y
            (3, 4, 0, 32 - 1), // local_id_y
            (4, 0, 0, 8 - 1),  // group_id_x
            (5, 3, 0, 64 - 1), // local_id_x
        ]
        .iter()
        .cloned()
        .for_each(|(loop_idx, inst_idx, start, final_iv)| {
            let v = def_use
                .get_def_use(InstructionUse::op(inst_idx, 1, 0))
                .expect("No value in def-use");
            let expr = scev.get_scev(VirtualUse::Value(v));
            if let Some(SCEVExpr::AddRec(ref addrec)) = expr.get(&scev) {
                assert_eq!(addrec.loop_info_id, loop_idx);
            } else {
                assert!(false);
            }
            let l = lba
                .analyze(&li.loops[loop_idx])
                .expect("No bounds for implicit loop");
            assert_eq!(
                Some(Value::Constant(APConstant::ConstantInt(start))),
                l.start
            );
            assert_eq!(
                Value::Constant(APConstant::ConstantInt(final_iv)),
                l.final_iv
            );
        });
    }

    #[test]
    fn test_vgpr_induction() {
        let (ki, func) = crate::tests::cfg::vgpr_induction_loop();
        test_loop(&ki, &func, 0, 1, 0, 127);
    }
}
