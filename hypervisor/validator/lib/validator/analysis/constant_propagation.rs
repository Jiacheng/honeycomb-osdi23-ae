use crate::analysis::phi::InstructionUse;
use crate::analysis::PHIAnalysis;
use crate::ir::machine::Register;
use crate::ir::{APConstant, Function, Value};
use crate::isa::rdna2::isa::Opcode;
use crate::isa::rdna2::opcodes::{SOP1Opcode, SOPKOpcode, VOP1Opcode};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Clone, Debug)]
struct PHIUse {
    phi_idx: usize,
    op_idx: usize,
}

//
// Describe a use for either an instruction or a PHI node.
// XXX: Should consider moving to the IR module and make it a first-class citizen.
#[derive(Clone, Debug)]
enum OperandUse {
    Instruction(InstructionUse),
    PHIUse(PHIUse),
}

pub(crate) struct ConstantPropagation {}

enum WorkItem {
    Instruction(usize),
    Phi(usize),
}

impl ConstantPropagation {
    pub(crate) fn run(mut phi: PHIAnalysis) -> PHIAnalysis {
        let func = phi.func;
        let mut w = (0..func.instructions.len())
            .filter(|idx| match &func.instructions[*idx].op {
                Opcode::SOP1(opcode) => {
                    matches!(opcode, SOP1Opcode::S_MOV_B32 | SOP1Opcode::S_MOV_B64)
                }
                Opcode::SOPK(SOPKOpcode::S_MOVK_I32) => true,
                Opcode::VOP1(VOP1Opcode::V_MOV_B32) => true,
                // This effectively makes the constant propagation pass merges the ranges for the whole warp into one
                Opcode::VOP1(VOP1Opcode::V_READFIRSTLANE_B32) => true,
                _ => false,
            })
            .map(WorkItem::Instruction)
            .collect::<VecDeque<WorkItem>>();

        let mut def_use = Self::collect_def_use(func, &phi);
        while let Some(work) = w.pop_front() {
            match work {
                WorkItem::Instruction(idx) => {
                    let inst = &func.instructions[idx];
                    match inst.op {
                        Opcode::SOP1(SOP1Opcode::S_MOV_B32)
                        | Opcode::VOP1(VOP1Opcode::V_MOV_B32)
                        | Opcode::VOP1(VOP1Opcode::V_READFIRSTLANE_B32)
                        | Opcode::SOPK(SOPKOpcode::S_MOVK_I32) => {
                            if let Some(v) = phi.get_def_use(InstructionUse::op(idx, 1, 0)) {
                                let old_value = Value::Instruction(idx, 0);
                                let changed =
                                    Self::replace_uses(&mut def_use, &mut phi, &old_value, v);
                                w.extend(changed.into_iter());
                            }
                        }
                        Opcode::SOP1(SOP1Opcode::S_MOV_B64) => {
                            let ops = [
                                phi.get_def_use(InstructionUse::op(idx, 1, 0)),
                                phi.get_def_use(InstructionUse::op(idx, 1, 1)),
                            ];
                            for i in 0..2 {
                                let old_value = Value::Instruction(idx, i);
                                let n = if let Some(Value::Constant(APConstant::ConstantInt(c))) =
                                    ops[0]
                                {
                                    let v = (c >> (i * 32)) as i32;
                                    Some(Value::Constant(APConstant::ConstantInt(v as isize)))
                                } else {
                                    ops[i]
                                };
                                if let Some(v) = n {
                                    w.extend(Self::replace_uses(
                                        &mut def_use,
                                        &mut phi,
                                        &old_value,
                                        v,
                                    ));
                                }
                            }
                        }
                        _ => {}
                    }
                }
                WorkItem::Phi(idx) => {
                    let p = &phi.phis[idx];
                    let c = p
                        .values
                        .iter()
                        .map(|(_, v)| *v.as_ref())
                        .collect::<HashSet<Value>>();
                    if c.len() == 1 {
                        let v = *c.iter().next().unwrap();
                        let old_value = Value::Phi(idx);
                        let changed = Self::replace_uses(&mut def_use, &mut phi, &old_value, v);
                        w.extend(changed.into_iter());
                    }
                }
            }
        }
        phi
    }

    fn replace_uses(
        def_use: &mut HashMap<Value, Vec<OperandUse>>,
        phi: &mut PHIAnalysis,
        old_v: &Value,
        new_v: Value,
    ) -> Vec<WorkItem> {
        let mut changed = Vec::new();
        if let Some(uses) = def_use.get(old_v) {
            uses.iter().for_each(|u| match u {
                OperandUse::Instruction(inst_use) if phi.use_def.contains_key(inst_use) => {
                    phi.use_def.insert(*inst_use, new_v);
                    changed.push(WorkItem::Instruction(inst_use.inst_idx));
                }
                OperandUse::PHIUse(pu) => {
                    let p = &mut phi.phis[pu.phi_idx];
                    p.values[pu.op_idx].1 = Box::new(new_v);
                    changed.push(WorkItem::Phi(pu.phi_idx));
                }
                _ => {}
            });
        }
        changed
    }

    fn collect_def_use(func: &Function<'_>, phi: &PHIAnalysis) -> HashMap<Value, Vec<OperandUse>> {
        let mut def_use: HashMap<Value, Vec<OperandUse>> = HashMap::new();
        for inst_use in func
            .instructions
            .iter()
            .enumerate()
            .flat_map(|(inst_idx, inst)| {
                inst.operands
                    .iter()
                    .enumerate()
                    .filter_map(move |(op_idx, op)| {
                        let r = Register::from_operand(op)?;
                        Some((inst_idx, op_idx, r.1))
                    })
            })
            .flat_map(|(inst_idx, op_idx, len)| {
                (0..len).map(move |off| (InstructionUse::op(inst_idx, op_idx, off)))
            })
        {
            if let Some(def) = phi.get_def_use(inst_use) {
                if let Some(x) = def_use.get_mut(&def) {
                    x.push(OperandUse::Instruction(inst_use));
                } else {
                    def_use.insert(def, vec![OperandUse::Instruction(inst_use)]);
                }
            }
        }

        for (phi_idx, op_idx, v) in phi
            .use_def
            .values()
            .filter_map(|x| match *x {
                Value::Phi(phi_idx) => Some((phi_idx, &phi.phis[phi_idx])),
                _ => None,
            })
            .flat_map(|(phi_idx, p)| {
                p.values
                    .iter()
                    .enumerate()
                    .map(move |(op_idx, (_, v))| (phi_idx, op_idx, v))
            })
        {
            let pu = OperandUse::PHIUse(PHIUse { phi_idx, op_idx });
            if let Some(x) = def_use.get_mut(v.as_ref()) {
                x.push(pu);
            } else {
                def_use.insert(*v.as_ref(), vec![pu]);
            }
        }
        def_use
    }
}

#[cfg(test)]
mod tests {
    use crate::analysis::phi::InstructionUse;
    use crate::analysis::{ConstantPropagation, DomFrontier, PHIAnalysis};
    use crate::fileformat::Disassembler;
    use crate::ir::{APConstant, DomTree, Value};
    use crate::tests::cfg::simple_kernel_info;

    #[test]
    fn test_constant_propagation() {
        const CODE: &[u32] = &[
            0x7e000280, // v_mov_b32_e32 v0, 0
            0x7e020300, // v_mov_b32_e32 v1, v0
            0xbe840380, // s_mov_b32 s4, 0
            0xbe820480, // s_mov_b64 s[2:3], 0
            0xbe850304, // s_mov_b32 s5, s4
            0xbe850303, // s_mov_b32 s5, s3
            0xbe860305, // s_mov_b32 s6, s5
            0xbf85fffd, // s_cbranch_scc1 65533
            0xb0040100, // s_movk_i32 s4, 256
            0xbe850304, // s_mov_b32 s5, s4
            0xf4040003, 0xfa000000, // s_load_dwordx2 s[0:1], s[6:7], 0x0
            0xbe840400, // s_mov_b64 s[4:5], s[0:1]
            0xbe820304, // s_mov_b32 s2, s4
            0xbf850002, // s_cbranch_scc1 2
            0xbe830300, // s_mov_b32 s3, s0
            0xbe820303, // s_mov_b32 s2, s3
            0x7e100501, // v_readfirstlane_b32 s8, v1
            0xbe800308, // s_mov_b32 s0, s8
            0xbf810000, // s_endpgm
        ];

        let ki = simple_kernel_info("f", CODE);

        let func = Disassembler::parse_kernel(&ki).expect("Failed to parse the function");
        let dom = DomTree::analyze(&func);
        let df = DomFrontier::new(&dom);
        let phi = PHIAnalysis::analyze(&func, &dom, &df, &ki).expect("Cannot analyze phi");
        let phi = ConstantPropagation::run(phi);

        assert_eq!(
            Some(Value::Constant(APConstant::ConstantInt(0))),
            phi.get_def_use(InstructionUse::op(1, 1, 0))
        );

        assert_eq!(
            Some(Value::Constant(APConstant::ConstantInt(0))),
            phi.get_def_use(InstructionUse::op(4, 1, 0))
        );

        assert_eq!(
            Some(Value::Constant(APConstant::ConstantInt(0))),
            phi.get_def_use(InstructionUse::op(5, 1, 0))
        );

        assert_eq!(
            Some(Value::Constant(APConstant::ConstantInt(0))),
            phi.get_def_use(InstructionUse::op(6, 1, 0))
        );

        assert_eq!(
            Some(Value::Constant(APConstant::ConstantInt(256))),
            phi.get_def_use(InstructionUse::op(9, 1, 0))
        );

        assert_eq!(
            phi.get_def_use(InstructionUse::op(12, 1, 0)),
            phi.get_def_use(InstructionUse::op(15, 1, 0))
        );

        assert_eq!(
            Some(Value::Constant(APConstant::ConstantInt(0))),
            phi.get_def_use(InstructionUse::op(17, 1, 0))
        );
    }
}
