use crate::analysis::DomFrontier;
use crate::error::Result;
use crate::fileformat::KernelInfo;
use crate::ir::machine::Register;
use crate::ir::{APConstant, DomTree, Function, Value, PHI};
use crate::isa::rdna2::isa::Operand;
use crate::isa::rdna2::Effect::{ReadOnly, ReadWrite, Write};
use crate::isa::rdna2::{Effect, RDNA2Target};
use std::collections::{BTreeSet, HashMap};
use std::fmt::Debug;
use std::iter::zip;

#[derive(Clone, Default, Debug, PartialEq, Eq)]
struct LiveSet {
    data: HashMap<Register, HashMap<usize, Value>>,
}

impl LiveSet {
    fn from_hashmap(d: HashMap<Register, Value>, bb_idx: usize) -> Self {
        let data = d
            .into_iter()
            .map(|(k, v)| {
                let mut s = HashMap::<usize, Value>::new();
                s.insert(bb_idx, v);
                (k, s)
            })
            .collect();
        Self { data }
    }

    #[cfg(test)]
    fn insert(&mut self, k: Register, bb_idx: usize, v: Value) -> bool {
        if let Some(x) = self.data.get_mut(&k) {
            x.insert(bb_idx, v).is_none()
        } else {
            let mut s = HashMap::new();
            s.insert(bb_idx, v);
            self.data.insert(k, s);
            true
        }
    }
}

struct ValueStack {
    values: Vec<HashMap<Register, Value>>,
}

impl ValueStack {
    fn new() -> Self {
        Self { values: vec![] }
    }

    fn push(&mut self) {
        self.values.push(HashMap::new())
    }

    fn pop(&mut self) {
        self.values.pop();
    }

    fn get(&self, r: &Register) -> Option<&Value> {
        for h in self.values.iter().rev() {
            if let Some(x) = h.get(r) {
                return Some(x);
            }
        }
        None
    }

    fn add(&mut self, r: Register, v: Value) {
        self.values.last_mut().and_then(|h| h.insert(r, v));
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub struct InstructionUse {
    pub inst_idx: usize,
    pub op_idx: usize,
    // Offset of an operand that uses multiple registers
    pub offset: u8,
}

impl InstructionUse {
    pub fn op(inst_idx: usize, op_idx: usize, offset: u8) -> InstructionUse {
        Self {
            inst_idx,
            op_idx,
            offset,
        }
    }
}

struct PHIInfo {
    // bb_idx -> (reg -> indices of the PHI array)
    locations: Vec<HashMap<Register, usize>>,
    nodes: Vec<PHI>,
}

struct Context<'a, 'b> {
    func: &'a Function<'a>,
    dom: &'b DomTree<'a>,
    df: &'b DomFrontier,
    ki: &'a KernelInfo<'a>,
}

/**
 * Construct PHI nodes to represent the convergences of the data flows.
 **/
pub struct PHIAnalysis<'a> {
    pub(crate) func: &'a Function<'a>,
    pub(crate) use_def: HashMap<InstructionUse, Value>,
    pub(crate) phis: Vec<PHI>,
}

impl<'a> PHIAnalysis<'a> {
    pub fn analyze<'b>(
        func: &'a Function<'a>,
        dom: &'b DomTree<'a>,
        df: &'b DomFrontier,
        ki: &'a KernelInfo,
    ) -> Result<PHIAnalysis<'a>> {
        let ctx = Context { func, dom, df, ki };
        ctx.analyze(func)
    }

    pub fn get_def_use(&self, k: InstructionUse) -> Option<Value> {
        if let Some(x) = self.use_def.get(&k) {
            Some(*x)
        } else if k.offset == 0 {
            let op = &self.func.instructions[k.inst_idx].get_operands()[k.op_idx];
            match op {
                Operand::Constant(x) => Some(Value::Constant(APConstant::ConstantInt(*x as isize))),
                _ => None,
            }
        } else {
            None
        }
    }
}

impl<'a, 'b> Context<'a, 'b> {
    fn analyze(&self, func: &'a Function<'a>) -> Result<PHIAnalysis<'a>> {
        let ki = self.ki;
        const BB_ENTRY_INDEX: usize = 0;
        let gen = (0..self.func.basic_blocks.len())
            .map(|x| self.compute_gen_set(x))
            .collect::<Vec<LiveSet>>();
        let mut phi_info = self.place_phi(&gen)?;
        let mut stacks = ValueStack::new();
        stacks.push();

        RDNA2Target::get_entry_live_entries(ki)?
            .into_iter()
            .for_each(|(x, v)| {
                stacks.add(x, v);
            });
        let mut use_def = HashMap::new();
        self.analyze_use_def(
            BB_ENTRY_INDEX,
            &phi_info.locations,
            &mut phi_info.nodes,
            &mut stacks,
            &mut use_def,
        );
        Self::post_process_phis(&phi_info.locations, &mut phi_info.nodes);
        Ok(PHIAnalysis {
            func,
            use_def,
            phis: phi_info.nodes,
        })
    }

    // the definitions of phis and bb -> phis
    fn place_phi(&self, gen: &[LiveSet]) -> Result<PHIInfo> {
        let mut worklist = gen
            .iter()
            .enumerate()
            .filter(|(bb_idx, _)| {
                self.df
                    .get_frontier(*bb_idx)
                    .map(|x| !x.is_empty())
                    .unwrap_or(false)
            })
            .flat_map(|(bb_idx, gen)| gen.data.keys().map(move |r| (bb_idx, *r)))
            .collect::<Vec<(usize, Register)>>();
        // use BTreeSet to make the traversal order deterministic
        let mut phis = Vec::<BTreeSet<Register>>::new();
        phis.resize(self.func.basic_blocks.len(), BTreeSet::new());
        while let Some((bb_idx, r)) = worklist.pop() {
            if let Some(x) = self.df.get_frontier(bb_idx) {
                x.iter().for_each(|frontier| {
                    if phis[*frontier].insert(r) {
                        worklist.push((*frontier, r));
                    }
                });
            }
        }

        let mut res_phi = Vec::new();
        let mut m = Vec::new();
        for regs in phis.into_iter() {
            let mut p = HashMap::new();
            for r in regs.into_iter() {
                let id = res_phi.len();
                res_phi.push(PHI::default());
                p.insert(r, id);
            }
            m.push(p);
        }
        Ok(PHIInfo {
            locations: m,
            nodes: res_phi,
        })
    }

    fn post_process_phis(phi_maps: &[HashMap<Register, usize>], phis: &mut [PHI]) {
        for (bb_idx, phi_idx) in phi_maps
            .iter()
            .enumerate()
            .flat_map(|(bb_idx, h)| h.iter().map(move |(_, idx)| (bb_idx, idx)))
        {
            let p = &mut phis[*phi_idx];
            p.bb_idx = bb_idx;
            p.values.sort_by_key(|(incoming_bb, _)| *incoming_bb);
        }
    }

    fn analyze_use_def(
        &self,
        bb_idx: usize,
        phi_maps: &Vec<HashMap<Register, usize>>,
        phis: &mut Vec<PHI>,
        stacks: &mut ValueStack,
        ret: &mut HashMap<InstructionUse, Value>,
    ) {
        let bb = &self.func.basic_blocks[bb_idx];
        const UNDEFINED: Value = Value::Undefined;
        stacks.push();
        phi_maps[bb_idx].iter().for_each(|(r, idx)| {
            stacks.add(*r, Value::Phi(*idx));
        });

        for inst_idx in bb.instructions.clone() {
            let inst = &self.func.instructions[inst_idx];
            let effects = Effect::get_effect(inst.op);

            for (op_idx, off, reg) in zip(effects.iter(), inst.operands.iter().enumerate())
                .filter_map(|(e, (op_idx, op))| match *e {
                    ReadOnly | ReadWrite => {
                        Register::from_operand(op).map(|(r, len)| (op_idx, r, len))
                    }
                    _ => None,
                })
                .flat_map(|(op_idx, r, len)| {
                    (0..len).map(move |off| (op_idx, off, r.offset(off as i8)))
                })
            {
                let v = if let Some(x) = stacks.get(&reg) {
                    x
                } else {
                    &UNDEFINED
                };
                ret.insert(InstructionUse::op(inst_idx, op_idx, off), *v);
            }

            zip(effects.iter(), inst.operands.iter())
                .filter_map(|(e, op)| match *e {
                    Write | ReadWrite => Register::from_operand(op),
                    _ => None,
                })
                .fold(0, |base, (reg, len)| {
                    (0..len).for_each(|off| {
                        let r = reg.offset(off as i8);
                        stacks.add(r, Value::Instruction(inst_idx, base + off as usize));
                    });
                    base + len as usize
                });
        }

        bb.successors
            .iter()
            .flat_map(|succ| phi_maps[*succ].iter())
            .for_each(|(r, phi_idx)| {
                let p = &mut phis[*phi_idx];
                p.values
                    .push((bb_idx, Box::new(*stacks.get(r).unwrap_or(&UNDEFINED))))
            });

        self.dom
            .get_node(bb_idx)
            .unwrap()
            .children_iter()
            .for_each(|x| self.analyze_use_def(*x, phi_maps, phis, stacks, ret));

        stacks.pop();
    }

    fn compute_gen_set(&self, bb_idx: usize) -> LiveSet {
        let bb = &self.func.basic_blocks[bb_idx];
        let mut gen = HashMap::new();
        for inst_idx in bb.instructions.clone() {
            let inst = &self.func.instructions[inst_idx];
            let effects = Effect::get_effect(inst.op);
            zip(effects.iter(), inst.operands.iter())
                .filter_map(|(e, op)| match *e {
                    Write | ReadWrite => Register::from_operand(op),
                    _ => None,
                })
                .fold(0, |off, (r, len)| {
                    for i in 0..len {
                        let reg = r.offset(i as i8);
                        gen.insert(reg, Value::Instruction(inst_idx, off + i as usize));
                    }
                    off + len as usize
                });
        }
        LiveSet::from_hashmap(gen, bb_idx)
    }
}

#[cfg(test)]
mod tests {
    use crate::analysis::phi::{Context, InstructionUse, LiveSet};
    use crate::analysis::DomFrontier;
    use crate::analysis::PHIAnalysis;
    use crate::ir::machine::Register;
    use crate::ir::machine::Register::{Scalar, Vector};
    use crate::ir::DomTree;
    use crate::ir::{Function, Value, PHI};
    use smallvec::SmallVec;
    use std::collections::HashMap;

    enum DirectValue {
        Instruction(usize, usize),
        Phi(PHI),
    }

    #[test]
    fn test_live_analysis() {
        use DirectValue::*;
        let (ki, func) = crate::tests::cfg::nested_loop();
        let dom = DomTree::analyze(&func);
        let df = DomFrontier::new(&dom);
        let analysis = PHIAnalysis::analyze(&func, &dom, &df, &ki).expect("Cannot analyze PHI");
        let ctx = Context {
            func: &func,
            dom: &dom,
            df: &df,
            ki,
        };
        assert_eq!(
            new_liveset(
                &func,
                &[
                    (Scalar(0), &[Value::Instruction(0, 0)]),
                    (Scalar(1), &[Value::Instruction(0, 1)]),
                    (Scalar(4), &[Value::Instruction(3, 0)]),
                    (Vector(0), &[Value::Instruction(1, 0)]),
                    (Vector(1), &[Value::Instruction(2, 0)]),
                ]
            ),
            ctx.compute_gen_set(0)
        );

        fn desc_phi(bb_idx: usize, arms: &[(usize, Value)]) -> DirectValue {
            let v = arms
                .iter()
                .map(|(i, v)| (*i, Box::new(v.clone())))
                .collect::<SmallVec<[(usize, Box<Value>); 2]>>();
            Phi(PHI::new(bb_idx, v))
        }

        fn op_use(inst_idx: usize, op_idx: usize) -> InstructionUse {
            InstructionUse::op(inst_idx, op_idx, 0)
        }
        fn op_use_offset(inst_idx: usize, op_idx: usize, off: u8) -> InstructionUse {
            InstructionUse::op(inst_idx, op_idx, off)
        }

        validate_use_def(
            &analysis,
            &analysis.use_def,
            &[
                (
                    op_use(6, 2),
                    desc_phi(
                        1,
                        &[
                            (0, Value::Instruction(0, 0)),
                            (3, Value::Instruction(14, 0)),
                        ],
                    ),
                ),
                (
                    op_use(6, 3),
                    desc_phi(
                        2,
                        &[(1, Value::Instruction(4, 0)), (2, Value::Instruction(8, 0))],
                    ),
                ),
                // SCC
                (op_use(7, 1), Instruction(6, 1)),
                (
                    op_use(7, 2),
                    desc_phi(
                        1,
                        &[
                            (0, Value::Instruction(0, 1)),
                            (3, Value::Instruction(15, 0)),
                        ],
                    ),
                ),
                (
                    op_use(7, 3),
                    desc_phi(
                        2,
                        &[(1, Value::Instruction(4, 1)), (2, Value::Instruction(9, 0))],
                    ),
                ),
                (
                    op_use(8, 2),
                    desc_phi(
                        2,
                        &[(1, Value::Instruction(4, 0)), (2, Value::Instruction(8, 0))],
                    ),
                ),
                (op_use(9, 1), Instruction(8, 1)),
                (
                    op_use(9, 2),
                    desc_phi(
                        2,
                        &[(1, Value::Instruction(4, 1)), (2, Value::Instruction(9, 0))],
                    ),
                ),
                (op_use_offset(11, 3, 0), Instruction(6, 0)),
                (op_use_offset(11, 3, 1), Instruction(7, 0)),
                (op_use(10, 1), Instruction(8, 0)),
                (op_use(12, 1), Instruction(10, 0)),
            ],
        );
    }

    fn get_inst_bb_idx(func: &Function, inst_idx: usize) -> Option<usize> {
        func.basic_blocks
            .iter()
            .enumerate()
            .find(|(_, bb)| bb.instructions.contains(&inst_idx))
            .map(|(idx, _)| idx)
    }

    fn new_liveset(func: &Function, values: &[(Register, &[Value])]) -> LiveSet {
        let mut out = LiveSet::default();
        values.iter().for_each(|(r, d)| {
            d.iter().for_each(|v| {
                if let Value::Instruction(inst_idx, _) = v {
                    let idx = get_inst_bb_idx(func, *inst_idx).expect("Cannot find basic block");
                    out.insert(*r, idx, v.clone());
                } else {
                    unreachable!();
                }
            })
        });
        out
    }

    fn validate_use_def<'a>(
        def_use: &'a PHIAnalysis,
        v: &HashMap<InstructionUse, Value>,
        expected: &[(InstructionUse, DirectValue)],
    ) {
        let r = expected.iter().all(|(u, d)| {
            let actual = v.get(u).expect("Cannot find the use");
            match d {
                DirectValue::Instruction(idx, off) => {
                    assert_eq!(Value::Instruction(*idx, *off), *actual);
                    *actual == Value::Instruction(*idx, *off)
                }
                DirectValue::Phi(p) => match actual {
                    Value::Phi(idx) => {
                        assert_eq!(*p, def_use.phis[*idx]);
                        *p == def_use.phis[*idx]
                    }
                    _ => false,
                },
            }
        });
        assert!(r);
    }
}
