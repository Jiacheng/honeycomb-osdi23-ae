// This is essentially a reimplementation of the LLVM's LoopInfo pass.
// It should be moved out of the TCB.

use crate::analysis::{phi::InstructionUse, PHIAnalysis};
use crate::ir::ImplicitLoopInfo;
use crate::ir::{DomTree, Function, Value, CFG};
use petgraph::visit::DfsPostOrder;
use smallvec::SmallVec;
use std::collections::HashMap;

pub struct Loop {
    pub(crate) id: usize,
    pub(crate) is_implicit: bool,
    pub(crate) parent_loop: Option<usize>,
    pub(crate) sub_loops: Vec<usize>,
    pub(crate) blocks: Vec<usize>,
}

pub struct LoopAnalysis<'a> {
    pub(crate) func: &'a Function<'a>,
    // basic block -> loop
    bb_map: HashMap<usize, usize>,
    pub(crate) loops: Vec<Loop>,
    top_level_loops: Vec<usize>,
}

impl Loop {
    /**
     * If the given loop's header has exactly one unique predecessor outside the loop, return it.
     * Otherwise return null.
     * This is less strict that the loop "preheader" concept, which requires
     * the predecessor to have exactly one successor.
     **/
    pub fn get_loop_predecessor<'a>(&self, func: &'a Function<'a>) -> Option<usize> {
        let bb = &func.basic_blocks[self.get_header()?];
        let mut it = bb.predecessors.iter().filter(|x| !self.contains(**x));
        let r = it.next().cloned();
        if it.next().is_none() {
            r
        } else {
            None
        }
    }

    pub fn get_header(&self) -> Option<usize> {
        self.blocks.first().cloned()
    }

    pub fn is_loop_invariant(&self, v: &Value, def_use: &PHIAnalysis) -> bool {
        match v {
            Value::Constant(_) => true,
            Value::Argument(_) => true,
            Value::Instruction(inst_idx, _) => {
                // This analysis is not complete,
                // but we assume the compiler has processed an LICM pass
                !self.blocks.iter().any(|&bb_idx| {
                    def_use.func.basic_blocks[bb_idx]
                        .instructions
                        .contains(inst_idx)
                })
            }
            // The PHI node is defined out of the loop, or the values of all arms expect the PHI node itself are defined outside of the loop
            Value::Phi(phi) => {
                let p = &def_use.phis[*phi];
                !self.contains(p.bb_idx)
                    || !p
                        .values
                        .iter()
                        .any(|(bbi, arm)| self.contains(*bbi) && arm.as_ref() != v)
            }
            Value::Undefined => false,
        }
    }

    pub fn get_loop_latch(&self, func: &Function) -> Option<usize> {
        let header = self.get_header()?;
        let mut latch_iter = func.basic_blocks[header]
            .predecessors
            .iter()
            .filter(|p| self.contains(**p));
        let latch = *latch_iter.next()?;
        latch_iter.next().is_none().then_some(latch)
    }

    pub fn get_latch_condition(&self, func: &Function, use_def: &PHIAnalysis) -> Option<Value> {
        let latch = self.get_loop_latch(func)?;
        let branch_inst_idx = func.basic_blocks[latch].instructions.clone().last()?;
        let branch_inst = func.instructions[branch_inst_idx].clone().wrap()?;
        let branch_inst = branch_inst.as_branch_inst()?;
        if !branch_inst.fall_through() {
            return None;
        }
        use_def.get_def_use(InstructionUse::op(branch_inst_idx, 1, 0))
    }

    pub fn contains(&self, bb_idx: usize) -> bool {
        self.blocks.iter().any(|x| *x == bb_idx)
    }

    fn get_outermost_loop(&self, li: &LoopAnalysis) -> usize {
        let mut l = self;
        while let Some(up) = l.parent_loop {
            l = &li.loops[up];
        }
        l.id
    }
}

impl<'a> LoopAnalysis<'a> {
    pub fn new(func: &'a Function<'a>) -> LoopAnalysis<'a> {
        LoopAnalysis {
            func,
            bb_map: HashMap::new(),
            loops: Vec::new(),
            top_level_loops: vec![],
        }
    }

    pub fn analyze(&mut self, dom: &DomTree<'a>) {
        // Postorder traversal of the dominator tree.
        for header in dom.post_order_iter(dom.root()) {
            let header_bb = &dom.func().basic_blocks[header];
            // Check each predecessor of the potential loop header.
            let back_edges: SmallVec<[usize; 4]> = header_bb
                .predecessors
                .iter()
                .filter_map(|back_edge| {
                    if dom.dominates(header, *back_edge) && dom.is_reachable_from_entry(*back_edge)
                    {
                        Some(*back_edge)
                    } else {
                        None
                    }
                })
                .collect();

            // Perform a backward CFG traversal to discover and map blocks in this loop.
            if !back_edges.is_empty() {
                let id = self.loops.len();
                let l = Loop {
                    id,
                    is_implicit: false,
                    parent_loop: None,
                    sub_loops: vec![],
                    blocks: vec![header],
                };
                self.loops.push(l);
                self.discover_and_map_subloop(back_edges, id, dom);
            }
        }

        // Perform a single forward CFG traversal to populate block and subloop
        // vectors for all loops.
        let cfg = &CFG::new(self.func);
        let mut postorder = DfsPostOrder::new(cfg, dom.root());
        while let Some(n) = postorder.next(cfg) {
            self.insert_into_loop(n);
        }
    }

    pub fn get_loop_for(&self, bb_idx: usize) -> Option<&Loop> {
        self.bb_map.get(&bb_idx).map(|x| &self.loops[*x])
    }

    /**
     *  Discover a subloop with the specified backedges such that: All blocks within
     * this loop are mapped to this loop or a subloop. And all subloops within this
     * loop have their parent loop set to this loop or a subloop.
     **/
    fn discover_and_map_subloop(
        &mut self,
        backedges: SmallVec<[usize; 4]>,
        lid: usize,
        dom: &DomTree<'a>,
    ) {
        // Perform a backward CFG traversal using a worklist.
        let mut reverse_cfg_worklist = backedges;
        while let Some(pred_bb) = reverse_cfg_worklist.pop() {
            if let Some(subloop) = self.bb_map.get(&pred_bb) {
                // This is a discovered block. Find its outermost discovered loop.
                let subloop = self.loops[*subloop].get_outermost_loop(self);

                // If it is already discovered to be a subloop of this loop, continue.
                if subloop == lid {
                    continue;
                }

                // Discover a subloop of this loop.
                self.loops[subloop].parent_loop = Some(lid);
                // ++NumSubloops;
                // NumBlocks += Subloop->getBlocksVector().capacity();
                let bb_id = self.loops[subloop].get_header().unwrap();
                let bb = &self.func.basic_blocks[bb_id];
                // Continue traversal along predecessors that are not loop-back edges from
                // within this subloop tree itself. Note that a predecessor may directly
                // reach another subloop that is not yet discovered to be a subloop of
                // this loop, which we must traverse.
                reverse_cfg_worklist.extend(bb.predecessors.iter().filter_map(|x| {
                    match self.bb_map.get(x) {
                        Some(y) if *y == subloop => None,
                        _ => Some(*x),
                    }
                }));
            } else {
                if !dom.is_reachable_from_entry(pred_bb) {
                    continue;
                }
                // This is an undiscovered block. Map it to the current loop.
                self.bb_map.insert(pred_bb, lid);
                // ++NumBlocks;
                if let Some(p) = self.loops[lid].get_header() {
                    if pred_bb == p {
                        continue;
                    }
                }
                // Push all block predecessors on the worklist.
                let bb = &self.func.basic_blocks[pred_bb];
                reverse_cfg_worklist.extend_from_slice(bb.predecessors.as_slice());
            }
        }
    }

    /**
     * Add a single Block to its ancestor loops in PostOrder. If the block is a
     * subloop header, add the subloop to its parent in PostOrder, then reverse the
     * Block and Subloop vectors of the now complete subloop to achieve RPO.
     **/
    fn insert_into_loop(&mut self, bb_id: usize) {
        let loop_id = self.bb_map.get(&bb_id);
        let mut iter_id = loop_id.copied();
        if let Some(lid) = loop_id {
            let l = self.loops.get_mut(*lid).unwrap();
            if let Some(header_id) = l.get_header() {
                if header_id == bb_id {
                    // For convenience, Blocks and Subloops are inserted in postorder. Reverse
                    // the lists, except for the loop header, which is always at the beginning.
                    let (_, other_bb) = l.blocks.split_first_mut().unwrap();
                    other_bb.reverse();
                    l.sub_loops.reverse();
                    iter_id = l.parent_loop;

                    // We reach this point once per subloop after processing all the blocks in
                    // the subloop.
                    if let Some(parent_id) = l.parent_loop {
                        let parent = self.loops.get_mut(parent_id);
                        parent.unwrap().sub_loops.push(*lid);
                    } else {
                        self.top_level_loops.push(*lid);
                    }
                }
            }
        }
        while let Some(id) = iter_id {
            let curr = self.loops.get_mut(id).unwrap();
            curr.blocks.push(bb_id);
            iter_id = curr.parent_loop;
        }
    }

    /**
     * Augment the analysis with the six (workgroup / grid over x/y/z) implicit loops
     */
    pub(crate) fn augment_with_implicit_loops(&mut self) {
        for l in self.loops.iter_mut() {
            l.id += ImplicitLoopInfo::IMPLICIT_LOOP_NUM;
            if let Some(x) = l.parent_loop {
                l.parent_loop = Some(x + ImplicitLoopInfo::IMPLICIT_LOOP_NUM);
            }
            l.sub_loops
                .iter_mut()
                .for_each(|x| *x += ImplicitLoopInfo::IMPLICIT_LOOP_NUM);
        }
        self.bb_map
            .iter_mut()
            .for_each(|(_, x)| *x += ImplicitLoopInfo::IMPLICIT_LOOP_NUM);
        self.top_level_loops
            .iter_mut()
            .for_each(|x| *x += ImplicitLoopInfo::IMPLICIT_LOOP_NUM);

        // XXX: we are yet to care about the data inside implicit loops
        (0..ImplicitLoopInfo::IMPLICIT_LOOP_NUM)
            .rev()
            .for_each(|x| {
                self.loops.insert(
                    0,
                    Loop {
                        id: x,
                        is_implicit: true,
                        parent_loop: None,
                        sub_loops: vec![],
                        blocks: vec![],
                    },
                );
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{phi::InstructionUse, ConstantPropagation, DomFrontier};
    use crate::fileformat::Disassembler;
    use crate::ir::BasicBlock;
    use crate::tests::cfg::simple_kernel_info;
    use smallvec::smallvec;
    use std::ops::Range;

    #[test]
    fn test_loop_analysis() {
        const NULL_RANGE: Range<usize> = Range { start: 0, end: 0 };
        let basic_blocks = vec![
            BasicBlock {
                offset: 0,
                predecessors: smallvec![3],
                successors: smallvec![1],
                instructions: NULL_RANGE,
            },
            BasicBlock {
                offset: 1,
                predecessors: smallvec![0, 2],
                successors: smallvec![2],
                instructions: NULL_RANGE,
            },
            BasicBlock {
                offset: 2,
                predecessors: smallvec![1],
                successors: smallvec![3],
                instructions: NULL_RANGE,
            },
            BasicBlock {
                offset: 3,
                predecessors: smallvec![2],
                successors: smallvec![0],
                instructions: NULL_RANGE,
            },
        ];
        let func = Function {
            name: "test",
            basic_blocks,
            instructions: vec![],
        };

        let dom = DomTree::analyze(&func);
        let mut li = LoopAnalysis::new(&func);
        li.analyze(&dom);
        assert_eq!(2, li.loops.len());
        assert_eq!(Some(1), li.loops[0].parent_loop);
        assert_eq!([1, 2], li.loops[0].blocks.as_slice());
        assert_eq!(None, li.loops[1].parent_loop);
        assert_eq!([0, 1, 2, 3], li.loops[1].blocks.as_slice());
    }

    #[test]
    fn test_loop_invariant() {
        let (ki, func) = crate::tests::cfg::nested_loop();
        let dom = DomTree::analyze(&func);
        let df = DomFrontier::new(&dom);
        let def_use = PHIAnalysis::analyze(&func, &dom, &df, &ki).expect("Cannot analyze Phi");
        let def_use = ConstantPropagation::run(def_use);
        let mut la = LoopAnalysis::new(&func);
        la.analyze(&dom);
        assert_eq!(2, la.loops.len());
        let inner_loop = &la.loops[0];
        let outer_loop = &la.loops[1];
        assert_eq!(inner_loop.parent_loop, Some(1));
        assert!(outer_loop.is_loop_invariant(&Value::Instruction(1, 0), &def_use));
        // `s_add_i32 s4, s4, 1` is the step inst of outer loop
        let s4 = Value::Instruction(13, 0);
        let s4_old = def_use.get_def_use(InstructionUse::op(13, 2, 0)).unwrap();
        assert!(matches!(s4_old, Value::Phi(_)));
        assert!(!outer_loop.is_loop_invariant(&s4, &def_use));
        assert!(inner_loop.is_loop_invariant(&s4, &def_use));
        assert!(!outer_loop.is_loop_invariant(&s4_old, &def_use));
        assert!(inner_loop.is_loop_invariant(&s4_old, &def_use));
    }

    #[test]
    fn test_loop_invariant_of_phi() {
        const CODE: &[u32] = &[
            // BB0:
            0xBE800380, // s_mov_b32 s0, 0
            0xBF840001, // s_cbranch_scc0 BB2
            // BB1:
            0xBE800381, // s_mov_b32 s0, 1
            // BB2:
            0xBE810300, // s_mov_b32 s1, s0
            0xBF84FFFE, // s_cbranch_scc0 BB2
            0xBF810000, // s_endpgm
        ];
        let ki = simple_kernel_info("", CODE);
        let func = Disassembler::parse_kernel(&ki).expect("Failed to parse the function");
        let dom = DomTree::analyze(&func);
        let df = DomFrontier::new(&dom);
        let def_use = PHIAnalysis::analyze(&func, &dom, &df, &ki).expect("Cannot analyze Phi");
        let def_use = ConstantPropagation::run(def_use);
        let mut la = LoopAnalysis::new(&func);
        la.analyze(&dom);
        assert_eq!(1, la.loops.len());
        let loop_bb2 = &la.loops[0];
        let s0 = def_use.get_def_use(InstructionUse::op(3, 1, 0)).unwrap();
        assert!(matches!(s0, Value::Phi(_)));
        assert!(loop_bb2.is_loop_invariant(&s0, &def_use));
    }
}
