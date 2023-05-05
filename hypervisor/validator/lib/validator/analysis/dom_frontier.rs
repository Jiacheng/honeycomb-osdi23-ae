use crate::ir::DomTree;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct DomFrontier {
    pub(crate) frontiers: HashMap<usize, HashSet<usize>>,
}

impl DomFrontier {
    pub fn new(dom: &DomTree) -> DomFrontier {
        Self {
            frontiers: Self::analyze(dom),
        }
    }

    pub fn get_frontier(&self, bb_idx: usize) -> Option<&HashSet<usize>> {
        self.frontiers.get(&bb_idx)
    }

    fn analyze(dt: &DomTree) -> HashMap<usize, HashSet<usize>> {
        #[derive(Copy, Clone)]
        struct WorkItem {
            curr_bb: usize,
            parent_bb: Option<usize>,
        }

        impl WorkItem {
            fn new(curr_bb: usize, parent_bb: Option<usize>) -> Self {
                Self { curr_bb, parent_bb }
            }
        }

        let mut visited = HashSet::new();
        let mut frontiers: HashMap<usize, HashSet<usize>> = HashMap::new();
        let mut work_list = vec![WorkItem::new(dt.root(), None)];

        while let Some(w) = work_list.last().cloned() {
            let (bb_idx, parent_bb_idx) = (w.curr_bb, w.parent_bb);
            if visited.insert(bb_idx) {
                let curr_bb = &dt.func().basic_blocks[bb_idx];
                let ds: HashSet<usize> = curr_bb
                    .successors
                    .iter()
                    .filter(|s| dt.immediate_dominator(**s) != Some(bb_idx))
                    .copied()
                    .collect();

                // At this point, S is DFlocal.  Now we union in DFup's of our children...
                // Loop through and visit the nodes that Node immediately dominates (Node's
                // children in the IDomTree)
                if let Some(x) = frontiers.get_mut(&bb_idx) {
                    x.extend(ds.into_iter());
                } else {
                    frontiers.insert(bb_idx, ds);
                }
            }

            let n = dt.get_node(bb_idx).expect("Cannot find the DOM node");
            let to_be_visited = n
                .children_iter()
                .filter(|x| !visited.contains(*x))
                .map(|child| WorkItem::new(*child, Some(bb_idx)));
            let visit_child = to_be_visited.clone().count() > 0;

            // If all children are visited or there is any child then pop this block
            // from the workList.
            if !visit_child {
                if let Some(parent) = parent_bb_idx {
                    let local = frontiers.get(&bb_idx).map(|c| {
                        c.iter()
                            .filter(|child| !dt.properly_dominates(parent, **child))
                            .copied()
                            .collect::<HashSet<usize>>()
                    });
                    if let Some(x) = frontiers.get_mut(&parent) {
                        if let Some(l) = local {
                            x.extend(l.into_iter());
                        }
                    } else if let Some(l) = local {
                        frontiers.insert(parent, l);
                    }
                } else {
                    break;
                }
                work_list.pop();
            } else {
                work_list.extend(to_be_visited);
            }
        }
        frontiers
    }
}

#[cfg(test)]
mod tests {
    use crate::analysis::dom_frontier::DomFrontier;
    use crate::ir::{BasicBlock, DomTree, Function};
    use smallvec::ToSmallVec;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn test_dom_frontier() {
        fn bb(id: usize, preds: &[usize], succs: &[usize]) -> BasicBlock {
            BasicBlock {
                offset: id,
                predecessors: preds.to_smallvec(),
                successors: succs.to_smallvec(),
                instructions: Default::default(),
            }
        }

        // Test case from https://people.cs.pitt.edu/~jmisurda/teaching/cs1622/slides/cs2210-ssa.pdf
        let basic_blocks = vec![
            bb(0, &[], &[1]),
            bb(1, &[0], &[2]),
            bb(2, &[1, 5], &[3, 6]),
            bb(3, &[2], &[4, 5]),
            bb(4, &[3], &[5]),
            bb(5, &[3, 4], &[2]),
            bb(6, &[2], &[]),
        ];

        let func = Function {
            name: "test",
            basic_blocks,
            instructions: vec![],
        };
        let dom = DomTree::analyze(&func);
        let dt = DomFrontier::new(&dom);
        const EXPECTED: &[(usize, &[usize])] = &[
            (0, &[]),
            (1, &[]),
            (2, &[2]),
            (3, &[2]),
            (4, &[5]),
            (5, &[2]),
            (6, &[]),
        ];
        let expected = EXPECTED
            .into_iter()
            .map(|(bb_idx, frontiers)| {
                (
                    *bb_idx,
                    frontiers.iter().map(|x| *x).collect::<HashSet<usize>>(),
                )
            })
            .collect::<HashMap<usize, HashSet<usize>>>();
        assert_eq!(expected, dt.frontiers);
    }
}
