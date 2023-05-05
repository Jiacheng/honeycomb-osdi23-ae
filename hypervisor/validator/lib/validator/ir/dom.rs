use crate::ir::Function;
use petgraph::algo::dominators::simple_fast;
use petgraph::prelude::DfsPostOrder;
use petgraph::visit::{GraphBase, IntoNeighbors, Visitable};
use smallvec::{smallvec, SmallVec};
use std::collections::{HashMap, HashSet};
use std::iter::Iterator;
use std::slice::Iter;

#[derive(Clone)]
pub struct DomTree<'a> {
    func: &'a Function<'a>,
    root: usize,
    nodes: Vec<DomNode>,
}

pub struct CFG<'a>(&'a Function<'a>);

#[derive(Clone)]
pub struct DomNode {
    parent: Option<usize>,
    children: SmallVec<[usize; 4]>,
}

impl<'a> GraphBase for CFG<'a> {
    type EdgeId = usize;
    type NodeId = usize;
}

pub struct DomTreePostOrderIter<'a, 'b> {
    g: &'b DomTree<'a>,
    post_order: DfsPostOrder<usize, HashSet<usize>>,
}

impl Iterator for DomTreePostOrderIter<'_, '_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.post_order.next(self.g)
    }
}

impl<'a> GraphBase for DomTree<'a> {
    type EdgeId = usize;
    type NodeId = usize;
}

impl<'a> Visitable for DomTree<'a> {
    type Map = HashSet<usize>;

    fn visit_map(&self) -> Self::Map {
        Self::Map::new()
    }

    fn reset_map(&self, map: &mut Self::Map) {
        map.clear()
    }
}

impl<'a> DomTree<'a> {
    pub fn func(&self) -> &'a Function {
        self.func
    }
}

pub struct NeighborIter<'a, T> {
    iter: std::slice::Iter<'a, T>,
}

impl<'a, T> Iterator for NeighborIter<'a, T>
where
    T: Copy,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().copied()
    }
}

impl<'a, 'b> IntoNeighbors for &'b DomTree<'a> {
    type Neighbors = NeighborIter<'b, usize>;

    fn neighbors(self, n: Self::NodeId) -> Self::Neighbors {
        NeighborIter {
            iter: self.nodes[n].children.iter(),
        }
    }
}

impl<'a, 'b> IntoNeighbors for &'b CFG<'a> {
    type Neighbors = NeighborIter<'a, usize>;

    fn neighbors(self, n: Self::NodeId) -> Self::Neighbors {
        NeighborIter {
            iter: self.0.basic_blocks[n].successors.iter(),
        }
    }
}

impl Visitable for CFG<'_> {
    type Map = HashSet<usize>;

    fn visit_map(&self) -> Self::Map {
        Self::Map::new()
    }

    fn reset_map(&self, map: &mut Self::Map) {
        map.clear()
    }
}

impl<'a> CFG<'a> {
    pub fn new(func: &'a Function<'a>) -> CFG {
        CFG(func)
    }
}

impl<'a> DomTree<'a> {
    pub fn analyze(func: &'a Function) -> DomTree<'a> {
        let mut nodes = Vec::new();

        let cfg = CFG(func);
        let dom = simple_fast(&cfg, 0);
        let mut children: HashMap<usize, SmallVec<[usize; 4]>> = HashMap::new();
        for i in 0..func.basic_blocks.len() {
            let parent = dom.immediate_dominator(i);
            let n = DomNode {
                parent,
                children: SmallVec::new(),
            };
            if let Some(p) = parent {
                if let Some(v) = children.get_mut(&p) {
                    v.push(i);
                } else {
                    children.insert(p, smallvec![i]);
                }
            }
            nodes.push(n);
        }

        children.into_iter().for_each(|(parent, children)| {
            nodes[parent].children = children;
        });
        DomTree {
            func,
            root: 0,
            nodes,
        }
    }

    pub fn root(&self) -> usize {
        0
    }

    /**
     * Return true if a dominates b. Not a constant time operation.
     **/
    pub fn dominates(&self, a: usize, b: usize) -> bool {
        // A node trivially dominates itself.
        if a == b {
            return true;
        }

        if let Some(x) = self.immediate_dominator(b) {
            if x == a {
                return true;
            }
        } else if let Some(x) = self.immediate_dominator(a) {
            if x == b {
                return false;
            }
        }

        let mut idom = b;
        while let Some(next) = self.immediate_dominator(idom) {
            idom = next;
            if idom == a {
                return true;
            }
        }
        idom == a
    }

    pub fn immediate_dominator(&self, node: usize) -> Option<usize> {
        self.nodes[node].parent
    }

    pub fn post_order_iter(&self, node: usize) -> DomTreePostOrderIter {
        DomTreePostOrderIter {
            g: self,
            post_order: DfsPostOrder::new(self, node),
        }
    }

    pub fn is_reachable_from_entry(&self, node: usize) -> bool {
        node == self.root || self.immediate_dominator(node).is_some()
    }

    pub fn get_node(&self, node: usize) -> Option<&DomNode> {
        self.nodes.get(node)
    }

    /// properlyDominates - Returns true iff A dominates B and A != B.
    /// Note that this is not a constant time operation!
    ///
    pub fn properly_dominates(&self, a: usize, b: usize) -> bool {
        if a == b {
            false
        } else {
            self.dominates(a, b)
        }
    }
}

impl DomNode {
    pub fn parent(&self) -> Option<usize> {
        self.parent
    }

    pub fn children_iter(&self) -> Iter<'_, usize> {
        self.children.iter()
    }
}
