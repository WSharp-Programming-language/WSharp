//! Control Flow Graph utilities.

use crate::mir::{BasicBlockId, MirBody, TerminatorKind};
use std::collections::{HashMap, HashSet, VecDeque};

/// Represents the control flow graph structure.
#[derive(Clone, Debug)]
pub struct ControlFlowGraph {
    /// Predecessors for each block.
    pub predecessors: HashMap<BasicBlockId, Vec<BasicBlockId>>,
    /// Successors for each block.
    pub successors: HashMap<BasicBlockId, Vec<BasicBlockId>>,
}

impl ControlFlowGraph {
    /// Build a CFG from a MIR body.
    pub fn build(body: &MirBody) -> Self {
        let mut predecessors: HashMap<BasicBlockId, Vec<BasicBlockId>> = HashMap::new();
        let mut successors: HashMap<BasicBlockId, Vec<BasicBlockId>> = HashMap::new();

        // Initialize empty lists for all blocks
        for &block_id in body.basic_blocks.keys() {
            predecessors.insert(block_id, Vec::new());
            successors.insert(block_id, Vec::new());
        }

        // Build edges from terminators
        for (&block_id, block) in &body.basic_blocks {
            if let Some(ref term) = block.terminator {
                let succs = Self::terminator_successors(&term.kind);
                for &succ in &succs {
                    if let Some(preds) = predecessors.get_mut(&succ) {
                        preds.push(block_id);
                    }
                }
                successors.insert(block_id, succs);
            }
        }

        Self {
            predecessors,
            successors,
        }
    }

    /// Get successors of a terminator.
    fn terminator_successors(kind: &TerminatorKind) -> Vec<BasicBlockId> {
        match kind {
            TerminatorKind::Goto { target } => vec![*target],
            TerminatorKind::SwitchInt { targets, .. } => {
                let mut succs: Vec<BasicBlockId> =
                    targets.targets.iter().map(|(_, t)| *t).collect();
                succs.push(targets.otherwise);
                succs
            }
            TerminatorKind::Return => vec![],
            TerminatorKind::Unreachable => vec![],
            TerminatorKind::Call { target, .. } => target.iter().copied().collect(),
            TerminatorKind::Drop { target, .. } => vec![*target],
            TerminatorKind::Assert { target, .. } => vec![*target],
            TerminatorKind::Yield { resume, .. } => vec![*resume],
        }
    }

    /// Get predecessors of a block.
    pub fn predecessors(&self, block: BasicBlockId) -> &[BasicBlockId] {
        self.predecessors.get(&block).map_or(&[], |v| v.as_slice())
    }

    /// Get successors of a block.
    pub fn successors(&self, block: BasicBlockId) -> &[BasicBlockId] {
        self.successors.get(&block).map_or(&[], |v| v.as_slice())
    }

    /// Check if a block is reachable from the entry.
    pub fn is_reachable(&self, entry: BasicBlockId, target: BasicBlockId) -> bool {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(entry);

        while let Some(block) = queue.pop_front() {
            if block == target {
                return true;
            }
            if visited.insert(block) {
                for &succ in self.successors(block) {
                    queue.push_back(succ);
                }
            }
        }

        false
    }

    /// Compute all reachable blocks from entry.
    pub fn reachable_blocks(&self, entry: BasicBlockId) -> HashSet<BasicBlockId> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(entry);

        while let Some(block) = queue.pop_front() {
            if visited.insert(block) {
                for &succ in self.successors(block) {
                    queue.push_back(succ);
                }
            }
        }

        visited
    }
}

/// Dominator tree computation.
#[derive(Clone, Debug)]
pub struct Dominators {
    /// Immediate dominator for each block.
    pub idom: HashMap<BasicBlockId, BasicBlockId>,
    /// Dominance frontier for each block.
    pub frontier: HashMap<BasicBlockId, HashSet<BasicBlockId>>,
}

impl Dominators {
    /// Compute dominators using the simple iterative algorithm.
    pub fn compute(body: &MirBody, cfg: &ControlFlowGraph) -> Self {
        let entry = BasicBlockId::ENTRY;
        let blocks: Vec<BasicBlockId> = body.basic_blocks.keys().copied().collect();

        // Initialize dominators
        let mut dom: HashMap<BasicBlockId, HashSet<BasicBlockId>> = HashMap::new();
        for &block in &blocks {
            if block == entry {
                let mut set = HashSet::new();
                set.insert(entry);
                dom.insert(block, set);
            } else {
                dom.insert(block, blocks.iter().copied().collect());
            }
        }

        // Iterate until fixed point
        let mut changed = true;
        while changed {
            changed = false;
            for &block in &blocks {
                if block == entry {
                    continue;
                }

                let preds = cfg.predecessors(block);
                if preds.is_empty() {
                    continue;
                }

                // New dominators = intersection of predecessor dominators + self
                let mut new_dom: HashSet<BasicBlockId> = dom[&preds[0]].clone();
                for &pred in &preds[1..] {
                    new_dom = new_dom.intersection(&dom[&pred]).copied().collect();
                }
                new_dom.insert(block);

                if new_dom != dom[&block] {
                    dom.insert(block, new_dom);
                    changed = true;
                }
            }
        }

        // Compute immediate dominators
        let mut idom: HashMap<BasicBlockId, BasicBlockId> = HashMap::new();
        for &block in &blocks {
            if block == entry {
                continue;
            }

            let mut doms: Vec<BasicBlockId> = dom[&block].iter().copied().collect();
            doms.retain(|&d| d != block);

            // Find the immediate dominator (the one that dominates all others)
            for &candidate in &doms {
                let is_idom = doms.iter().all(|&other| {
                    other == candidate || dom[&candidate].contains(&other)
                });
                if is_idom {
                    idom.insert(block, candidate);
                    break;
                }
            }
        }

        // Compute dominance frontiers
        let mut frontier: HashMap<BasicBlockId, HashSet<BasicBlockId>> = HashMap::new();
        for &block in &blocks {
            frontier.insert(block, HashSet::new());
        }

        for &block in &blocks {
            let preds = cfg.predecessors(block);
            if preds.len() >= 2 {
                for &pred in preds {
                    let mut runner = pred;
                    while runner != idom.get(&block).copied().unwrap_or(entry) {
                        frontier.get_mut(&runner).unwrap().insert(block);
                        runner = match idom.get(&runner) {
                            Some(&i) => i,
                            None => break,
                        };
                    }
                }
            }
        }

        Self { idom, frontier }
    }

    /// Check if block A dominates block B.
    pub fn dominates(&self, a: BasicBlockId, b: BasicBlockId) -> bool {
        if a == b {
            return true;
        }

        let mut current = b;
        while let Some(&idom) = self.idom.get(&current) {
            if idom == a {
                return true;
            }
            current = idom;
        }

        false
    }
}

/// Loop detection and analysis.
#[derive(Clone, Debug)]
pub struct LoopInfo {
    /// Back edges in the CFG (target, source).
    pub back_edges: Vec<(BasicBlockId, BasicBlockId)>,
    /// Blocks in each loop, keyed by loop header.
    pub loops: HashMap<BasicBlockId, HashSet<BasicBlockId>>,
}

impl LoopInfo {
    /// Detect natural loops in the CFG.
    pub fn compute(cfg: &ControlFlowGraph, dominators: &Dominators) -> Self {
        let mut back_edges = Vec::new();
        let mut loops: HashMap<BasicBlockId, HashSet<BasicBlockId>> = HashMap::new();

        // Find back edges (edges where target dominates source)
        for (&block, succs) in &cfg.successors {
            for &succ in succs {
                if dominators.dominates(succ, block) {
                    back_edges.push((succ, block));
                }
            }
        }

        // For each back edge, compute the natural loop
        for &(header, back_edge_source) in &back_edges {
            let mut loop_blocks = HashSet::new();
            loop_blocks.insert(header);

            // Find all blocks that can reach back_edge_source without going through header
            let mut stack = vec![back_edge_source];
            while let Some(block) = stack.pop() {
                if loop_blocks.insert(block) {
                    for pred in cfg.predecessors(block) {
                        if *pred != header {
                            stack.push(*pred);
                        }
                    }
                }
            }

            loops.insert(header, loop_blocks);
        }

        Self { back_edges, loops }
    }

    /// Get the loop header for a block, if any.
    pub fn loop_header(&self, block: BasicBlockId) -> Option<BasicBlockId> {
        for (&header, blocks) in &self.loops {
            if blocks.contains(&block) {
                return Some(header);
            }
        }
        None
    }
}

/// Post-order traversal of the CFG.
pub fn post_order(body: &MirBody, cfg: &ControlFlowGraph) -> Vec<BasicBlockId> {
    let mut visited = HashSet::new();
    let mut result = Vec::new();

    fn visit(
        block: BasicBlockId,
        cfg: &ControlFlowGraph,
        visited: &mut HashSet<BasicBlockId>,
        result: &mut Vec<BasicBlockId>,
    ) {
        if visited.insert(block) {
            for &succ in cfg.successors(block) {
                visit(succ, cfg, visited, result);
            }
            result.push(block);
        }
    }

    visit(BasicBlockId::ENTRY, cfg, &mut visited, &mut result);
    result
}

/// Reverse post-order traversal (good for forward dataflow).
pub fn reverse_post_order(body: &MirBody, cfg: &ControlFlowGraph) -> Vec<BasicBlockId> {
    let mut order = post_order(body, cfg);
    order.reverse();
    order
}
