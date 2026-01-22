//! Control Flow Graph utilities.

use crate::mir::{
    BasicBlockId, Local, MirBody, Operand, Place, Rvalue, StatementKind, TerminatorKind,
};
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

// =============================================================================
// Live Variable Analysis
// =============================================================================

/// Live Variable Analysis result.
///
/// Computes which locals are "live" at each program point. A variable is live
/// if it may be used before being redefined. This is a backward dataflow analysis.
#[derive(Clone, Debug)]
pub struct LiveVariableAnalysis {
    /// Live variables at the entry of each block (before any statements execute).
    pub live_in: HashMap<BasicBlockId, HashSet<Local>>,
    /// Live variables at the exit of each block (after terminator).
    pub live_out: HashMap<BasicBlockId, HashSet<Local>>,
}

impl LiveVariableAnalysis {
    /// Compute live variable analysis for a MIR body.
    ///
    /// Uses iterative backward dataflow analysis with equations:
    /// - live_out[B] = ∪ live_in[S] for all successors S of B
    /// - live_in[B] = use[B] ∪ (live_out[B] - def[B])
    pub fn compute(body: &MirBody, cfg: &ControlFlowGraph) -> Self {
        let mut live_in: HashMap<BasicBlockId, HashSet<Local>> = HashMap::new();
        let mut live_out: HashMap<BasicBlockId, HashSet<Local>> = HashMap::new();

        // Initialize all blocks with empty sets
        for &block_id in body.basic_blocks.keys() {
            live_in.insert(block_id, HashSet::new());
            live_out.insert(block_id, HashSet::new());
        }

        // Compute use and def sets for each block
        let mut use_sets: HashMap<BasicBlockId, HashSet<Local>> = HashMap::new();
        let mut def_sets: HashMap<BasicBlockId, HashSet<Local>> = HashMap::new();

        for (&block_id, block) in &body.basic_blocks {
            let (uses, defs) = Self::compute_use_def(block, body);
            use_sets.insert(block_id, uses);
            def_sets.insert(block_id, defs);
        }

        // Iterate until fixed point (backward analysis, so use post-order)
        let order = post_order(body, cfg);
        let mut changed = true;
        while changed {
            changed = false;

            for &block_id in &order {
                // live_out[B] = ∪ live_in[S] for all successors S
                let mut new_live_out = HashSet::new();
                for &succ in cfg.successors(block_id) {
                    if let Some(succ_live_in) = live_in.get(&succ) {
                        new_live_out.extend(succ_live_in.iter().copied());
                    }
                }

                // live_in[B] = use[B] ∪ (live_out[B] - def[B])
                let uses = use_sets.get(&block_id).cloned().unwrap_or_default();
                let defs = def_sets.get(&block_id).cloned().unwrap_or_default();
                let mut new_live_in: HashSet<Local> =
                    new_live_out.difference(&defs).copied().collect();
                new_live_in.extend(uses.iter());

                // Check for changes
                if new_live_out != *live_out.get(&block_id).unwrap() {
                    live_out.insert(block_id, new_live_out);
                    changed = true;
                }
                if new_live_in != *live_in.get(&block_id).unwrap() {
                    live_in.insert(block_id, new_live_in);
                    changed = true;
                }
            }
        }

        Self { live_in, live_out }
    }

    /// Compute the use and def sets for a basic block.
    ///
    /// - `use`: Variables used before being defined in this block
    /// - `def`: Variables defined in this block
    fn compute_use_def(
        block: &crate::mir::BasicBlock,
        _body: &MirBody,
    ) -> (HashSet<Local>, HashSet<Local>) {
        let mut uses = HashSet::new();
        let mut defs = HashSet::new();

        // Process statements in order
        for stmt in &block.statements {
            match &stmt.kind {
                StatementKind::Assign(place, rvalue) => {
                    // First, collect uses from rvalue (uses come before defs)
                    Self::collect_rvalue_uses(rvalue, &mut uses, &defs);

                    // Then record definition (but only the base local)
                    if place.projection.is_empty() {
                        defs.insert(place.local);
                    } else {
                        // For projections like _1.0, we use _1 but don't def it completely
                        if !defs.contains(&place.local) {
                            uses.insert(place.local);
                        }
                    }
                }
                StatementKind::StorageLive(local) => {
                    // StorageLive doesn't use or def the value
                    let _ = local;
                }
                StatementKind::StorageDead(local) => {
                    // StorageDead doesn't use or def the value
                    let _ = local;
                }
                StatementKind::Nop => {}
            }
        }

        // Process terminator
        if let Some(ref term) = block.terminator {
            Self::collect_terminator_uses(&term.kind, &mut uses, &defs);
        }

        (uses, defs)
    }

    /// Collect locals used by an rvalue.
    fn collect_rvalue_uses(rvalue: &Rvalue, uses: &mut HashSet<Local>, defs: &HashSet<Local>) {
        match rvalue {
            Rvalue::Use(op) => Self::collect_operand_uses(op, uses, defs),
            Rvalue::Repeat(op, _) => Self::collect_operand_uses(op, uses, defs),
            Rvalue::Ref(place, _) | Rvalue::AddressOf(place, _) => {
                Self::collect_place_uses(place, uses, defs);
            }
            Rvalue::Len(place) => Self::collect_place_uses(place, uses, defs),
            Rvalue::BinaryOp(_, op1, op2) | Rvalue::CheckedBinaryOp(_, op1, op2) => {
                Self::collect_operand_uses(op1, uses, defs);
                Self::collect_operand_uses(op2, uses, defs);
            }
            Rvalue::UnaryOp(_, op) => Self::collect_operand_uses(op, uses, defs),
            Rvalue::NullaryOp(_, _) => {}
            Rvalue::Cast(_, op, _) => Self::collect_operand_uses(op, uses, defs),
            Rvalue::Discriminant(place) => Self::collect_place_uses(place, uses, defs),
            Rvalue::Aggregate(_, ops) => {
                for op in ops {
                    Self::collect_operand_uses(op, uses, defs);
                }
            }
            Rvalue::ShallowInitBox(op, _) => Self::collect_operand_uses(op, uses, defs),
        }
    }

    /// Collect locals used by an operand.
    fn collect_operand_uses(operand: &Operand, uses: &mut HashSet<Local>, defs: &HashSet<Local>) {
        match operand {
            Operand::Copy(place) | Operand::Move(place) => {
                Self::collect_place_uses(place, uses, defs);
            }
            Operand::Constant(_) => {}
        }
    }

    /// Collect locals used by a place (including projections).
    fn collect_place_uses(place: &Place, uses: &mut HashSet<Local>, defs: &HashSet<Local>) {
        // The base local is used
        if !defs.contains(&place.local) {
            uses.insert(place.local);
        }

        // Index projections also use a local
        for elem in &place.projection {
            if let crate::mir::PlaceElem::Index(idx) = elem {
                if !defs.contains(idx) {
                    uses.insert(*idx);
                }
            }
        }
    }

    /// Collect locals used by a terminator.
    fn collect_terminator_uses(
        kind: &TerminatorKind,
        uses: &mut HashSet<Local>,
        defs: &HashSet<Local>,
    ) {
        match kind {
            TerminatorKind::Goto { .. } => {}
            TerminatorKind::SwitchInt { discr, .. } => {
                Self::collect_operand_uses(discr, uses, defs);
            }
            TerminatorKind::Return => {
                // Return uses the return place (_0)
                if !defs.contains(&Local::RETURN_PLACE) {
                    uses.insert(Local::RETURN_PLACE);
                }
            }
            TerminatorKind::Unreachable => {}
            TerminatorKind::Call {
                func,
                args,
                destination,
                ..
            } => {
                Self::collect_operand_uses(func, uses, defs);
                for arg in args {
                    Self::collect_operand_uses(arg, uses, defs);
                }
                // The destination is a def, but handled in next block
                let _ = destination;
            }
            TerminatorKind::Drop { place, .. } => {
                Self::collect_place_uses(place, uses, defs);
            }
            TerminatorKind::Assert { cond, .. } => {
                Self::collect_operand_uses(cond, uses, defs);
            }
            TerminatorKind::Yield { value, .. } => {
                Self::collect_operand_uses(value, uses, defs);
            }
        }
    }

    /// Get the set of locals that are live across a yield point.
    ///
    /// A local is live across a yield if it's in live_out of the yield block
    /// AND live_in of the resume block (i.e., it's used after the yield).
    pub fn locals_across_yield(
        &self,
        yield_block: BasicBlockId,
        resume_block: BasicBlockId,
    ) -> HashSet<Local> {
        let live_at_yield = self.live_out.get(&yield_block).cloned().unwrap_or_default();
        let live_at_resume = self.live_in.get(&resume_block).cloned().unwrap_or_default();

        // Locals that are live at both points need to be saved
        live_at_yield
            .intersection(&live_at_resume)
            .copied()
            .collect()
    }

    /// Get all locals that are live at any yield point in the function.
    ///
    /// This is useful for determining which locals need to be stored in
    /// the coroutine state struct.
    pub fn all_locals_across_yields(&self, body: &MirBody, cfg: &ControlFlowGraph) -> HashSet<Local> {
        let mut result = HashSet::new();

        for (&block_id, block) in &body.basic_blocks {
            if let Some(ref term) = block.terminator {
                if let TerminatorKind::Yield { resume, .. } = term.kind {
                    let locals = self.locals_across_yield(block_id, resume);
                    result.extend(locals);
                }
            }
        }

        // Also need to consider locals that span multiple yield points via loops
        // For now, this simple version captures the basics
        let _ = cfg; // May be needed for more sophisticated analysis

        result
    }

    /// Check if a local is live at the entry of a block.
    pub fn is_live_at_entry(&self, local: Local, block: BasicBlockId) -> bool {
        self.live_in
            .get(&block)
            .map_or(false, |set| set.contains(&local))
    }

    /// Check if a local is live at the exit of a block.
    pub fn is_live_at_exit(&self, local: Local, block: BasicBlockId) -> bool {
        self.live_out
            .get(&block)
            .map_or(false, |set| set.contains(&local))
    }
}
