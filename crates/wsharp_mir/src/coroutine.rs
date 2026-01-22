//! Coroutine transformation for async/await.
//!
//! This module transforms async functions into state machines. The transformation:
//! 1. Discovers yield points (await expressions)
//! 2. Computes which locals are live across yield points
//! 3. Builds a state struct layout with saved locals
//! 4. Transforms the function into a poll function with state machine dispatch

use crate::cfg::{ControlFlowGraph, Dominators, LiveVariableAnalysis, LoopInfo};
use crate::mir::{
    AggregateKind, BasicBlockId, BodyId, Constant, CoroutineInfo, CoroutineStateLayout, Local,
    LocalDecl, LoopContext, MirBody, NestedCoroutineInfo, Operand, Place, Rvalue, Statement,
    StatementKind, SwitchTargets, Terminator, TerminatorKind, YieldPointInfo,
};
use std::collections::HashMap;
use wsharp_types::{PrimitiveType, Type};

/// The coroutine transformer.
///
/// Converts an async MirBody into a state machine suitable for polling.
pub struct CoroutineTransformer {
    /// The original function name.
    name: String,

    /// The return type of the async function (inner type of Future<T>).
    result_type: Type,

    /// Yield points discovered in the function.
    yield_points: Vec<YieldPointInfo>,

    /// The state layout being built.
    state_layout: CoroutineStateLayout,

    /// Mapping from original locals to state field indices.
    local_to_state_field: HashMap<Local, usize>,

    /// Loop contexts for loops containing yields.
    loop_contexts: Vec<LoopContext>,

    /// Nested coroutines (async closures).
    nested_coroutines: Vec<NestedCoroutineInfo>,

    /// Next state index to assign.
    next_state_index: u32,
}

impl CoroutineTransformer {
    /// Create a new transformer for an async function.
    pub fn new(name: String, result_type: Type) -> Self {
        Self {
            name,
            result_type: result_type.clone(),
            yield_points: Vec::new(),
            state_layout: CoroutineStateLayout::new(result_type),
            local_to_state_field: HashMap::new(),
            loop_contexts: Vec::new(),
            nested_coroutines: Vec::new(),
            next_state_index: 1, // State 0 is initial
        }
    }

    /// Transform an async MirBody into a coroutine state machine.
    ///
    /// This is the main entry point for the transformation.
    pub fn transform(mut self, mut body: MirBody) -> MirBody {
        // Step 1: Build CFG and compute analyses
        let cfg = ControlFlowGraph::build(&body);
        let live_analysis = LiveVariableAnalysis::compute(&body, &cfg);
        let dominators = Dominators::compute(&body, &cfg);
        let loop_info = LoopInfo::compute(&cfg, &dominators);

        // Step 2: Discover yield points
        self.discover_yield_points(&body);

        if self.yield_points.is_empty() {
            // No yield points - async function that never awaits
            // Still needs to be wrapped but simpler case
            return self.transform_trivial_async(body);
        }

        // Step 3: Compute locals that need to be saved
        self.compute_saved_locals(&body, &live_analysis);

        // Step 4: Analyze loops containing yields
        self.analyze_loops_with_yields(&body, &cfg, &loop_info);

        // Step 5: Transform the body into a state machine
        self.transform_to_state_machine(&mut body, &cfg);

        // Step 6: Attach coroutine info to body
        body.coroutine_info = Some(self.build_coroutine_info(&body));

        body
    }

    /// Discover all yield points in the function.
    fn discover_yield_points(&mut self, body: &MirBody) {
        for (&block_id, block) in &body.basic_blocks {
            if let Some(ref term) = block.terminator {
                if let TerminatorKind::Yield { value, resume } = &term.kind {
                    let state_index = self.next_state_index;
                    self.next_state_index += 1;

                    self.yield_points.push(YieldPointInfo {
                        yield_block: block_id,
                        resume_block: *resume,
                        state_index,
                        live_locals: Vec::new(), // Filled in later
                        yielded_value: Some(value.clone()),
                        resume_result: None, // TODO: track where result goes
                    });
                }
            }
        }
    }

    /// Compute which locals need to be saved in the state struct.
    fn compute_saved_locals(&mut self, body: &MirBody, live_analysis: &LiveVariableAnalysis) {
        // For each yield point, find locals live across it
        for yield_info in &mut self.yield_points {
            let live_locals = live_analysis.locals_across_yield(
                yield_info.yield_block,
                yield_info.resume_block,
            );

            // Filter out return place and arguments (they're handled differently)
            let saved_locals: Vec<Local> = live_locals
                .into_iter()
                .filter(|&local| {
                    local != Local::RETURN_PLACE && local.0 as usize > body.arg_count
                })
                .collect();

            yield_info.live_locals = saved_locals.clone();

            // Add fields for locals not yet in state layout
            for local in saved_locals {
                if !self.local_to_state_field.contains_key(&local) {
                    let local_ty = body.locals[local.0 as usize].ty.clone();
                    let field_name = format!("__local_{}", local.0);
                    let field_idx = self.state_layout.add_local_field(field_name, local_ty);
                    self.local_to_state_field.insert(local, field_idx);
                }
            }
        }
    }

    /// Analyze loops that contain yield points.
    ///
    /// For each loop containing await expressions, we need to:
    /// 1. Identify break targets (exits from the loop)
    /// 2. Identify continue targets (back-edges to loop header)
    /// 3. Find loop-carried locals that need preservation
    fn analyze_loops_with_yields(
        &mut self,
        body: &MirBody,
        cfg: &ControlFlowGraph,
        loop_info: &LoopInfo,
    ) {
        for (&header, loop_blocks) in &loop_info.loops {
            // Check if any yield point is in this loop
            let yields_in_loop: Vec<u32> = self
                .yield_points
                .iter()
                .filter(|yp| loop_blocks.contains(&yp.yield_block))
                .map(|yp| yp.state_index)
                .collect();

            if !yields_in_loop.is_empty() {
                // This loop contains yields - needs special handling
                let header_state = self.next_state_index;
                self.next_state_index += 1;
                let exit_state = self.next_state_index;
                self.next_state_index += 1;

                // Find break targets: blocks outside the loop that are successors of loop blocks
                let mut break_targets = Vec::new();
                for &block in loop_blocks {
                    for &succ in cfg.successors(block) {
                        if !loop_blocks.contains(&succ) && succ != header {
                            if !break_targets.contains(&succ) {
                                break_targets.push(succ);
                            }
                        }
                    }
                }

                // Continue targets: the loop header (back-edges)
                let continue_targets = vec![header];

                // Find loop-carried locals: variables that are live at the back-edge
                // These are locals that are defined in the loop and used in subsequent iterations
                let mut loop_carried_locals = Vec::new();
                for &block in loop_blocks {
                    // Check if this block has a back-edge to header
                    if cfg.successors(block).contains(&header) {
                        // This is a back-edge source - locals live here need to be preserved
                        for yield_info in &self.yield_points {
                            if loop_blocks.contains(&yield_info.yield_block) {
                                for &local in &yield_info.live_locals {
                                    if !loop_carried_locals.contains(&local) {
                                        loop_carried_locals.push(local);
                                    }
                                }
                            }
                        }
                    }
                }

                // Ensure loop-carried locals are in the state struct
                for &local in &loop_carried_locals {
                    if !self.local_to_state_field.contains_key(&local) {
                        let local_ty = body.locals[local.0 as usize].ty.clone();
                        let field_name = format!("__loop_local_{}", local.0);
                        let field_idx = self.state_layout.add_local_field(field_name, local_ty);
                        self.local_to_state_field.insert(local, field_idx);
                    }
                }

                self.loop_contexts.push(LoopContext {
                    header_state,
                    exit_state,
                    break_targets,
                    continue_targets,
                    contained_yields: yields_in_loop,
                    loop_carried_locals,
                });
            }
        }
    }

    /// Transform break statements within loops containing yields.
    ///
    /// Break becomes: save state → set __state to exit_state → goto pending
    /// On resume at exit_state, execution continues after the loop.
    fn transform_loop_breaks(
        &self,
        body: &mut MirBody,
        loop_ctx: &LoopContext,
        state_ptr_local: Local,
        pending_block: BasicBlockId,
    ) {
        // Find all Goto terminators that target break_targets from within the loop
        let blocks_to_transform: Vec<(BasicBlockId, BasicBlockId)> = body
            .basic_blocks
            .iter()
            .filter_map(|(&block_id, block)| {
                if let Some(ref term) = block.terminator {
                    if let TerminatorKind::Goto { target } = term.kind {
                        if loop_ctx.break_targets.contains(&target) {
                            return Some((block_id, target));
                        }
                    }
                }
                None
            })
            .collect();

        for (block_id, original_target) in blocks_to_transform {
            // Create a new block for the break transformation
            let break_transform_block = body.new_basic_block();

            // Collect local types first (before borrowing block)
            let local_saves: Vec<(Local, usize, Type)> = loop_ctx
                .loop_carried_locals
                .iter()
                .filter_map(|&local| {
                    self.local_to_state_field.get(&local).map(|&field_idx| {
                        let local_ty = body.locals[local.0 as usize].ty.clone();
                        (local, field_idx, local_ty)
                    })
                })
                .collect();

            // Set up the break transform block
            {
                let block = body.block_mut(break_transform_block);

                // Save loop-carried locals
                for (local, field_idx, local_ty) in local_saves {
                    let state_field =
                        Place::local(state_ptr_local).deref().field(field_idx, local_ty);
                    block.push_stmt(Statement {
                        kind: StatementKind::Assign(
                            state_field,
                            Rvalue::Use(Operand::Copy(Place::local(local))),
                        ),
                    });
                }

                // Set __state to exit_state
                let state_field = Place::local(state_ptr_local).deref().field(
                    CoroutineStateLayout::state_field_index(),
                    Type::Primitive(PrimitiveType::U32),
                );
                block.push_stmt(Statement {
                    kind: StatementKind::Assign(
                        state_field,
                        Rvalue::Use(Operand::Constant(Constant::Int(
                            loop_ctx.exit_state as i128,
                            Type::Primitive(PrimitiveType::U32),
                        ))),
                    ),
                });

                // Goto pending block (or directly to the original target for sync break)
                // For now, just goto the original target since we're not fully async yet
                block.set_terminator(Terminator {
                    kind: TerminatorKind::Goto {
                        target: original_target,
                    },
                });
            }

            // Update the original block to goto the transform block
            let original_block = body.block_mut(block_id);
            original_block.set_terminator(Terminator {
                kind: TerminatorKind::Goto {
                    target: break_transform_block,
                },
            });
        }

        let _ = pending_block; // Will be used when full async break is implemented
    }

    /// Transform continue statements within loops containing yields.
    ///
    /// Continue becomes: save state → set __state to header_state → goto pending
    /// On resume at header_state, execution continues at loop header.
    fn transform_loop_continues(
        &self,
        body: &mut MirBody,
        loop_ctx: &LoopContext,
        state_ptr_local: Local,
        _pending_block: BasicBlockId,
    ) {
        // Find all back-edges (Goto to header from within the loop)
        let header = loop_ctx.continue_targets.first().copied();
        if header.is_none() {
            return;
        }
        let header = header.unwrap();

        // For continues, we need to ensure loop-carried locals are saved
        // The actual transformation is similar to breaks but targets header_state
        let blocks_to_transform: Vec<BasicBlockId> = body
            .basic_blocks
            .iter()
            .filter_map(|(&block_id, block)| {
                if let Some(ref term) = block.terminator {
                    if let TerminatorKind::Goto { target } = term.kind {
                        if target == header {
                            return Some(block_id);
                        }
                    }
                }
                None
            })
            .collect();

        for block_id in blocks_to_transform {
            // For continues, we insert state saves before the goto
            // but keep the goto to header (for now)
            let local_saves: Vec<(Local, usize, Type)> = loop_ctx
                .loop_carried_locals
                .iter()
                .filter_map(|&local| {
                    self.local_to_state_field.get(&local).map(|&field_idx| {
                        let local_ty = body.locals[local.0 as usize].ty.clone();
                        (local, field_idx, local_ty)
                    })
                })
                .collect();

            let block = body.block_mut(block_id);

            // Insert saves before the terminator
            for (local, field_idx, local_ty) in local_saves {
                let state_field =
                    Place::local(state_ptr_local).deref().field(field_idx, local_ty);
                // Insert at the end of statements (before terminator)
                block.statements.push(Statement {
                    kind: StatementKind::Assign(
                        state_field,
                        Rvalue::Use(Operand::Copy(Place::local(local))),
                    ),
                });
            }
        }
    }

    // =========================================================================
    // Nested Async Support (Task 7)
    // =========================================================================

    /// Find all closures in the body that contain await expressions.
    ///
    /// These closures need to be transformed into nested coroutines.
    fn find_async_closures(&self, body: &MirBody) -> Vec<BodyId> {
        let mut async_closures = Vec::new();

        // Look for closure aggregates in the MIR
        for (_, block) in &body.basic_blocks {
            for stmt in &block.statements {
                if let StatementKind::Assign(_, rvalue) = &stmt.kind {
                    if let Rvalue::Aggregate(AggregateKind::Closure { body_id, .. }, _) = rvalue {
                        // Check if this closure's body is async or contains await
                        // For now, we track all closures and check async flag later
                        async_closures.push(*body_id);
                    }
                }
            }
        }

        async_closures
    }

    /// Detect nested async closures and prepare them for transformation.
    ///
    /// This should be called before the main transformation to handle
    /// bottom-up transformation of nested coroutines.
    fn detect_nested_async(&mut self, body: &MirBody, module: &crate::mir::MirModule) {
        let closure_ids = self.find_async_closures(body);

        for closure_body_id in closure_ids {
            if let Some(closure_body) = module.bodies.get(&closure_body_id) {
                if closure_body.is_async {
                    // This closure is async - it needs to be a nested coroutine
                    // Add a field for its state in the parent's state struct
                    let state_field_idx = self.state_layout.add_local_field(
                        format!("__nested_coroutine_{}", closure_body_id.0),
                        Type::Unknown, // Will be replaced with actual nested state type
                    );

                    // Collect captures (locals from outer scope used in closure)
                    let captures = self.find_closure_captures(body, closure_body_id);

                    self.nested_coroutines.push(NestedCoroutineInfo {
                        closure_body_id,
                        state_field_index: state_field_idx,
                        captures,
                        transformed: false,
                    });
                }
            }
        }
    }

    /// Find which locals from the outer scope are captured by a closure.
    fn find_closure_captures(&self, body: &MirBody, closure_body_id: BodyId) -> Vec<Local> {
        let mut captures = Vec::new();

        // Look for the closure creation to find captured variables
        for (_, block) in &body.basic_blocks {
            for stmt in &block.statements {
                if let StatementKind::Assign(_, rvalue) = &stmt.kind {
                    if let Rvalue::Aggregate(
                        AggregateKind::Closure { body_id, .. },
                        operands,
                    ) = rvalue
                    {
                        if *body_id == closure_body_id {
                            // The operands are the captured values
                            for op in operands {
                                if let Operand::Copy(place) | Operand::Move(place) = op {
                                    if place.projection.is_empty() {
                                        captures.push(place.local);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        captures
    }

    /// Transform a nested async closure to a coroutine.
    ///
    /// This creates a separate coroutine for the closure and stores
    /// its state as a field in the parent's state struct.
    fn transform_nested_coroutine(
        &mut self,
        nested_info: &mut NestedCoroutineInfo,
        closure_body: &mut MirBody,
    ) {
        if nested_info.transformed {
            return;
        }

        // Transform the closure body using a new transformer
        let result_type = match &closure_body.return_ty {
            Type::Future(inner) => (**inner).clone(),
            other => other.clone(),
        };

        let transformer = CoroutineTransformer::new(
            format!("__nested_{}", nested_info.closure_body_id.0),
            result_type,
        );

        // Note: This would need the full module context to properly transform
        // For now, we mark it as transformed and the actual transformation
        // happens in transform_async_functions
        nested_info.transformed = true;
    }

    /// Add statements to poll a nested coroutine when awaited.
    ///
    /// When the parent awaits on a nested async closure, we need to:
    /// 1. Poll the nested coroutine
    /// 2. If Pending, save parent state and return Pending
    /// 3. If Ready, extract the value and continue
    fn generate_nested_poll(
        &self,
        body: &mut MirBody,
        nested_info: &NestedCoroutineInfo,
        state_ptr_local: Local,
        result_local: Local,
        resume_block: BasicBlockId,
        pending_block: BasicBlockId,
    ) {
        // This is a placeholder for the actual nested poll generation
        // The full implementation would:
        // 1. Get pointer to nested state from parent state struct
        // 2. Call the nested coroutine's poll function
        // 3. Check if result is Pending or Ready
        // 4. Branch accordingly

        let _ = (body, nested_info, state_ptr_local, result_local, resume_block, pending_block);

        // For now, nested async closures are transformed independently
        // and the await on them works like any other await
    }

    /// Transform a trivial async function (no await points).
    fn transform_trivial_async(self, mut body: MirBody) -> MirBody {
        // For a trivial async function, we just wrap the result in Ready
        // The function immediately returns Ready(result)

        let coroutine_info = CoroutineInfo {
            state_layout: self.state_layout,
            local_to_state_field: self.local_to_state_field,
            yield_points: Vec::new(),
            loop_contexts: Vec::new(),
            nested_coroutines: Vec::new(),
            poll_entry_block: BasicBlockId::ENTRY,
            ready_block: BasicBlockId::ENTRY, // No separate ready block needed
            pending_block: BasicBlockId::ENTRY, // Not used
        };

        body.coroutine_info = Some(coroutine_info);
        body
    }

    /// Transform the body into a state machine.
    fn transform_to_state_machine(&mut self, body: &mut MirBody, _cfg: &ControlFlowGraph) {
        // Create the dispatch block (entry point that switches on __state)
        let dispatch_block = body.new_basic_block();

        // Create pending return block
        let pending_block = body.new_basic_block();
        self.setup_pending_block(body, pending_block);

        // Create ready return block
        let ready_block = body.new_basic_block();
        self.setup_ready_block(body, ready_block);

        // Build switch targets for dispatch
        let mut switch_targets: Vec<(u128, BasicBlockId)> = Vec::new();

        // State 0 -> original entry block
        switch_targets.push((0, BasicBlockId::ENTRY));

        // State N -> resume block for yield point N
        for yield_info in &self.yield_points {
            switch_targets.push((
                yield_info.state_index as u128,
                yield_info.resume_block,
            ));
        }

        // Add state parameter as first local (the state struct pointer)
        // Use Ref type to represent a mutable reference to the state
        let state_ptr_local = body.add_local(
            Type::Ref {
                inner: Box::new(Type::Unknown), // Will be replaced with actual state type
                mutable: true,
            },
            Some("__state_ptr".to_string()),
            false,
        );

        // Setup dispatch block with switch on state
        self.setup_dispatch_block(body, dispatch_block, state_ptr_local, &switch_targets, pending_block);

        // Transform each yield point
        for yield_info in self.yield_points.clone() {
            self.transform_yield_point(body, &yield_info, state_ptr_local, pending_block);
        }

        // Transform loops containing yields (break/continue handling)
        for loop_ctx in self.loop_contexts.clone() {
            self.transform_loop_breaks(body, &loop_ctx, state_ptr_local, pending_block);
            self.transform_loop_continues(body, &loop_ctx, state_ptr_local, pending_block);
        }

        // Transform return statements to set result and goto ready block
        self.transform_returns(body, state_ptr_local, ready_block);
    }

    /// Setup the dispatch block with state switch.
    fn setup_dispatch_block(
        &self,
        body: &mut MirBody,
        dispatch_block: BasicBlockId,
        state_ptr_local: Local,
        switch_targets: &[(u128, BasicBlockId)],
        otherwise: BasicBlockId,
    ) {
        // First, add the local for state value (before borrowing block)
        let state_local = Local(body.locals.len() as u32);
        body.locals.push(LocalDecl {
            ty: Type::Primitive(PrimitiveType::U32),
            name: Some("__state_value".to_string()),
            mutable: false,
        });

        // _state_value = (*state_ptr).__state
        let state_field_place = Place::local(state_ptr_local)
            .deref()
            .field(
                CoroutineStateLayout::state_field_index(),
                Type::Primitive(PrimitiveType::U32),
            );

        // Now borrow the block and set it up
        let block = body.block_mut(dispatch_block);

        block.push_stmt(Statement {
            kind: StatementKind::Assign(
                Place::local(state_local),
                Rvalue::Use(Operand::Copy(state_field_place)),
            ),
        });

        // Switch on state value
        block.set_terminator(Terminator {
            kind: TerminatorKind::SwitchInt {
                discr: Operand::Copy(Place::local(state_local)),
                targets: SwitchTargets::new(switch_targets.to_vec(), otherwise),
            },
        });
    }

    /// Setup the pending return block.
    fn setup_pending_block(&self, body: &mut MirBody, pending_block: BasicBlockId) {
        let block = body.block_mut(pending_block);

        // Set return value to Pending (discriminant 0)
        // For now, just return - codegen will handle PollResult
        block.set_terminator(Terminator {
            kind: TerminatorKind::Return,
        });
    }

    /// Setup the ready return block.
    fn setup_ready_block(&self, body: &mut MirBody, ready_block: BasicBlockId) {
        let block = body.block_mut(ready_block);

        // Return with Ready result
        block.set_terminator(Terminator {
            kind: TerminatorKind::Return,
        });
    }

    /// Transform a yield point into state save + pending return.
    fn transform_yield_point(
        &self,
        body: &mut MirBody,
        yield_info: &YieldPointInfo,
        state_ptr_local: Local,
        pending_block: BasicBlockId,
    ) {
        // Collect local types first (before borrowing block)
        let local_types: Vec<(Local, usize, Type)> = yield_info
            .live_locals
            .iter()
            .filter_map(|&local| {
                self.local_to_state_field.get(&local).map(|&field_idx| {
                    let local_ty = body.locals[local.0 as usize].ty.clone();
                    (local, field_idx, local_ty)
                })
            })
            .collect();

        // Now borrow the block
        let yield_block = body.block_mut(yield_info.yield_block);

        // Remove the Yield terminator - we'll replace it
        yield_block.terminator = None;

        // Save live locals to state struct
        for (local, field_idx, local_ty) in local_types {
            let state_field = Place::local(state_ptr_local).deref().field(field_idx, local_ty);

            yield_block.push_stmt(Statement {
                kind: StatementKind::Assign(
                    state_field,
                    Rvalue::Use(Operand::Copy(Place::local(local))),
                ),
            });
        }

        // Update __state to resume state
        let state_field = Place::local(state_ptr_local).deref().field(
            CoroutineStateLayout::state_field_index(),
            Type::Primitive(PrimitiveType::U32),
        );

        yield_block.push_stmt(Statement {
            kind: StatementKind::Assign(
                state_field,
                Rvalue::Use(Operand::Constant(Constant::Int(
                    yield_info.state_index as i128,
                    Type::Primitive(PrimitiveType::U32),
                ))),
            ),
        });

        // Goto pending block to return Pending
        yield_block.set_terminator(Terminator {
            kind: TerminatorKind::Goto {
                target: pending_block,
            },
        });

        // Setup resume block to restore locals
        self.setup_resume_block(body, yield_info, state_ptr_local);
    }

    /// Setup the resume block to restore locals from state.
    fn setup_resume_block(
        &self,
        body: &mut MirBody,
        yield_info: &YieldPointInfo,
        state_ptr_local: Local,
    ) {
        // Collect local types and build restore statements first
        let restore_stmts: Vec<Statement> = yield_info
            .live_locals
            .iter()
            .filter_map(|&local| {
                self.local_to_state_field.get(&local).map(|&field_idx| {
                    let local_ty = body.locals[local.0 as usize].ty.clone();
                    let state_field =
                        Place::local(state_ptr_local).deref().field(field_idx, local_ty);

                    Statement {
                        kind: StatementKind::Assign(
                            Place::local(local),
                            Rvalue::Use(Operand::Copy(state_field)),
                        ),
                    }
                })
            })
            .collect();

        // Now borrow the block and prepend restore statements
        let resume_block = body.block_mut(yield_info.resume_block);
        let mut new_statements = restore_stmts;
        new_statements.append(&mut resume_block.statements);
        resume_block.statements = new_statements;
    }

    /// Transform return statements to set result and goto ready.
    fn transform_returns(
        &self,
        body: &mut MirBody,
        state_ptr_local: Local,
        ready_block: BasicBlockId,
    ) {
        // Find all blocks with Return terminator (except the ready block itself)
        let return_blocks: Vec<BasicBlockId> = body
            .basic_blocks
            .iter()
            .filter_map(|(&id, block)| {
                if id == ready_block {
                    return None;
                }
                if let Some(ref term) = block.terminator {
                    if matches!(term.kind, TerminatorKind::Return) {
                        return Some(id);
                    }
                }
                None
            })
            .collect();

        for block_id in return_blocks {
            let block = body.block_mut(block_id);

            // Remove return terminator
            block.terminator = None;

            // Copy return value to __result field in state
            let result_field = Place::local(state_ptr_local)
                .deref()
                .field(
                    CoroutineStateLayout::result_field_index(),
                    self.result_type.clone(),
                );

            block.push_stmt(Statement {
                kind: StatementKind::Assign(
                    result_field,
                    Rvalue::Use(Operand::Copy(Place::return_place())),
                ),
            });

            // Set __state to completed (u32::MAX)
            let state_field = Place::local(state_ptr_local)
                .deref()
                .field(
                    CoroutineStateLayout::state_field_index(),
                    Type::Primitive(PrimitiveType::U32),
                );

            block.push_stmt(Statement {
                kind: StatementKind::Assign(
                    state_field,
                    Rvalue::Use(Operand::Constant(Constant::Int(
                        u32::MAX as i128,
                        Type::Primitive(PrimitiveType::U32),
                    ))),
                ),
            });

            // Goto ready block
            block.set_terminator(Terminator {
                kind: TerminatorKind::Goto {
                    target: ready_block,
                },
            });
        }
    }

    /// Build the final CoroutineInfo.
    fn build_coroutine_info(&self, body: &MirBody) -> CoroutineInfo {
        // Find the dispatch and special blocks
        // They were added at the end, so we can compute their IDs
        let num_original_blocks = body.basic_blocks.len() - 3; // -3 for dispatch, pending, ready
        let dispatch_block = BasicBlockId(num_original_blocks as u32);
        let pending_block = BasicBlockId((num_original_blocks + 1) as u32);
        let ready_block = BasicBlockId((num_original_blocks + 2) as u32);

        CoroutineInfo {
            state_layout: self.state_layout.clone(),
            local_to_state_field: self.local_to_state_field.clone(),
            yield_points: self.yield_points.clone(),
            loop_contexts: self.loop_contexts.clone(),
            nested_coroutines: self.nested_coroutines.clone(),
            poll_entry_block: dispatch_block,
            ready_block,
            pending_block,
        }
    }
}

/// Transform all async functions in a MIR module.
pub fn transform_async_functions(module: &mut crate::mir::MirModule) {
    let body_ids: Vec<BodyId> = module.bodies.keys().copied().collect();

    for body_id in body_ids {
        let body = module.bodies.get(&body_id).unwrap();
        if body.is_async {
            let name = body.name.clone();
            let return_ty = body.return_ty.clone();

            // Extract the inner type from Future<T>
            let result_type = match &return_ty {
                Type::Future(inner) => (**inner).clone(),
                _ => return_ty.clone(),
            };

            let body = module.bodies.shift_remove(&body_id).unwrap();
            let transformer = CoroutineTransformer::new(name, result_type);
            let transformed = transformer.transform(body);
            module.bodies.insert(body_id, transformed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_layout_creation() {
        let layout = CoroutineStateLayout::new(Type::Primitive(PrimitiveType::I64));
        assert_eq!(layout.fields.len(), 2);
        assert_eq!(layout.fields[0].0, "__state");
        assert_eq!(layout.fields[1].0, "__result");
    }

    #[test]
    fn test_state_layout_add_field() {
        let mut layout = CoroutineStateLayout::new(Type::Primitive(PrimitiveType::I64));
        let idx = layout.add_local_field("__local_1".to_string(), Type::Primitive(PrimitiveType::I32));
        assert_eq!(idx, 2);
        assert_eq!(layout.fields.len(), 3);
    }
}
