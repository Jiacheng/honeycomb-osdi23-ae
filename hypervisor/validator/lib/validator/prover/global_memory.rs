use crate::analysis::{
    ConstantPropagation, DomFrontier, LoopAnalysis, LoopBoundAnalysis, PHIAnalysis,
    PolyhedralAnalysis, SCEVExpr, SCEVExprRef, ScalarEvolution, VirtualUse,
};
use crate::error::{Error, Result, ValidationError};
use crate::fileformat::KernelInfo;
use crate::ir::constraints::ValueDescriptor;
use crate::ir::instruction::{IRInstruction, LoadInst, StoreInst};
use crate::ir::{DomTree, Function, RuntimeInfo, Value};
use crate::isa::rdna2::{DispatchPacket, Instruction};
use crate::prover::scalar_range::ScalarRangeAnalysis;
use crate::prover::symbolic_heap::{SymbolicHeap, SymbolicLocation, SymbolicValue};
use crate::support::diagnostic::{DiagnosticContext, Remark};
use std::collections::HashSet;

pub(crate) const SIZE_OF_DWORD: usize = 4;

pub fn prove<'a, 'b>(
    ki: &'a KernelInfo<'a>,
    func: &'a Function<'a>,
    heap: &'b mut SymbolicHeap,
    rt_info: &'b RuntimeInfo,
    diag: &'a DiagnosticContext,
    marks: &'b HashSet<usize>,
) {
    heap.register_implicit_loop_bounds(&rt_info.implicit_loop_info);

    let dom = DomTree::analyze(func);
    let def_use = {
        let df = DomFrontier::new(&dom);
        let def_use = PHIAnalysis::analyze(func, &dom, &df, ki);
        if let Err(err) = def_use {
            diag.record(Remark::kernel(ki.name, 0, err));
            return;
        }
        ConstantPropagation::run(def_use.unwrap())
    };

    let mut li = LoopAnalysis::new(func);
    li.analyze(&dom);
    li.augment_with_implicit_loops();
    let scev = ScalarEvolution::new(&dom, &def_use, &li, Some(rt_info.implicit_loop_info));
    let mut poly = PolyhedralAnalysis::new(func, &scev);
    poly.analyze();
    let lba = LoopBoundAnalysis::new(&def_use, &scev, Some(rt_info.implicit_loop_info));
    let range = ScalarRangeAnalysis::new(&li, &lba, &scev, &poly);
    range.run(heap);
    let mut mem_access =
        GlobalMemoryAccessValidator::new(func, ki, &scev, &range, heap, diag, marks);
    mem_access.run();
}

pub(crate) struct GlobalMemoryAccessValidator<'a, 'b> {
    func: &'a Function<'a>,
    ki: &'a KernelInfo<'a>,
    scev: &'b ScalarEvolution<'a, 'b>,
    range: &'b ScalarRangeAnalysis<'a, 'b>,
    heap: &'b mut SymbolicHeap,
    // // Results of instructions loading kernel arguments -> constraint
    // values: HashMap<SCEVExpr, SymbolicValue>,
    diag: &'b DiagnosticContext,
    marks: &'b HashSet<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
enum SymbolicAddress {
    // Point to an item on the heap
    Location(SymbolicLocation),
    // Point to a range of the address for bounds checks
    AddressRange(usize, usize),
}

impl SymbolicLocation {
    /**
     * Access the dispatch packet or the kernel arguments with constant offsets
     **/
    pub(crate) fn trivial_ro_region_access(addr: &SCEVExprRef) -> Option<SymbolicLocation> {
        fn maybe_offset(
            base: &SCEVExprRef,
            addr: &SCEVExprRef,
            construct: &fn(usize) -> SymbolicLocation,
        ) -> Option<SymbolicLocation> {
            match addr {
                SCEVExprRef::Expr(SCEVExpr::Add(v)) if v.len() == 2 => {
                    let off = v[0].as_int()?;
                    (off >= 0 && v[1] == *base).then_some(construct(off as usize))
                }
                _ if addr == base => Some(construct(0)),
                _ => None,
            }
        }
        type TrivialAccessDescriptor = (SCEVExprRef, fn(usize) -> SymbolicLocation);
        const TRIVIAL_ACCESS: &[TrivialAccessDescriptor] = &[
            (
                SCEVExprRef::KernelArgumentBase,
                SymbolicLocation::KernelArgBuffer,
            ),
            (
                SCEVExprRef::DispatchPacketBase,
                SymbolicLocation::DispatchPacket,
            ),
        ];
        TRIVIAL_ACCESS
            .iter()
            .find_map(|(base, f)| maybe_offset(base, addr, f))
    }
}

impl<'a, 'b> GlobalMemoryAccessValidator<'a, 'b> {
    const RW_REGION_TOP: usize = (1usize << 48) - (2 << 30);
    const RO_REGION_TOP: usize = (1usize << 48) - (1 << 30);

    pub fn new(
        func: &'a Function<'a>,
        ki: &'a KernelInfo<'a>,
        scev: &'b ScalarEvolution<'a, 'b>,
        range: &'b ScalarRangeAnalysis<'a, 'b>,
        heap: &'b mut SymbolicHeap,
        diag: &'b DiagnosticContext,
        marks: &'b HashSet<usize>,
    ) -> Self {
        Self {
            func,
            ki,
            scev,
            range,
            heap,
            diag,
            marks,
        }
    }

    pub fn run(&'b mut self) {
        self.register_load_readonly_values();
        let diag = &self.diag;
        for (inst_idx, inst) in
            self.func
                .instructions
                .iter()
                .enumerate()
                .filter_map(|(idx, inst)| {
                    (!self.marks.contains(&idx))
                        .then(|| Instruction::wrap(inst.clone()).map(move |x| (idx, x)))?
                })
        {
            match inst {
                IRInstruction::StoreInst(si) => {
                    if let Err(e) = self.check_store_inst(inst_idx, &si) {
                        diag.record(Remark::kernel(self.func.name, inst_idx, e));
                    }
                }
                IRInstruction::LoadInst(li) => {
                    if let Err(e) = self.check_load_inst(inst_idx, &li) {
                        diag.record(Remark::kernel(self.func.name, inst_idx, e));
                    }
                }
                _ => {}
            }
        }
    }

    fn register_load_readonly_values(&mut self) {
        for (inst_idx, li) in
            self.func
                .instructions
                .iter()
                .enumerate()
                .filter_map(|(inst_idx, inst)| match Instruction::wrap(inst.clone())? {
                    IRInstruction::LoadInst(li) => Some((inst_idx, li)),
                    _ => None,
                })
        {
            let addr_expr = self.scev.get_scev(VirtualUse::Address(inst_idx));
            if let Some(SymbolicAddress::Location(loc)) = self.resolve_address(&addr_expr) {
                if let SymbolicLocation::KernelArgBuffer(off) = loc {
                    if let Some(sv) = self.heap.get(&loc, li.get_dst_dwords() * SIZE_OF_DWORD) {
                        sv.into_iter().for_each(|(_desc, off_bytes, sv)| {
                            assert_eq!(0, off_bytes % 4);
                            let sv_dword = sv.len / 4;
                            let uses = (0..sv_dword)
                                .map(|x| Value::Instruction(inst_idx, off + x))
                                .collect::<Vec<Value>>();
                            let res = self.scev.get_scev(VirtualUse::Group(uses));
                            self.heap.register_ro_load(res, sv);
                        })
                    }
                }
            }
        }
    }

    fn check_load_inst(&self, inst_idx: usize, li: &LoadInst) -> Result<()> {
        let addr_expr = self.scev.get_scev(VirtualUse::Address(inst_idx));
        let addr = self
            .resolve_address(&addr_expr)
            .ok_or(Error::ValidationError(ValidationError::UnresolvedAddress))?;
        match addr {
            SymbolicAddress::Location(loc) => match loc {
                SymbolicLocation::KernelArgBuffer(off) => {
                    return if off + li.get_dst_dwords() * 4 <= self.ki.kern_arg_size as usize {
                        Ok(())
                    } else {
                        Err(Error::ValidationError(ValidationError::OutOfBounds(
                            off,
                            self.ki.kern_arg_size as usize,
                        )))
                    }
                }
                SymbolicLocation::DispatchPacket(off) => {
                    return if off + li.get_dst_dwords() * SIZE_OF_DWORD
                        <= DispatchPacket::PACKET_SIZE
                    {
                        Ok(())
                    } else {
                        Err(Error::ValidationError(ValidationError::OutOfBounds(
                            off,
                            DispatchPacket::PACKET_SIZE,
                        )))
                    }
                }
                SymbolicLocation::Heap(loc) => {
                    let l = SymbolicLocation::Heap(loc);
                    if let Some(sv) = self.heap.get(&l, li.get_dst_dwords() * SIZE_OF_DWORD) {
                        sv.iter()
                            .try_for_each(|(_desc, off, v)| self.check_load_inbound(*off, v))?;
                    } else {
                        return Err(Error::ValidationError(ValidationError::InvalidAccess));
                    }
                }
            },
            SymbolicAddress::AddressRange(min, max) => {
                self.check_load_inbound_range(min, max, li.get_dst_dwords() * SIZE_OF_DWORD)?;
            }
        }

        Ok(())
    }

    fn check_store_inst(&self, inst_idx: usize, si: &StoreInst) -> Result<()> {
        let addr_expr = self.scev.get_scev(VirtualUse::Address(inst_idx));
        let addr = self
            .resolve_address(&addr_expr)
            .ok_or(Error::ValidationError(ValidationError::UnresolvedAddress))?;
        match addr {
            SymbolicAddress::Location(loc) => match loc {
                SymbolicLocation::KernelArgBuffer(_) | SymbolicLocation::DispatchPacket(_) => {
                    return Err(Error::ValidationError(ValidationError::InvalidAccess));
                }
                SymbolicLocation::Heap(loc) => {
                    // Only allow store on RW space
                    let l = SymbolicLocation::Heap(loc);
                    if let Some(sv) = self.heap.get(&l, si.get_dst_dwords() * SIZE_OF_DWORD) {
                        sv.iter()
                            .try_for_each(|(_desc, off, v)| self.check_store_inbound(*off, v))?;
                    } else {
                        return Err(Error::ValidationError(ValidationError::InvalidAccess));
                    }
                }
            },
            SymbolicAddress::AddressRange(min, max) => {
                self.check_store_inbound_range(min, max, si.get_dst_dwords() * SIZE_OF_DWORD)?;
            }
        }

        Ok(())
    }

    // Resolve addresss for add / addrec. Might need to do it recursively
    fn resolve_address(&self, addr_expr: &SCEVExprRef) -> Option<SymbolicAddress> {
        if let Some(t) = SymbolicLocation::trivial_ro_region_access(addr_expr) {
            return Some(SymbolicAddress::Location(t));
        }
        let range = self.range.get(addr_expr)?;
        assert!(range.min <= range.max);
        let (start, end) = (range.min as usize, range.max as usize);
        Some(SymbolicAddress::AddressRange(start, end))
    }

    fn check_load_inbound(&self, _off: usize, _v: &SymbolicValue) -> Result<()> {
        Ok(())
    }

    fn check_load_inbound_range(&self, _min: usize, max: usize, size: usize) -> Result<()> {
        if max + size >= Self::RO_REGION_TOP {
            Err(Error::ValidationError(ValidationError::OutOfBounds(
                max,
                Self::RO_REGION_TOP,
            )))
        } else {
            Ok(())
        }
    }

    fn check_store_inbound(&self, _off: usize, v: &SymbolicValue) -> Result<()> {
        match v.value {
            ValueDescriptor::SystemPointer(_) => Ok(()),
            _ => Err(Error::ValidationError(ValidationError::InvalidAccess)),
        }
    }

    fn check_store_inbound_range(&self, _min: usize, max: usize, size: usize) -> Result<()> {
        if max + size >= Self::RW_REGION_TOP {
            Err(Error::ValidationError(ValidationError::OutOfBounds(
                max,
                Self::RW_REGION_TOP,
            )))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::ir::constraints::Constraint;
    use crate::ir::constraints::ValueDescriptor::HeapPointer;
    use crate::ir::RuntimeInfo;
    use crate::prover::prove;
    use crate::prover::symbolic_heap::{SymbolicHeap, SymbolicValue};
    use crate::support::diagnostic::DiagnosticContext;

    #[test]
    fn test_validate_global_memory_access() {
        let (ki, func) = crate::tests::cfg::nested_loop();
        let diag = DiagnosticContext::default();
        let mut heap = SymbolicHeap::new();
        heap.register_argument_constraint(
            0,
            SymbolicValue {
                len: 8,
                value: HeapPointer(Constraint {
                    min: 0x2000,
                    max: 0x2000,
                }),
            },
        );
        let rt_info = RuntimeInfo::new([1, 1, 1], [1, 1, 1]);
        prove(
            ki,
            &func,
            &mut heap,
            &rt_info,
            &diag,
            &HashSet::<usize>::new(),
        );
        assert_eq!(0, diag.remarks().len());
    }

    #[test]
    fn test_load_arguments() {
        let t = crate::tests::memory_analysis::load_arguments();
        let diag = DiagnosticContext::default();
        let mut heap = SymbolicHeap::new();
        t.register_constraints(&mut heap);
        let rt_info = RuntimeInfo::new([1, 1, 1], [1, 1, 1]);
        prove(
            &t.kernel,
            &t.func,
            &mut heap,
            &rt_info,
            &diag,
            &HashSet::<usize>::new(),
        );
        assert_eq!(0, diag.remarks().len());
    }
}
