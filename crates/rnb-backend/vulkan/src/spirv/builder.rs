type Word = u32;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Id(pub Word);

pub(crate) mod op {
    pub const CAPABILITY: u32 = 17;
    pub const MEMORY_MODEL: u32 = 14;
    pub const ENTRY_POINT: u32 = 15;
    pub const EXECUTION_MODE: u32 = 16;
    pub const TYPE_VOID: u32 = 19;
    pub const TYPE_BOOL: u32 = 20;
    pub const TYPE_INT: u32 = 21;
    pub const TYPE_FLOAT: u32 = 22;
    pub const TYPE_VECTOR: u32 = 23;
    pub const TYPE_ARRAY: u32 = 28;
    pub const TYPE_RUNTIME_ARRAY: u32 = 29;
    pub const TYPE_STRUCT: u32 = 30;
    pub const TYPE_POINTER: u32 = 32;
    pub const TYPE_FUNCTION: u32 = 33;
    pub const CONSTANT: u32 = 43;
    pub const VARIABLE: u32 = 59;
    pub const LOAD: u32 = 61;
    pub const STORE: u32 = 62;
    pub const ACCESS_CHAIN: u32 = 65;
    pub const FUNCTION: u32 = 54;
    pub const FUNCTION_END: u32 = 56;
    pub const LABEL: u32 = 248;
    pub const RETURN: u32 = 253;
    pub const BRANCH: u32 = 249;
    pub const BRANCH_CONDITIONAL: u32 = 250;
    pub const LOOP_MERGE: u32 = 246;
    pub const SELECTION_MERGE: u32 = 247;
    pub const PHI: u32 = 245;
    pub const IADD: u32 = 128;
    pub const ISUB: u32 = 130;
    pub const IMUL: u32 = 132;
    pub const FADD: u32 = 129;
    pub const FSUB: u32 = 131;
    pub const FMUL: u32 = 133;
    pub const FDIV: u32 = 136;
    pub const UMOD: u32 = 137;
    pub const UDIV: u32 = 134;
    pub const ULESS_THAN: u32 = 176;
    pub const SLESS_THAN: u32 = 177;
    pub const F_ORD_GREATER_THAN: u32 = 186;
    pub const BITCAST: u32 = 124;
    pub const CONVERT_S_TO_F: u32 = 111;
    pub const CONVERT_U_TO_F: u32 = 112;
    // mv38 B: SPIR-V spec — 109=ConvertFToU, 110=ConvertFToS (이전 swap 됨).
    pub const CONVERT_F_TO_U: u32 = 109;
    pub const CONVERT_F_TO_S: u32 = 110;
    pub const F_ORD_LESS_THAN: u32 = 184;
    pub const SHIFT_RIGHT_ARITHMETIC: u32 = 195;
    pub const SHIFT_RIGHT_LOGICAL: u32 = 194;
    pub const SHIFT_LEFT_LOGICAL: u32 = 196;
    pub const BITWISE_OR: u32 = 197;
    pub const BITWISE_AND: u32 = 199;
    pub const CONTROL_BARRIER: u32 = 224;
    pub const COMPOSITE_EXTRACT: u32 = 81;
    pub const DECORATE: u32 = 71;
    pub const MEMBER_DECORATE: u32 = 72;
    pub const FNEGATE: u32 = 127;
    pub const I_NOT_EQUAL: u32 = 171;
    pub const SELECT: u32 = 169;
    pub const EXT_INST_IMPORT: u32 = 11;
    pub const EXT_INST: u32 = 12;
}

pub(crate) mod decoration {
    pub const BUILTIN: u32 = 11;
    pub const BINDING: u32 = 33;
    pub const DESCRIPTOR_SET: u32 = 34;
    pub const OFFSET: u32 = 35;
    pub const ARRAY_STRIDE: u32 = 6;
    pub const BLOCK: u32 = 2;
    /// SPIR-V NoContraction = 42. Marks fp result so driver MUST NOT contract
    /// `mul + add` into fused multiply-add (single-rounding FMA).
    pub const NO_CONTRACTION: u32 = 42;
}

pub(crate) mod builtin {
    pub const WORKGROUP_ID: u32 = 26;
    pub const GLOBAL_INVOCATION_ID: u32 = 28;
    pub const LOCAL_INVOCATION_ID: u32 = 27;
}

pub(crate) mod storage_class {
    pub const INPUT: u32 = 1;
    pub const WORKGROUP: u32 = 4;
    pub const FUNCTION: u32 = 7;
    pub const PUSH_CONSTANT: u32 = 9;
    pub const STORAGE_BUFFER: u32 = 12;
}

pub(crate) mod scope {
    pub const WORKGROUP: u32 = 2;
}

pub(crate) mod memory_semantics {
    pub const WORKGROUP_MEMORY: u32 = 0x100;
    pub const ACQUIRE_RELEASE: u32 = 0x8;
}

pub struct SpirvModule {
    pub(crate) bound: Word,
    pub(crate) capabilities: Vec<Vec<Word>>,
    pub(crate) extensions: Vec<Vec<Word>>,
    pub(crate) ext_inst_imports: Vec<Vec<Word>>,
    pub(crate) memory_model: Vec<Word>,
    pub(crate) entry_points: Vec<Vec<Word>>,
    pub(crate) execution_modes: Vec<Vec<Word>>,
    pub(crate) decorations: Vec<Vec<Word>>,
    pub(crate) type_declarations: Vec<Vec<Word>>,
    pub(crate) global_variables: Vec<Vec<Word>>,
    pub(crate) functions: Vec<Vec<Word>>,
}

impl SpirvModule {
    pub fn new() -> Self {
        Self {
            bound: 1,
            capabilities: Vec::new(),
            extensions: Vec::new(),
            ext_inst_imports: Vec::new(),
            memory_model: Vec::new(),
            entry_points: Vec::new(),
            execution_modes: Vec::new(),
            decorations: Vec::new(),
            type_declarations: Vec::new(),
            global_variables: Vec::new(),
            functions: Vec::new(),
        }
    }

    pub fn alloc_id(&mut self) -> Id {
        let id = Id(self.bound);
        self.bound += 1;
        id
    }

    pub(crate) fn encode_inst(opcode: u32, words: &[Word]) -> Vec<Word> {
        let word_count = (words.len() + 1) as u32;
        let mut inst = vec![(word_count << 16) | opcode];
        inst.extend_from_slice(words);
        inst
    }

    fn encode_string(s: &str) -> Vec<Word> {
        let bytes = s.as_bytes();
        let len = bytes.len() + 1; // +1 for null terminator
        let num_words = (len + 3) / 4;
        let mut words = vec![0u32; num_words];
        for (i, &b) in bytes.iter().enumerate() {
            words[i / 4] |= (b as u32) << ((i % 4) * 8);
        }
        words
    }

    // --- Capability & memory model ---

    pub fn capability(&mut self, cap: u32) {
        let inst = Self::encode_inst(op::CAPABILITY, &[cap]);
        self.capabilities.push(inst);
    }

    pub fn extension(&mut self, name: &str) {
        let mut words = Self::encode_string(name);
        let mut inst = vec![(((words.len() + 1) as u32) << 16) | 10]; // OpExtension = 10
        inst.append(&mut words);
        self.extensions.push(inst);
    }

    pub fn memory_model(&mut self, addressing: u32, memory: u32) {
        let inst = Self::encode_inst(op::MEMORY_MODEL, &[addressing, memory]);
        self.memory_model = inst;
    }

    pub fn ext_inst_import(&mut self, name: &str) -> Id {
        let id = self.alloc_id();
        let mut words = vec![id.0];
        words.extend(Self::encode_string(name));
        let inst = Self::encode_inst(op::EXT_INST_IMPORT, &words);
        self.ext_inst_imports.push(inst);
        id
    }

    // --- Entry points & execution modes ---

    pub fn entry_point(&mut self, exec_model: u32, func_id: Id, name: &str, interface: &[Id]) {
        let mut words = vec![exec_model, func_id.0];
        words.extend(Self::encode_string(name));
        for iface in interface {
            words.push(iface.0);
        }
        let inst = Self::encode_inst(op::ENTRY_POINT, &words);
        self.entry_points.push(inst);
    }

    pub fn execution_mode_local_size(&mut self, entry: Id, x: u32, y: u32, z: u32) {
        // ExecutionMode LocalSize = 17
        let inst = Self::encode_inst(op::EXECUTION_MODE, &[entry.0, 17, x, y, z]);
        self.execution_modes.push(inst);
    }

    // --- Decorations ---

    pub fn decorate(&mut self, target: Id, decoration: u32, operands: &[u32]) {
        let mut words = vec![target.0, decoration];
        words.extend_from_slice(operands);
        let inst = Self::encode_inst(op::DECORATE, &words);
        self.decorations.push(inst);
    }

    pub fn member_decorate(
        &mut self,
        struct_type: Id,
        member: u32,
        decoration: u32,
        operands: &[u32],
    ) {
        let mut words = vec![struct_type.0, member, decoration];
        words.extend_from_slice(operands);
        let inst = Self::encode_inst(op::MEMBER_DECORATE, &words);
        self.decorations.push(inst);
    }

    // --- Type declarations ---

    pub fn type_void(&mut self) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::TYPE_VOID, &[id.0]);
        self.type_declarations.push(inst);
        id
    }

    pub fn type_bool(&mut self) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::TYPE_BOOL, &[id.0]);
        self.type_declarations.push(inst);
        id
    }

    pub fn type_int(&mut self, width: u32, signedness: u32) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::TYPE_INT, &[id.0, width, signedness]);
        self.type_declarations.push(inst);
        id
    }

    pub fn type_float(&mut self, width: u32) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::TYPE_FLOAT, &[id.0, width]);
        self.type_declarations.push(inst);
        id
    }

    pub fn type_vector(&mut self, component: Id, count: u32) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::TYPE_VECTOR, &[id.0, component.0, count]);
        self.type_declarations.push(inst);
        id
    }

    pub fn type_array(&mut self, element: Id, length: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::TYPE_ARRAY, &[id.0, element.0, length.0]);
        self.type_declarations.push(inst);
        id
    }

    pub fn type_runtime_array(&mut self, element: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::TYPE_RUNTIME_ARRAY, &[id.0, element.0]);
        self.type_declarations.push(inst);
        id
    }

    pub fn type_struct(&mut self, members: &[Id]) -> Id {
        let id = self.alloc_id();
        let mut words = vec![id.0];
        for m in members {
            words.push(m.0);
        }
        let inst = Self::encode_inst(op::TYPE_STRUCT, &words);
        self.type_declarations.push(inst);
        id
    }

    pub fn type_pointer(&mut self, storage_class: u32, pointee: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::TYPE_POINTER, &[id.0, storage_class, pointee.0]);
        self.type_declarations.push(inst);
        id
    }

    pub fn type_function(&mut self, return_type: Id, params: &[Id]) -> Id {
        let id = self.alloc_id();
        let mut words = vec![id.0, return_type.0];
        for p in params {
            words.push(p.0);
        }
        let inst = Self::encode_inst(op::TYPE_FUNCTION, &words);
        self.type_declarations.push(inst);
        id
    }

    // --- Constants ---

    pub fn constant_u32(&mut self, ty: Id, value: u32) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::CONSTANT, &[ty.0, id.0, value]);
        self.type_declarations.push(inst);
        id
    }

    pub fn constant_f32(&mut self, ty: Id, value: f32) -> Id {
        let id = self.alloc_id();
        let bits = value.to_bits();
        let inst = Self::encode_inst(op::CONSTANT, &[ty.0, id.0, bits]);
        self.type_declarations.push(inst);
        id
    }

    // --- Global variables ---

    pub fn variable(&mut self, ptr_type: Id, storage_class: u32) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::VARIABLE, &[ptr_type.0, id.0, storage_class]);
        self.global_variables.push(inst);
        id
    }

    // --- Function body ---

    pub fn function(&mut self, result_type: Id, func_id: Id, control: u32, func_type: Id) {
        let inst = Self::encode_inst(
            op::FUNCTION,
            &[result_type.0, func_id.0, control, func_type.0],
        );
        self.functions.push(inst);
    }

    pub fn function_end(&mut self) {
        let inst = Self::encode_inst(op::FUNCTION_END, &[]);
        self.functions.push(inst);
    }

    pub fn label(&mut self) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::LABEL, &[id.0]);
        self.functions.push(inst);
        id
    }

    pub fn ret(&mut self) {
        let inst = Self::encode_inst(op::RETURN, &[]);
        self.functions.push(inst);
    }

    // --- Memory ops ---

    pub fn load(&mut self, result_type: Id, pointer: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::LOAD, &[result_type.0, id.0, pointer.0]);
        self.functions.push(inst);
        id
    }

    pub fn store(&mut self, pointer: Id, value: Id) {
        let inst = Self::encode_inst(op::STORE, &[pointer.0, value.0]);
        self.functions.push(inst);
    }

    pub fn access_chain(&mut self, result_type: Id, base: Id, indices: &[Id]) -> Id {
        let id = self.alloc_id();
        let mut words = vec![result_type.0, id.0, base.0];
        for idx in indices {
            words.push(idx.0);
        }
        let inst = Self::encode_inst(op::ACCESS_CHAIN, &words);
        self.functions.push(inst);
        id
    }

    pub fn composite_extract(&mut self, result_type: Id, composite: Id, index: u32) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(
            op::COMPOSITE_EXTRACT,
            &[result_type.0, id.0, composite.0, index],
        );
        self.functions.push(inst);
        id
    }

    // --- Arithmetic ---

    pub fn iadd(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::IADD, &[ty.0, id.0, a.0, b.0]);
        self.functions.push(inst);
        id
    }

    pub fn isub(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::ISUB, &[ty.0, id.0, a.0, b.0]);
        self.functions.push(inst);
        id
    }

    pub fn imul(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::IMUL, &[ty.0, id.0, a.0, b.0]);
        self.functions.push(inst);
        id
    }

    pub fn fadd(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::FADD, &[ty.0, id.0, a.0, b.0]);
        self.functions.push(inst);
        id
    }

    pub fn fsub(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::FSUB, &[ty.0, id.0, a.0, b.0]);
        self.functions.push(inst);
        id
    }

    pub fn fmul(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::FMUL, &[ty.0, id.0, a.0, b.0]);
        self.functions.push(inst);
        id
    }

    pub fn fdiv(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::FDIV, &[ty.0, id.0, a.0, b.0]);
        self.functions.push(inst);
        id
    }

    /// mv38 C: NoContraction-decorated fp ops. Driver 가 자동 fma fusion 하지
    /// 않게 강제 — separate fmul + fadd 로 ARM CPU NEON 의 separate rounding
    /// 과 비트 일치 가까움.
    pub fn fadd_nc(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.fadd(ty, a, b);
        self.decorate(id, decoration::NO_CONTRACTION, &[]);
        id
    }

    pub fn fsub_nc(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.fsub(ty, a, b);
        self.decorate(id, decoration::NO_CONTRACTION, &[]);
        id
    }

    pub fn fmul_nc(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.fmul(ty, a, b);
        self.decorate(id, decoration::NO_CONTRACTION, &[]);
        id
    }

    pub fn fdiv_nc(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.fdiv(ty, a, b);
        self.decorate(id, decoration::NO_CONTRACTION, &[]);
        id
    }

    pub fn udiv(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::UDIV, &[ty.0, id.0, a.0, b.0]);
        self.functions.push(inst);
        id
    }

    pub fn umod(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::UMOD, &[ty.0, id.0, a.0, b.0]);
        self.functions.push(inst);
        id
    }

    // --- Bitwise ops ---

    pub fn shift_right_arithmetic(&mut self, ty: Id, base: Id, shift: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::SHIFT_RIGHT_ARITHMETIC, &[ty.0, id.0, base.0, shift.0]);
        self.functions.push(inst);
        id
    }

    pub fn shift_right_logical(&mut self, ty: Id, base: Id, shift: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::SHIFT_RIGHT_LOGICAL, &[ty.0, id.0, base.0, shift.0]);
        self.functions.push(inst);
        id
    }

    pub fn shift_left_logical(&mut self, ty: Id, base: Id, shift: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::SHIFT_LEFT_LOGICAL, &[ty.0, id.0, base.0, shift.0]);
        self.functions.push(inst);
        id
    }

    pub fn bitwise_and(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::BITWISE_AND, &[ty.0, id.0, a.0, b.0]);
        self.functions.push(inst);
        id
    }

    pub fn bitwise_or(&mut self, ty: Id, a: Id, b: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::BITWISE_OR, &[ty.0, id.0, a.0, b.0]);
        self.functions.push(inst);
        id
    }

    // --- Conversions ---

    pub fn bitcast(&mut self, result_type: Id, operand: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::BITCAST, &[result_type.0, id.0, operand.0]);
        self.functions.push(inst);
        id
    }

    pub fn convert_s_to_f(&mut self, result_type: Id, operand: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::CONVERT_S_TO_F, &[result_type.0, id.0, operand.0]);
        self.functions.push(inst);
        id
    }

    pub fn convert_u_to_f(&mut self, result_type: Id, operand: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::CONVERT_U_TO_F, &[result_type.0, id.0, operand.0]);
        self.functions.push(inst);
        id
    }

    // --- Control flow ---

    pub fn branch(&mut self, target: Id) {
        let inst = Self::encode_inst(op::BRANCH, &[target.0]);
        self.functions.push(inst);
    }

    pub fn branch_conditional(&mut self, cond: Id, true_label: Id, false_label: Id) {
        let inst = Self::encode_inst(
            op::BRANCH_CONDITIONAL,
            &[cond.0, true_label.0, false_label.0],
        );
        self.functions.push(inst);
    }

    pub fn loop_merge(&mut self, merge: Id, continue_target: Id, control: u32) {
        let inst = Self::encode_inst(op::LOOP_MERGE, &[merge.0, continue_target.0, control]);
        self.functions.push(inst);
    }

    pub fn selection_merge(&mut self, merge: Id, control: u32) {
        let inst = Self::encode_inst(op::SELECTION_MERGE, &[merge.0, control]);
        self.functions.push(inst);
    }

    pub fn phi(&mut self, result_type: Id, incoming: &[(Id, Id)]) -> Id {
        let id = self.alloc_id();
        let mut words = vec![result_type.0, id.0];
        for (value, label) in incoming {
            words.push(value.0);
            words.push(label.0);
        }
        let inst = Self::encode_inst(op::PHI, &words);
        self.functions.push(inst);
        id
    }

    // --- Comparisons ---

    pub fn u_less_than(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::ULESS_THAN, &[bool_ty.0, id.0, a.0, b.0]);
        self.functions.push(inst);
        id
    }

    pub fn s_less_than(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::SLESS_THAN, &[bool_ty.0, id.0, a.0, b.0]);
        self.functions.push(inst);
        id
    }

    pub fn f_ord_greater_than(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::F_ORD_GREATER_THAN, &[bool_ty.0, id.0, a.0, b.0]);
        self.functions.push(inst);
        id
    }

    pub fn f_ord_less_than(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::F_ORD_LESS_THAN, &[bool_ty.0, id.0, a.0, b.0]);
        self.functions.push(inst);
        id
    }

    pub fn convert_f_to_s(&mut self, result_type: Id, operand: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::CONVERT_F_TO_S, &[result_type.0, id.0, operand.0]);
        self.functions.push(inst);
        id
    }

    pub fn convert_f_to_u(&mut self, result_type: Id, operand: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::CONVERT_F_TO_U, &[result_type.0, id.0, operand.0]);
        self.functions.push(inst);
        id
    }

    pub fn fnegate(&mut self, result_type: Id, operand: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::FNEGATE, &[result_type.0, id.0, operand.0]);
        self.functions.push(inst);
        id
    }

    /// OpExtInst: call extended instruction (e.g. GLSL.std.450 Exp, Sqrt)
    pub fn ext_inst(
        &mut self,
        result_type: Id,
        ext_set: Id,
        instruction: u32,
        operands: &[Id],
    ) -> Id {
        let id = self.alloc_id();
        let mut words = vec![result_type.0, id.0, ext_set.0, instruction];
        for op in operands {
            words.push(op.0);
        }
        self.functions.push(Self::encode_inst(op::EXT_INST, &words));
        id
    }

    pub fn i_not_equal(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::I_NOT_EQUAL, &[bool_ty.0, id.0, a.0, b.0]);
        self.functions.push(inst);
        id
    }

    /// OpSelect: result = cond ? true_val : false_val
    pub fn select(&mut self, result_type: Id, cond: Id, true_val: Id, false_val: Id) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(
            op::SELECT,
            &[result_type.0, id.0, cond.0, true_val.0, false_val.0],
        );
        self.functions.push(inst);
        id
    }

    // --- Barrier ---

    pub fn control_barrier(&mut self, execution: Id, memory: Id, semantics: Id) {
        let inst = Self::encode_inst(op::CONTROL_BARRIER, &[execution.0, memory.0, semantics.0]);
        self.functions.push(inst);
    }

    /// Function-scope OpVariable (must be emitted at the top of the first block)
    pub fn function_variable(&mut self, ptr_type: Id, storage_class: u32) -> Id {
        let id = self.alloc_id();
        let inst = Self::encode_inst(op::VARIABLE, &[ptr_type.0, id.0, storage_class]);
        self.functions.push(inst);
        id
    }

    // --- Binary encode ---

    pub fn encode(&self) -> Vec<Word> {
        let mut out = Vec::new();

        // SPIR-V header
        out.push(0x07230203); // magic
        out.push(0x00010300); // version 1.3 (Vulkan 1.1+)
        out.push(0); // generator magic number
        out.push(self.bound); // bound
        out.push(0); // reserved schema

        // Logical layout sections
        for inst in &self.capabilities {
            out.extend_from_slice(inst);
        }
        for inst in &self.extensions {
            out.extend_from_slice(inst);
        }
        for inst in &self.ext_inst_imports {
            out.extend_from_slice(inst);
        }
        if !self.memory_model.is_empty() {
            out.extend_from_slice(&self.memory_model);
        }
        for inst in &self.entry_points {
            out.extend_from_slice(inst);
        }
        for inst in &self.execution_modes {
            out.extend_from_slice(inst);
        }
        for inst in &self.decorations {
            out.extend_from_slice(inst);
        }
        for inst in &self.type_declarations {
            out.extend_from_slice(inst);
        }
        for inst in &self.global_variables {
            out.extend_from_slice(inst);
        }
        for inst in &self.functions {
            out.extend_from_slice(inst);
        }

        out
    }
}

/// Emit a row-major F32 GEMV compute shader as SPIR-V bytecode.
///
/// Storage buffers (set=0):
///   binding=0: weight (f32 array, row-major `[rows, cols]`)
///   binding=1: input  (f32 array)
///   binding=2: output (f32 array)
///
/// Push constants: { rows: u32, cols: u32, rows_per_wg: u32 }
pub fn emit_f32_gemv(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_arr_f32 = m.type_runtime_array(t_f32);

    let t_struct_weight = m.type_struct(&[t_arr_f32]);
    let t_struct_input = m.type_struct(&[t_arr_f32]);
    let t_struct_output = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);

    let t_ptr_sb_weight = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_weight);
    let t_ptr_sb_input = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_input);
    let t_ptr_sb_output = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_output);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);

    m.decorate(t_struct_weight, decoration::BLOCK, &[]);
    m.decorate(t_struct_input, decoration::BLOCK, &[]);
    m.decorate(t_struct_output, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_weight, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_input, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_output, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);

    let gvar_weight = m.variable(t_ptr_sb_weight, storage_class::STORAGE_BUFFER);
    let gvar_input = m.variable(t_ptr_sb_input, storage_class::STORAGE_BUFFER);
    let gvar_output = m.variable(t_ptr_sb_output, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let gvar_gid = m.variable(t_ptr_input_v3, storage_class::INPUT);

    m.decorate(gvar_weight, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_weight, decoration::BINDING, &[0]);
    m.decorate(gvar_input, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_input, decoration::BINDING, &[1]);
    m.decorate(gvar_output, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_output, decoration::BINDING, &[2]);
    m.decorate(
        gvar_gid,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_gid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    let lbl_entry = m.alloc_id();
    let lbl_bounds_body = m.alloc_id();
    let lbl_bounds_merge = m.alloc_id();
    let lbl_loop_header = m.alloc_id();
    let lbl_loop_cond = m.alloc_id();
    let lbl_loop_body = m.alloc_id();
    let lbl_loop_continue = m.alloc_id();
    let lbl_loop_merge = m.alloc_id();

    m.function(t_void, func_id, 0, t_fn_void);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let var_col = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);

    let gid_vec = m.load(t_v3u32, gvar_gid);
    let row = m.composite_extract(t_u32, gid_vec, 0);

    let pc_rows_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let rows = m.load(t_u32, pc_rows_ptr);
    let pc_cols_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let cols = m.load(t_u32, pc_cols_ptr);

    let in_bounds = m.u_less_than(t_bool, row, rows);
    m.selection_merge(lbl_bounds_merge, 0);
    m.branch_conditional(in_bounds, lbl_bounds_body, lbl_bounds_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_body.0]));
    m.store(var_col, c_u32_0);
    m.store(var_sum, c_f32_0);
    m.branch(lbl_loop_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_header.0]));
    m.loop_merge(lbl_loop_merge, lbl_loop_continue, 0);
    m.branch(lbl_loop_cond);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_cond.0]));
    let col_cur = m.load(t_u32, var_col);
    let loop_cond = m.u_less_than(t_bool, col_cur, cols);
    m.branch_conditional(loop_cond, lbl_loop_body, lbl_loop_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_body.0]));
    let row_base = m.imul(t_u32, row, cols);
    let weight_idx = m.iadd(t_u32, row_base, col_cur);
    let weight_ptr = m.access_chain(t_ptr_sb_f32, gvar_weight, &[c_u32_0, weight_idx]);
    let weight_val = m.load(t_f32, weight_ptr);
    let input_ptr = m.access_chain(t_ptr_sb_f32, gvar_input, &[c_u32_0, col_cur]);
    let input_val = m.load(t_f32, input_ptr);
    let product = m.fmul(t_f32, weight_val, input_val);
    let sum_old = m.load(t_f32, var_sum);
    let sum_new = m.fadd(t_f32, sum_old, product);
    m.store(var_sum, sum_new);
    m.branch(lbl_loop_continue);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_continue.0]));
    let col_for_next = m.load(t_u32, var_col);
    let col_next = m.iadd(t_u32, col_for_next, c_u32_1);
    m.store(var_col, col_next);
    m.branch(lbl_loop_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_merge.0]));
    let final_sum = m.load(t_f32, var_sum);
    let out_ptr = m.access_chain(t_ptr_sb_f32, gvar_output, &[c_u32_0, row]);
    m.store(out_ptr, final_sum);
    m.branch(lbl_bounds_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_merge.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// Emit a complete Q8_0 GEMV compute shader as SPIR-V bytecode.
///
/// Weight buffer uses transposed SoA layout for coalesced access:
///   weight[(block * 9 + word) * rows + row]
/// Adjacent threads read adjacent u32 addresses (perfect coalescing).
///
/// Storage buffers (set=0):
///   binding=0: weight (uint array, transposed SoA)
///   binding=1: input  (f32 array)
///   binding=2: output (f32 array)
///
/// Push constants: { rows: u32, cols: u32, rows_per_wg: u32 }
pub fn emit_q8_0_gemv(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    // Capabilities & memory model
    m.capability(1); // Shader
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1); // Logical, GLSL450

    // --- Types ---
    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_i32 = m.type_int(32, 1);
    let t_f32 = m.type_float(32);

    let t_v3u32 = m.type_vector(t_u32, 3);

    let t_arr_u32 = m.type_runtime_array(t_u32);
    let t_arr_f32 = m.type_runtime_array(t_f32);

    let t_struct_weight = m.type_struct(&[t_arr_u32]);
    let t_struct_input = m.type_struct(&[t_arr_f32]);
    let t_struct_output = m.type_struct(&[t_arr_f32]);

    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);

    // Pointer types
    let t_ptr_sb_struct_weight = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_weight);
    let t_ptr_sb_struct_input = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_input);
    let t_ptr_sb_struct_output = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_output);
    let t_ptr_pc_struct = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_u32 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);

    let t_fn_void = m.type_function(t_void, &[]);

    // --- Constants ---
    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_8 = m.constant_u32(t_u32, 8);
    let c_u32_9 = m.constant_u32(t_u32, 9);
    let c_u32_10 = m.constant_u32(t_u32, 10);
    let c_u32_13 = m.constant_u32(t_u32, 13);
    let c_u32_15 = m.constant_u32(t_u32, 15);
    let c_u32_16 = m.constant_u32(t_u32, 16);
    let c_u32_23 = m.constant_u32(t_u32, 23);
    let c_u32_24 = m.constant_u32(t_u32, 24);
    let c_u32_31 = m.constant_u32(t_u32, 31);
    let c_u32_32 = m.constant_u32(t_u32, 32);
    let c_u32_ff = m.constant_u32(t_u32, 0xFF);
    let c_u32_ffff = m.constant_u32(t_u32, 0xFFFF);
    let c_u32_1f = m.constant_u32(t_u32, 0x1F);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3FF);
    let c_u32_112 = m.constant_u32(t_u32, 112);

    let c_i32_24 = m.constant_u32(t_i32, 24);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);
    let c_f32_2pow_neg24 = m.constant_f32(t_f32, 5.9604644775390625e-8);

    // --- Decorations ---
    m.decorate(t_struct_weight, decoration::BLOCK, &[]);
    m.decorate(t_struct_input, decoration::BLOCK, &[]);
    m.decorate(t_struct_output, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_weight, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_input, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_output, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_u32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);

    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);

    // --- Global variables ---
    let gvar_weight = m.variable(t_ptr_sb_struct_weight, storage_class::STORAGE_BUFFER);
    let gvar_input = m.variable(t_ptr_sb_struct_input, storage_class::STORAGE_BUFFER);
    let gvar_output = m.variable(t_ptr_sb_struct_output, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc_struct, storage_class::PUSH_CONSTANT);
    let gvar_glob_id = m.variable(t_ptr_input_u32, storage_class::INPUT);

    m.decorate(gvar_weight, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_weight, decoration::BINDING, &[0]);
    m.decorate(gvar_input, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_input, decoration::BINDING, &[1]);
    m.decorate(gvar_output, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_output, decoration::BINDING, &[2]);
    m.decorate(
        gvar_glob_id,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    // --- Entry point ---
    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_glob_id]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    // Pre-allocate labels
    let lbl_entry = m.alloc_id();
    let lbl_bounds_merge = m.alloc_id();
    let lbl_bounds_true = m.alloc_id();
    let lbl_outer_header = m.alloc_id();
    let lbl_outer_body = m.alloc_id();
    let lbl_outer_continue = m.alloc_id();
    let lbl_outer_merge = m.alloc_id();
    // inner loop labels removed — loop is fully unrolled

    // --- Function ---
    m.function(t_void, func_id, 0, t_fn_void);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    // Function variables
    let var_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_blk = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);

    // Load GlobalInvocationID.x → row
    let glob_id_vec = m.load(t_v3u32, gvar_glob_id);
    let row = m.composite_extract(t_u32, glob_id_vec, 0);

    // Load push constants: rows, cols
    let pc_ptr_rows = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let pc_rows = m.load(t_u32, pc_ptr_rows);
    let pc_ptr_cols = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let pc_cols = m.load(t_u32, pc_ptr_cols);

    // Bounds check: row < rows
    let in_bounds = m.u_less_than(t_bool, row, pc_rows);
    m.selection_merge(lbl_bounds_merge, 0);
    m.branch_conditional(in_bounds, lbl_bounds_true, lbl_bounds_merge);

    // --- In-bounds block ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_true.0]));

    m.store(var_sum, c_f32_0);

    // num_blocks = cols / 32
    let num_blocks = m.udiv(t_u32, pc_cols, c_u32_32);

    m.store(var_blk, c_u32_0);
    m.branch(lbl_outer_header);

    // --- Outer loop header (block loop) ---
    let lbl_outer_cond = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_header.0]));
    m.loop_merge(lbl_outer_merge, lbl_outer_continue, 0);
    m.branch(lbl_outer_cond);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_cond.0]));
    let blk_cur = m.load(t_u32, var_blk);
    let outer_cond = m.u_less_than(t_bool, blk_cur, num_blocks);
    m.branch_conditional(outer_cond, lbl_outer_body, lbl_outer_merge);

    // --- Outer loop body ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_body.0]));

    // === TRANSPOSED SoA weight layout ===
    // weight[(block * 9 + word) * rows + row]
    // plane_base = block * 9 * rows
    let blk_x_9 = m.imul(t_u32, blk_cur, c_u32_9);
    let plane_base = m.imul(t_u32, blk_x_9, pc_rows);

    // Load f16 scale: weight[plane_base + row]
    let scale_addr = m.iadd(t_u32, plane_base, row);
    let scale_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, scale_addr]);
    let scale_word = m.load(t_u32, scale_ptr);
    let f16_bits_raw = m.bitwise_and(t_u32, scale_word, c_u32_ffff);

    // f16 → f32 conversion
    let f16_sign = m.shift_right_logical(t_u32, f16_bits_raw, c_u32_15);
    let f16_sign_bit = m.bitwise_and(t_u32, f16_sign, c_u32_1);
    let f16_exp_raw = m.shift_right_logical(t_u32, f16_bits_raw, c_u32_10);
    let f16_exp = m.bitwise_and(t_u32, f16_exp_raw, c_u32_1f);
    let f16_mant = m.bitwise_and(t_u32, f16_bits_raw, c_u32_3ff);

    let f32_sign_part = m.shift_left_logical(t_u32, f16_sign_bit, c_u32_31);
    let f32_exp_adj = m.iadd(t_u32, f16_exp, c_u32_112);
    let f32_exp_part = m.shift_left_logical(t_u32, f32_exp_adj, c_u32_23);
    let f32_mant_part = m.shift_left_logical(t_u32, f16_mant, c_u32_13);
    let f32_bits_mid = m.bitwise_or(t_u32, f32_sign_part, f32_exp_part);
    let f32_bits = m.bitwise_or(t_u32, f32_bits_mid, f32_mant_part);
    let normal_scale = m.bitcast(t_f32, f32_bits);
    let mant_f = m.convert_u_to_f(t_f32, f16_mant);
    let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
    let denorm_neg = m.fnegate(t_f32, denorm_abs);
    let sign_set = m.i_not_equal(t_bool, f16_sign_bit, c_u32_0);
    let denorm_scale = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
    let exp_nonzero = m.i_not_equal(t_bool, f16_exp, c_u32_0);
    let scale_f32 = m.select(t_f32, exp_nonzero, normal_scale, denorm_scale);

    // qs planes start at plane_base + rows (word 1..8)
    let qs_plane_base = m.iadd(t_u32, plane_base, pc_rows);

    // Unrolled inner loop: 8 words × 4 bytes = 32 qs values per block
    // No loop overhead: no counter, no branch, no merge
    let mut block_sum = c_f32_0;
    let blk_x_32 = m.imul(t_u32, blk_cur, c_u32_32);

    for w in 0..8u32 {
        // Load qs word: weight[qs_plane_base + w * rows + row]
        let w_offset = if w == 0 {
            qs_plane_base
        } else {
            let c_w = m.constant_u32(t_u32, w);
            let w_x_rows = m.imul(t_u32, c_w, pc_rows);
            m.iadd(t_u32, qs_plane_base, w_x_rows)
        };
        let qs_addr = m.iadd(t_u32, w_offset, row);
        let w_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, qs_addr]);
        let w_word = m.load(t_u32, w_ptr);

        // Input base: blk*32 + w*4
        let inp_base = if w == 0 {
            blk_x_32
        } else {
            let c_w4 = m.constant_u32(t_u32, w * 4);
            m.iadd(t_u32, blk_x_32, c_w4)
        };

        // byte 0
        let b0_raw = m.bitwise_and(t_u32, w_word, c_u32_ff);
        let b0_i32 = m.bitcast(t_i32, b0_raw);
        let b0_sl = m.shift_left_logical(t_i32, b0_i32, c_i32_24);
        let b0_se = m.shift_right_arithmetic(t_i32, b0_sl, c_i32_24);
        let b0_f = m.convert_s_to_f(t_f32, b0_se);
        let i0_ptr = m.access_chain(t_ptr_sb_f32, gvar_input, &[c_u32_0, inp_base]);
        let i0 = m.load(t_f32, i0_ptr);
        let p0 = m.fmul(t_f32, b0_f, i0);
        block_sum = m.fadd(t_f32, block_sum, p0);

        // byte 1
        let b1_shr = m.shift_right_logical(t_u32, w_word, c_u32_8);
        let b1_raw = m.bitwise_and(t_u32, b1_shr, c_u32_ff);
        let b1_i32 = m.bitcast(t_i32, b1_raw);
        let b1_sl = m.shift_left_logical(t_i32, b1_i32, c_i32_24);
        let b1_se = m.shift_right_arithmetic(t_i32, b1_sl, c_i32_24);
        let b1_f = m.convert_s_to_f(t_f32, b1_se);
        let i1_idx = m.iadd(t_u32, inp_base, c_u32_1);
        let i1_ptr = m.access_chain(t_ptr_sb_f32, gvar_input, &[c_u32_0, i1_idx]);
        let i1 = m.load(t_f32, i1_ptr);
        let p1 = m.fmul(t_f32, b1_f, i1);
        block_sum = m.fadd(t_f32, block_sum, p1);

        // byte 2
        let b2_shr = m.shift_right_logical(t_u32, w_word, c_u32_16);
        let b2_raw = m.bitwise_and(t_u32, b2_shr, c_u32_ff);
        let b2_i32 = m.bitcast(t_i32, b2_raw);
        let b2_sl = m.shift_left_logical(t_i32, b2_i32, c_i32_24);
        let b2_se = m.shift_right_arithmetic(t_i32, b2_sl, c_i32_24);
        let b2_f = m.convert_s_to_f(t_f32, b2_se);
        let i2_idx = m.iadd(t_u32, inp_base, c_u32_2);
        let i2_ptr = m.access_chain(t_ptr_sb_f32, gvar_input, &[c_u32_0, i2_idx]);
        let i2 = m.load(t_f32, i2_ptr);
        let p2 = m.fmul(t_f32, b2_f, i2);
        block_sum = m.fadd(t_f32, block_sum, p2);

        // byte 3
        let b3_shr = m.shift_right_logical(t_u32, w_word, c_u32_24);
        let b3_raw = m.bitwise_and(t_u32, b3_shr, c_u32_ff);
        let b3_i32 = m.bitcast(t_i32, b3_raw);
        let b3_sl = m.shift_left_logical(t_i32, b3_i32, c_i32_24);
        let b3_se = m.shift_right_arithmetic(t_i32, b3_sl, c_i32_24);
        let b3_f = m.convert_s_to_f(t_f32, b3_se);
        let i3_idx = m.iadd(t_u32, inp_base, c_u32_3);
        let i3_ptr = m.access_chain(t_ptr_sb_f32, gvar_input, &[c_u32_0, i3_idx]);
        let i3 = m.load(t_f32, i3_ptr);
        let p3 = m.fmul(t_f32, b3_f, i3);
        block_sum = m.fadd(t_f32, block_sum, p3);
    }

    let block_raw = block_sum;
    let block_scaled = m.fmul(t_f32, block_raw, scale_f32);
    let prev_total = m.load(t_f32, var_sum);
    let new_total = m.fadd(t_f32, prev_total, block_scaled);
    m.store(var_sum, new_total);
    m.branch(lbl_outer_continue);

    // --- Outer loop continue ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_continue.0]));
    let blk_next = m.iadd(t_u32, blk_cur, c_u32_1);
    m.store(var_blk, blk_next);
    m.branch(lbl_outer_header);

    // --- Outer loop merge ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_merge.0]));

    let final_sum = m.load(t_f32, var_sum);
    let out_ptr = m.access_chain(t_ptr_sb_f32, gvar_output, &[c_u32_0, row]);
    m.store(out_ptr, final_sum);
    m.branch(lbl_bounds_merge);

    // --- Bounds merge ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_merge.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// Emit a complete Q4_K GEMV compute shader as SPIR-V bytecode.
///
/// Weight buffer uses transposed SoA layout for coalesced access:
///   weight[(block * 36 + plane) * rows + row]
///
/// Q4_K block structure (256 elements, 36 planes):
///   Plane 0:     d(f16, lower 16 bits) + dmin(f16, upper 16 bits)
///   Planes 1-3:  12 bytes of 6-bit packed scales/mins
///   Planes 4-35: 128 bytes of 4-bit nibbles (32 u32 words)
///
/// Storage buffers (set=0):
///   binding=0: weight (uint array, transposed SoA)
///   binding=1: input  (f32 array)
///   binding=2: output (f32 array)
///
/// Push constants: { rows: u32, cols: u32, rows_per_wg: u32 }
pub fn emit_q4k_gemv(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    // Capabilities & memory model
    m.capability(1); // Shader
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1); // Logical, GLSL450

    // --- Types ---
    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let _t_i32 = m.type_int(32, 1);
    let t_f32 = m.type_float(32);

    let t_v3u32 = m.type_vector(t_u32, 3);

    let t_arr_u32 = m.type_runtime_array(t_u32);
    let t_arr_f32 = m.type_runtime_array(t_f32);

    let t_struct_weight = m.type_struct(&[t_arr_u32]);
    let t_struct_input = m.type_struct(&[t_arr_f32]);
    let t_struct_output = m.type_struct(&[t_arr_f32]);

    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);

    // Pointer types
    let t_ptr_sb_struct_weight = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_weight);
    let t_ptr_sb_struct_input = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_input);
    let t_ptr_sb_struct_output = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_output);
    let t_ptr_pc_struct = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_u32 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);

    let t_fn_void = m.type_function(t_void, &[]);

    // --- Constants ---
    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_4 = m.constant_u32(t_u32, 4);
    let c_u32_10 = m.constant_u32(t_u32, 10);
    let c_u32_13 = m.constant_u32(t_u32, 13);
    let c_u32_15 = m.constant_u32(t_u32, 15);
    let c_u32_16 = m.constant_u32(t_u32, 16);
    let c_u32_23 = m.constant_u32(t_u32, 23);
    let c_u32_31 = m.constant_u32(t_u32, 31);
    let c_u32_36 = m.constant_u32(t_u32, 36);
    let c_u32_ff = m.constant_u32(t_u32, 0xFF);
    let c_u32_0f = m.constant_u32(t_u32, 0x0F);
    let c_u32_3f = m.constant_u32(t_u32, 0x3F);
    let c_u32_ffff = m.constant_u32(t_u32, 0xFFFF);
    let c_u32_1f = m.constant_u32(t_u32, 0x1F);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3FF);
    let c_u32_112 = m.constant_u32(t_u32, 112);
    let c_u32_256 = m.constant_u32(t_u32, 256);

    let c_f32_0 = m.constant_f32(t_f32, 0.0);

    // --- Decorations ---
    m.decorate(t_struct_weight, decoration::BLOCK, &[]);
    m.decorate(t_struct_input, decoration::BLOCK, &[]);
    m.decorate(t_struct_output, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_weight, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_input, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_output, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_u32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);

    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);

    // --- Global variables ---
    let gvar_weight = m.variable(t_ptr_sb_struct_weight, storage_class::STORAGE_BUFFER);
    let gvar_input = m.variable(t_ptr_sb_struct_input, storage_class::STORAGE_BUFFER);
    let gvar_output = m.variable(t_ptr_sb_struct_output, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc_struct, storage_class::PUSH_CONSTANT);
    let gvar_glob_id = m.variable(t_ptr_input_u32, storage_class::INPUT);

    m.decorate(gvar_weight, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_weight, decoration::BINDING, &[0]);
    m.decorate(gvar_input, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_input, decoration::BINDING, &[1]);
    m.decorate(gvar_output, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_output, decoration::BINDING, &[2]);
    m.decorate(
        gvar_glob_id,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    // --- Entry point ---
    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_glob_id]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    // Pre-allocate labels
    let lbl_entry = m.alloc_id();
    let lbl_bounds_merge = m.alloc_id();
    let lbl_bounds_true = m.alloc_id();
    let lbl_outer_header = m.alloc_id();
    let lbl_outer_body = m.alloc_id();
    let lbl_outer_continue = m.alloc_id();
    let lbl_outer_merge = m.alloc_id();

    // --- Function ---
    m.function(t_void, func_id, 0, t_fn_void);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    // Function variables
    let var_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_blk = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);

    // Load GlobalInvocationID.x → row
    let glob_id_vec = m.load(t_v3u32, gvar_glob_id);
    let row = m.composite_extract(t_u32, glob_id_vec, 0);

    // Load push constants: rows, cols
    let pc_ptr_rows = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let pc_rows = m.load(t_u32, pc_ptr_rows);
    let pc_ptr_cols = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let pc_cols = m.load(t_u32, pc_ptr_cols);

    // Bounds check: row < rows
    let in_bounds = m.u_less_than(t_bool, row, pc_rows);
    m.selection_merge(lbl_bounds_merge, 0);
    m.branch_conditional(in_bounds, lbl_bounds_true, lbl_bounds_merge);

    // --- In-bounds block ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_true.0]));

    m.store(var_sum, c_f32_0);

    // num_blocks = cols / 256
    let num_blocks = m.udiv(t_u32, pc_cols, c_u32_256);

    m.store(var_blk, c_u32_0);
    m.branch(lbl_outer_header);

    // --- Outer loop header (block loop) ---
    let lbl_outer_cond = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_header.0]));
    m.loop_merge(lbl_outer_merge, lbl_outer_continue, 0);
    m.branch(lbl_outer_cond);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_cond.0]));
    let blk_cur = m.load(t_u32, var_blk);
    let outer_cond = m.u_less_than(t_bool, blk_cur, num_blocks);
    m.branch_conditional(outer_cond, lbl_outer_body, lbl_outer_merge);

    // --- Outer loop body ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_body.0]));

    // === TRANSPOSED SoA weight layout ===
    // weight[(block * 36 + plane) * rows + row]
    // plane_base_offset = block * 36 * rows
    let blk_x_36 = m.imul(t_u32, blk_cur, c_u32_36);
    let plane_base = m.imul(t_u32, blk_x_36, pc_rows);

    // ============================================================
    // Load d and dmin from plane 0
    // packed = weight[plane_base + row]
    // d    = f16_to_f32(packed & 0xFFFF)
    // dmin = f16_to_f32(packed >> 16)
    // ============================================================
    let packed_addr = m.iadd(t_u32, plane_base, row);
    let packed_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, packed_addr]);
    let packed_word = m.load(t_u32, packed_ptr);

    // f16→f32 helper constant: 2^(-24) = 5.9604644775390625e-8
    // Used for denormal conversion: value = mant * 2^(-24)
    let c_f32_2pow_neg24 = m.constant_f32(t_f32, 5.9604644775390625e-8);

    // f16→f32 with proper denormal + sign handling
    let d_f16_raw = m.bitwise_and(t_u32, packed_word, c_u32_ffff);
    let d_f32 = {
        let exp_raw = m.shift_right_logical(t_u32, d_f16_raw, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, d_f16_raw, c_u32_3ff);
        let sign = m.shift_right_logical(t_u32, d_f16_raw, c_u32_15);
        let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
        let s_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
        let e_adj = m.iadd(t_u32, exp, c_u32_112);
        let e_part = m.shift_left_logical(t_u32, e_adj, c_u32_23);
        let m_part = m.shift_left_logical(t_u32, mant, c_u32_13);
        let bits_mid = m.bitwise_or(t_u32, s_part, e_part);
        let bits = m.bitwise_or(t_u32, bits_mid, m_part);
        let normal = m.bitcast(t_f32, bits);
        // Denormal path with sign: mant * 2^(-24), then negate if sign bit set
        let mant_f = m.convert_u_to_f(t_f32, mant);
        let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
        let denorm_neg = m.fnegate(t_f32, denorm_abs);
        let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
        let denormal = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
        let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
        m.select(t_f32, exp_nonzero, normal, denormal)
    };

    // dmin = f16_to_f32(packed >> 16) — same logic with sign
    let dmin_f16_raw = m.shift_right_logical(t_u32, packed_word, c_u32_16);
    let dmin_f32 = {
        let exp_raw = m.shift_right_logical(t_u32, dmin_f16_raw, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, dmin_f16_raw, c_u32_3ff);
        let sign = m.shift_right_logical(t_u32, dmin_f16_raw, c_u32_15);
        let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
        let s_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
        let e_adj = m.iadd(t_u32, exp, c_u32_112);
        let e_part = m.shift_left_logical(t_u32, e_adj, c_u32_23);
        let m_part = m.shift_left_logical(t_u32, mant, c_u32_13);
        let bits_mid = m.bitwise_or(t_u32, s_part, e_part);
        let bits = m.bitwise_or(t_u32, bits_mid, m_part);
        let normal = m.bitcast(t_f32, bits);
        let mant_f = m.convert_u_to_f(t_f32, mant);
        let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
        let denorm_neg = m.fnegate(t_f32, denorm_abs);
        let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
        let denormal = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
        let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
        m.select(t_f32, exp_nonzero, normal, denormal)
    };

    // ============================================================
    // Load scales bytes from planes 1-3
    // s0 = weight[(plane_base + 1*rows) + row]  (bytes 0-3)
    // s1 = weight[(plane_base + 2*rows) + row]  (bytes 4-7)
    // s2 = weight[(plane_base + 3*rows) + row]  (bytes 8-11)
    // ============================================================
    let s0_addr = m.iadd(t_u32, plane_base, pc_rows);
    let s0_addr = m.iadd(t_u32, s0_addr, row);
    let s0_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, s0_addr]);
    let s0_word = m.load(t_u32, s0_ptr);

    let s1_offset = {
        let two_rows = m.imul(t_u32, c_u32_2, pc_rows);
        m.iadd(t_u32, plane_base, two_rows)
    };
    let s1_addr = m.iadd(t_u32, s1_offset, row);
    let s1_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, s1_addr]);
    let s1_word = m.load(t_u32, s1_ptr);

    let s2_offset = {
        let three_rows = m.imul(t_u32, c_u32_3, pc_rows);
        m.iadd(t_u32, plane_base, three_rows)
    };
    let s2_addr = m.iadd(t_u32, s2_offset, row);
    let s2_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, s2_addr]);
    let s2_word = m.load(t_u32, s2_ptr);

    // ============================================================
    // Extract 12 individual bytes: sb[0..11]
    //   sb[0..3] from s0_word, sb[4..7] from s1_word, sb[8..11] from s2_word
    // ============================================================
    let mut sb = [c_u32_0; 12];
    for i in 0..4u32 {
        let word = match i / 4 {
            _ => {
                // i is 0..3, always s0_word
                s0_word
            }
        };
        if i == 0 {
            sb[0] = m.bitwise_and(t_u32, word, c_u32_ff);
        } else {
            let shift = m.constant_u32(t_u32, i * 8);
            let shifted = m.shift_right_logical(t_u32, word, shift);
            sb[i as usize] = m.bitwise_and(t_u32, shifted, c_u32_ff);
        }
    }
    for i in 0..4u32 {
        if i == 0 {
            sb[4] = m.bitwise_and(t_u32, s1_word, c_u32_ff);
        } else {
            let shift = m.constant_u32(t_u32, i * 8);
            let shifted = m.shift_right_logical(t_u32, s1_word, shift);
            sb[4 + i as usize] = m.bitwise_and(t_u32, shifted, c_u32_ff);
        }
    }
    for i in 0..4u32 {
        if i == 0 {
            sb[8] = m.bitwise_and(t_u32, s2_word, c_u32_ff);
        } else {
            let shift = m.constant_u32(t_u32, i * 8);
            let shifted = m.shift_right_logical(t_u32, s2_word, shift);
            sb[8 + i as usize] = m.bitwise_and(t_u32, shifted, c_u32_ff);
        }
    }

    // ============================================================
    // Extract 8 scales and 8 mins using 6-bit packing
    //
    // For j in 0..4:
    //   sc[j] = sb[j] & 63
    //   mn[j] = sb[j+4] & 63
    //
    // For j in 4..8:
    //   sc[j] = (sb[j+4] & 0x0F) | ((sb[j-4] >> 6) << 4)
    //   mn[j] = (sb[j+4] >> 4)   | ((sb[j]   >> 6) << 4)
    // ============================================================
    let c_u32_6 = m.constant_u32(t_u32, 6);

    let mut scales = [c_u32_0; 8];
    let mut mins = [c_u32_0; 8];

    for j in 0..4usize {
        scales[j] = m.bitwise_and(t_u32, sb[j], c_u32_3f);
        mins[j] = m.bitwise_and(t_u32, sb[j + 4], c_u32_3f);
    }

    for j in 4..8usize {
        // sc[j] = (sb[j+4] & 0x0F) | ((sb[j-4] >> 6) << 4)
        let lo = m.bitwise_and(t_u32, sb[j + 4], c_u32_0f);
        let hi_raw = m.shift_right_logical(t_u32, sb[j - 4], c_u32_6);
        let hi = m.shift_left_logical(t_u32, hi_raw, c_u32_4);
        scales[j] = m.bitwise_or(t_u32, lo, hi);

        // mn[j] = (sb[j+4] >> 4) | ((sb[j] >> 6) << 4)
        let lo2 = m.shift_right_logical(t_u32, sb[j + 4], c_u32_4);
        let hi2_raw = m.shift_right_logical(t_u32, sb[j], c_u32_6);
        let hi2 = m.shift_left_logical(t_u32, hi2_raw, c_u32_4);
        mins[j] = m.bitwise_or(t_u32, lo2, hi2);
    }

    // ============================================================
    // Process 8 sub-blocks, fully unrolled
    // Using optimized form:
    //   sum += d * sc[sb] * sum(nibble * input) - dmin * mn[sb] * sum(input)
    // ============================================================
    let blk_x_256 = m.imul(t_u32, blk_cur, c_u32_256);
    let mut total_sum = c_f32_0;

    for sb_idx in 0..8u32 {
        let sc_f = {
            let sc_u = scales[sb_idx as usize];
            m.convert_u_to_f(t_f32, sc_u)
        };
        let mn_f = {
            let mn_u = mins[sb_idx as usize];
            m.convert_u_to_f(t_f32, mn_u)
        };

        let qs_group = sb_idx / 2;
        let is_high = (sb_idx & 1) != 0;
        let qs_base_plane = 4 + qs_group * 8;

        let c_qs_bp = m.constant_u32(t_u32, qs_base_plane);
        let qs_plane_offset = m.imul(t_u32, c_qs_bp, pc_rows);
        let qs_plane_start = m.iadd(t_u32, plane_base, qs_plane_offset);

        let c_sb_offset = m.constant_u32(t_u32, sb_idx * 32);
        let inp_sb_base = m.iadd(t_u32, blk_x_256, c_sb_offset);

        let mut nibble_input_sum = c_f32_0;
        let mut input_sum = c_f32_0;

        for w in 0..8u32 {
            let w_offset = if w == 0 {
                qs_plane_start
            } else {
                let c_w = m.constant_u32(t_u32, w);
                let w_x_rows = m.imul(t_u32, c_w, pc_rows);
                m.iadd(t_u32, qs_plane_start, w_x_rows)
            };
            let qs_addr = m.iadd(t_u32, w_offset, row);
            let qs_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, qs_addr]);
            let qs_word = m.load(t_u32, qs_ptr);

            let inp_w_base = if w == 0 {
                inp_sb_base
            } else {
                let c_w4 = m.constant_u32(t_u32, w * 4);
                m.iadd(t_u32, inp_sb_base, c_w4)
            };

            for byte_idx in 0..4u32 {
                let shift_amt = if is_high {
                    byte_idx * 8 + 4
                } else {
                    byte_idx * 8
                };
                let nibble = if shift_amt == 0 {
                    m.bitwise_and(t_u32, qs_word, c_u32_0f)
                } else {
                    let c_shift = m.constant_u32(t_u32, shift_amt);
                    let shifted = m.shift_right_logical(t_u32, qs_word, c_shift);
                    m.bitwise_and(t_u32, shifted, c_u32_0f)
                };
                let nibble_f = m.convert_u_to_f(t_f32, nibble);

                let inp_idx = if byte_idx == 0 {
                    inp_w_base
                } else {
                    let c_bi = m.constant_u32(t_u32, byte_idx);
                    m.iadd(t_u32, inp_w_base, c_bi)
                };
                let inp_ptr = m.access_chain(t_ptr_sb_f32, gvar_input, &[c_u32_0, inp_idx]);
                let inp_val = m.load(t_f32, inp_ptr);

                let ni_prod = m.fmul(t_f32, nibble_f, inp_val);
                nibble_input_sum = m.fadd(t_f32, nibble_input_sum, ni_prod);
                input_sum = m.fadd(t_f32, input_sum, inp_val);
            }
        }

        let d_sc = m.fmul(t_f32, d_f32, sc_f);
        let term1 = m.fmul(t_f32, d_sc, nibble_input_sum);
        let dmin_mn = m.fmul(t_f32, dmin_f32, mn_f);
        let term2 = m.fmul(t_f32, dmin_mn, input_sum);
        let sb_result = m.fsub(t_f32, term1, term2);
        total_sum = m.fadd(t_f32, total_sum, sb_result);
    }

    // Accumulate block result
    let prev_total = m.load(t_f32, var_sum);
    let new_total = m.fadd(t_f32, prev_total, total_sum);
    m.store(var_sum, new_total);
    m.branch(lbl_outer_continue);

    // --- Outer loop continue ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_continue.0]));
    let blk_next = m.iadd(t_u32, blk_cur, c_u32_1);
    m.store(var_blk, blk_next);
    m.branch(lbl_outer_header);

    // --- Outer loop merge ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_merge.0]));

    let final_sum = m.load(t_f32, var_sum);
    let out_ptr = m.access_chain(t_ptr_sb_f32, gvar_output, &[c_u32_0, row]);
    m.store(out_ptr, final_sum);
    m.branch(lbl_bounds_merge);

    // --- Bounds merge ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_merge.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// Emit a Q4_K weight × Q8K activation integer-dot GEMV compute shader.
///
/// Mirrors the CPU strict fast path semantics (`dot_q4_k_q8k_scalar`):
///   per block: sumi = Σ sc[is]*Σ(lo_nibble × q8b.qs) + sc[is+1]*Σ(hi_nibble × q8b.qs)
///              summ = Σ mn[is]*bsums[2g] + mn[is+1]*bsums[2g+1]
///              acc += q8b.d * (d * sumi - dmin * summ)
///
/// Weight buffer (binding 0) is identical to `emit_q4k_gemv` (transposed SoA).
///
/// Activation buffer (binding 1) is a packed u32 array, per-block stride = 69 u32 (276 B):
///   words 0..64  : qs (256 signed i8 values, 4 bytes per u32, little-endian)
///   word  64     : d (f32 bitcast as u32)
///   words 65..69 : bsums (8 signed i16 values, 2 per u32, low half then high half)
///
/// Output buffer (binding 2): f32 array, one element per row.
/// Push constants: { rows: u32, cols: u32, rows_per_wg: u32 }.
pub fn emit_q4k_q8k_gemv(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1); // Shader
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1); // Logical, GLSL450

    // --- Types ---
    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_i32 = m.type_int(32, 1);
    let t_f32 = m.type_float(32);

    let t_v3u32 = m.type_vector(t_u32, 3);

    let t_arr_u32 = m.type_runtime_array(t_u32);
    let t_arr_f32 = m.type_runtime_array(t_f32);

    let t_struct_weight = m.type_struct(&[t_arr_u32]);
    let t_struct_input = m.type_struct(&[t_arr_u32]); // Q8K packed
    let t_struct_output = m.type_struct(&[t_arr_f32]);

    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);

    let t_ptr_sb_struct_weight = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_weight);
    let t_ptr_sb_struct_input = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_input);
    let t_ptr_sb_struct_output = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_output);
    let t_ptr_pc_struct = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_u32 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);

    let t_fn_void = m.type_function(t_void, &[]);

    // --- Constants ---
    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_4 = m.constant_u32(t_u32, 4);
    let c_u32_10 = m.constant_u32(t_u32, 10);
    let c_u32_13 = m.constant_u32(t_u32, 13);
    let c_u32_15 = m.constant_u32(t_u32, 15);
    let c_u32_16 = m.constant_u32(t_u32, 16);
    let c_u32_23 = m.constant_u32(t_u32, 23);
    let c_u32_24 = m.constant_u32(t_u32, 24);
    let c_u32_31 = m.constant_u32(t_u32, 31);
    let c_u32_36 = m.constant_u32(t_u32, 36);
    let c_u32_64 = m.constant_u32(t_u32, 64);
    let c_u32_65 = m.constant_u32(t_u32, 65);
    let c_u32_69 = m.constant_u32(t_u32, 69);
    let c_u32_256 = m.constant_u32(t_u32, 256);
    let c_u32_ff = m.constant_u32(t_u32, 0xFF);
    let c_u32_0f = m.constant_u32(t_u32, 0x0F);
    let c_u32_3f = m.constant_u32(t_u32, 0x3F);
    let c_u32_ffff = m.constant_u32(t_u32, 0xFFFF);
    let c_u32_1f = m.constant_u32(t_u32, 0x1F);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3FF);
    let c_u32_112 = m.constant_u32(t_u32, 112);
    let c_u32_6 = m.constant_u32(t_u32, 6);

    let c_i32_0 = m.constant_u32(t_i32, 0);

    let c_f32_0 = m.constant_f32(t_f32, 0.0);

    // --- Decorations ---
    m.decorate(t_struct_weight, decoration::BLOCK, &[]);
    m.decorate(t_struct_input, decoration::BLOCK, &[]);
    m.decorate(t_struct_output, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_weight, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_input, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_output, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_u32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);

    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);

    // --- Global variables ---
    let gvar_weight = m.variable(t_ptr_sb_struct_weight, storage_class::STORAGE_BUFFER);
    let gvar_input = m.variable(t_ptr_sb_struct_input, storage_class::STORAGE_BUFFER);
    let gvar_output = m.variable(t_ptr_sb_struct_output, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc_struct, storage_class::PUSH_CONSTANT);
    let gvar_glob_id = m.variable(t_ptr_input_u32, storage_class::INPUT);

    m.decorate(gvar_weight, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_weight, decoration::BINDING, &[0]);
    m.decorate(gvar_input, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_input, decoration::BINDING, &[1]);
    m.decorate(gvar_output, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_output, decoration::BINDING, &[2]);
    m.decorate(
        gvar_glob_id,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    // --- Entry point ---
    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_glob_id]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    // Pre-allocate labels
    let lbl_entry = m.alloc_id();
    let lbl_bounds_merge = m.alloc_id();
    let lbl_bounds_true = m.alloc_id();
    let lbl_outer_header = m.alloc_id();
    let lbl_outer_body = m.alloc_id();
    let lbl_outer_continue = m.alloc_id();
    let lbl_outer_merge = m.alloc_id();

    // --- Function ---
    m.function(t_void, func_id, 0, t_fn_void);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let var_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_blk = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);

    // Load GlobalInvocationID.x → row
    let glob_id_vec = m.load(t_v3u32, gvar_glob_id);
    let row = m.composite_extract(t_u32, glob_id_vec, 0);

    let pc_ptr_rows = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let pc_rows = m.load(t_u32, pc_ptr_rows);
    let pc_ptr_cols = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let pc_cols = m.load(t_u32, pc_ptr_cols);

    let in_bounds = m.u_less_than(t_bool, row, pc_rows);
    m.selection_merge(lbl_bounds_merge, 0);
    m.branch_conditional(in_bounds, lbl_bounds_true, lbl_bounds_merge);

    // --- In-bounds block ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_true.0]));

    m.store(var_sum, c_f32_0);

    let num_blocks = m.udiv(t_u32, pc_cols, c_u32_256);

    m.store(var_blk, c_u32_0);
    m.branch(lbl_outer_header);

    // --- Outer loop header ---
    let lbl_outer_cond = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_header.0]));
    m.loop_merge(lbl_outer_merge, lbl_outer_continue, 0);
    m.branch(lbl_outer_cond);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_cond.0]));
    let blk_cur = m.load(t_u32, var_blk);
    let outer_cond = m.u_less_than(t_bool, blk_cur, num_blocks);
    m.branch_conditional(outer_cond, lbl_outer_body, lbl_outer_merge);

    // --- Outer loop body ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_body.0]));

    // Weight: transposed SoA, plane_base = block * 36 * rows
    let blk_x_36 = m.imul(t_u32, blk_cur, c_u32_36);
    let plane_base = m.imul(t_u32, blk_x_36, pc_rows);

    // Activation: per-block stride 69 u32, qs words 0..64, d word 64, bsums words 65..69
    let act_base = m.imul(t_u32, blk_cur, c_u32_69);

    // ---- Load weight d, dmin from plane 0 ----
    let packed_addr = m.iadd(t_u32, plane_base, row);
    let packed_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, packed_addr]);
    let packed_word = m.load(t_u32, packed_ptr);

    let c_f32_2pow_neg24 = m.constant_f32(t_f32, 5.9604644775390625e-8);

    let d_f16_raw = m.bitwise_and(t_u32, packed_word, c_u32_ffff);
    let d_f32 = {
        let exp_raw = m.shift_right_logical(t_u32, d_f16_raw, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, d_f16_raw, c_u32_3ff);
        let sign = m.shift_right_logical(t_u32, d_f16_raw, c_u32_15);
        let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
        let s_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
        let e_adj = m.iadd(t_u32, exp, c_u32_112);
        let e_part = m.shift_left_logical(t_u32, e_adj, c_u32_23);
        let m_part = m.shift_left_logical(t_u32, mant, c_u32_13);
        let bits_mid = m.bitwise_or(t_u32, s_part, e_part);
        let bits = m.bitwise_or(t_u32, bits_mid, m_part);
        let normal = m.bitcast(t_f32, bits);
        let mant_f = m.convert_u_to_f(t_f32, mant);
        let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
        let denorm_neg = m.fnegate(t_f32, denorm_abs);
        let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
        let denormal = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
        let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
        m.select(t_f32, exp_nonzero, normal, denormal)
    };

    let dmin_f16_raw = m.shift_right_logical(t_u32, packed_word, c_u32_16);
    let dmin_f32 = {
        let exp_raw = m.shift_right_logical(t_u32, dmin_f16_raw, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, dmin_f16_raw, c_u32_3ff);
        let sign = m.shift_right_logical(t_u32, dmin_f16_raw, c_u32_15);
        let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
        let s_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
        let e_adj = m.iadd(t_u32, exp, c_u32_112);
        let e_part = m.shift_left_logical(t_u32, e_adj, c_u32_23);
        let m_part = m.shift_left_logical(t_u32, mant, c_u32_13);
        let bits_mid = m.bitwise_or(t_u32, s_part, e_part);
        let bits = m.bitwise_or(t_u32, bits_mid, m_part);
        let normal = m.bitcast(t_f32, bits);
        let mant_f = m.convert_u_to_f(t_f32, mant);
        let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
        let denorm_neg = m.fnegate(t_f32, denorm_abs);
        let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
        let denormal = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
        let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
        m.select(t_f32, exp_nonzero, normal, denormal)
    };

    // ---- Load 12 bytes scales/mins from planes 1..3 ----
    let s0_addr = m.iadd(t_u32, plane_base, pc_rows);
    let s0_addr = m.iadd(t_u32, s0_addr, row);
    let s0_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, s0_addr]);
    let s0_word = m.load(t_u32, s0_ptr);

    let s1_offset = {
        let two_rows = m.imul(t_u32, c_u32_2, pc_rows);
        m.iadd(t_u32, plane_base, two_rows)
    };
    let s1_addr = m.iadd(t_u32, s1_offset, row);
    let s1_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, s1_addr]);
    let s1_word = m.load(t_u32, s1_ptr);

    let s2_offset = {
        let three_rows = m.imul(t_u32, c_u32_3, pc_rows);
        m.iadd(t_u32, plane_base, three_rows)
    };
    let s2_addr = m.iadd(t_u32, s2_offset, row);
    let s2_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, s2_addr]);
    let s2_word = m.load(t_u32, s2_ptr);

    let mut sb = [c_u32_0; 12];
    for i in 0..4u32 {
        if i == 0 {
            sb[0] = m.bitwise_and(t_u32, s0_word, c_u32_ff);
        } else {
            let shift = m.constant_u32(t_u32, i * 8);
            let shifted = m.shift_right_logical(t_u32, s0_word, shift);
            sb[i as usize] = m.bitwise_and(t_u32, shifted, c_u32_ff);
        }
    }
    for i in 0..4u32 {
        if i == 0 {
            sb[4] = m.bitwise_and(t_u32, s1_word, c_u32_ff);
        } else {
            let shift = m.constant_u32(t_u32, i * 8);
            let shifted = m.shift_right_logical(t_u32, s1_word, shift);
            sb[4 + i as usize] = m.bitwise_and(t_u32, shifted, c_u32_ff);
        }
    }
    for i in 0..4u32 {
        if i == 0 {
            sb[8] = m.bitwise_and(t_u32, s2_word, c_u32_ff);
        } else {
            let shift = m.constant_u32(t_u32, i * 8);
            let shifted = m.shift_right_logical(t_u32, s2_word, shift);
            sb[8 + i as usize] = m.bitwise_and(t_u32, shifted, c_u32_ff);
        }
    }

    // 8 scales + 8 mins (6-bit packed) — same extraction as f32 variant
    let mut scales_u = [c_u32_0; 8];
    let mut mins_u = [c_u32_0; 8];
    for j in 0..4usize {
        scales_u[j] = m.bitwise_and(t_u32, sb[j], c_u32_3f);
        mins_u[j] = m.bitwise_and(t_u32, sb[j + 4], c_u32_3f);
    }
    for j in 4..8usize {
        let lo = m.bitwise_and(t_u32, sb[j + 4], c_u32_0f);
        let hi_raw = m.shift_right_logical(t_u32, sb[j - 4], c_u32_6);
        let hi = m.shift_left_logical(t_u32, hi_raw, c_u32_4);
        scales_u[j] = m.bitwise_or(t_u32, lo, hi);

        let lo2 = m.shift_right_logical(t_u32, sb[j + 4], c_u32_4);
        let hi2_raw = m.shift_right_logical(t_u32, sb[j], c_u32_6);
        let hi2 = m.shift_left_logical(t_u32, hi2_raw, c_u32_4);
        mins_u[j] = m.bitwise_or(t_u32, lo2, hi2);
    }
    // Reinterpret as i32 for integer multiply
    let mut scales_i = [c_i32_0; 8];
    let mut mins_i = [c_i32_0; 8];
    for j in 0..8usize {
        scales_i[j] = m.bitcast(t_i32, scales_u[j]);
        mins_i[j] = m.bitcast(t_i32, mins_u[j]);
    }

    // ---- Load activation block scale d (word 64) ----
    let q8k_d_addr = m.iadd(t_u32, act_base, c_u32_64);
    let q8k_d_ptr = m.access_chain(t_ptr_sb_u32, gvar_input, &[c_u32_0, q8k_d_addr]);
    let q8k_d_word = m.load(t_u32, q8k_d_ptr);
    let q8k_d = m.bitcast(t_f32, q8k_d_word);

    // ---- Integer dot accumulators across 4 groups ----
    // Per CPU semantics: sumi & summ are per-block i32 accumulators,
    // promoted to f32 once at end via q8b.d * (d * sumi - dmin * summ).
    let mut sumi = c_i32_0;
    let mut summ = c_i32_0;

    // Precompute act_qs base (word offset 0) and act_bsums base (word offset 65).
    let act_bsums_base = m.iadd(t_u32, act_base, c_u32_65);

    for group in 0..4u32 {
        let is = (group * 2) as usize;
        // Weight nibble planes for this group: planes (4 + group*8) .. (4 + group*8 + 8)
        let qs_base_plane = 4 + group * 8;
        let c_qs_bp = m.constant_u32(t_u32, qs_base_plane);
        let qs_plane_offset = m.imul(t_u32, c_qs_bp, pc_rows);
        let qs_plane_start = m.iadd(t_u32, plane_base, qs_plane_offset);

        // Activation qs words for this group:
        //   sub-block 2g (lo nibbles): act_base + group*16 + (0..4)   — 4 u32 = 16 bytes? No: 4 u32 * 4 = 16 bytes only.
        // Wait: each sub-block has 32 i8 → 8 u32 words. 2 sub-blocks per group (lo+hi paired) → 16 u32.
        // group*16 stride is correct for activation qs.
        let c_g16 = m.constant_u32(t_u32, group * 16);
        let act_lo_base = m.iadd(t_u32, act_base, c_g16);
        let c_g16_plus_8 = m.constant_u32(t_u32, group * 16 + 8);
        let act_hi_base = m.iadd(t_u32, act_base, c_g16_plus_8);

        let mut isum0 = c_i32_0;
        let mut isum1 = c_i32_0;

        // 8 weight u32 words per sub-block-pair → covers 32 bytes nibble
        // each byte: lo nibble × q8b.qs[lo_block + l], hi nibble × q8b.qs[hi_block + l]
        for w in 0..8u32 {
            // weight word at plane (qs_base_plane + w)
            let w_offset = if w == 0 {
                qs_plane_start
            } else {
                let c_w = m.constant_u32(t_u32, w);
                let w_x_rows = m.imul(t_u32, c_w, pc_rows);
                m.iadd(t_u32, qs_plane_start, w_x_rows)
            };
            let qs_addr = m.iadd(t_u32, w_offset, row);
            let qs_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, qs_addr]);
            let qs_word = m.load(t_u32, qs_ptr);

            // activation u32 words: lo @ act_lo_base + w, hi @ act_hi_base + w
            let act_lo_addr = if w == 0 {
                act_lo_base
            } else {
                let c_w = m.constant_u32(t_u32, w);
                m.iadd(t_u32, act_lo_base, c_w)
            };
            let act_lo_ptr = m.access_chain(t_ptr_sb_u32, gvar_input, &[c_u32_0, act_lo_addr]);
            let act_lo_word = m.load(t_u32, act_lo_ptr);

            let act_hi_addr = if w == 0 {
                act_hi_base
            } else {
                let c_w = m.constant_u32(t_u32, w);
                m.iadd(t_u32, act_hi_base, c_w)
            };
            let act_hi_ptr = m.access_chain(t_ptr_sb_u32, gvar_input, &[c_u32_0, act_hi_addr]);
            let act_hi_word = m.load(t_u32, act_hi_ptr);

            for byte_idx in 0..4u32 {
                // lo nibble (unsigned 0..15) at byte_idx
                let lo_nib = if byte_idx == 0 {
                    m.bitwise_and(t_u32, qs_word, c_u32_0f)
                } else {
                    let c_shift = m.constant_u32(t_u32, byte_idx * 8);
                    let shifted = m.shift_right_logical(t_u32, qs_word, c_shift);
                    m.bitwise_and(t_u32, shifted, c_u32_0f)
                };
                // hi nibble (unsigned 0..15) at byte_idx + 4 bits
                let c_shift_hi = m.constant_u32(t_u32, byte_idx * 8 + 4);
                let hi_shifted = m.shift_right_logical(t_u32, qs_word, c_shift_hi);
                let hi_nib = m.bitwise_and(t_u32, hi_shifted, c_u32_0f);

                let lo_nib_i = m.bitcast(t_i32, lo_nib);
                let hi_nib_i = m.bitcast(t_i32, hi_nib);

                // Q8K activation byte (signed i8) — sign-extend via 24-bit shift trick.
                // For lo sub-block:
                let act_lo_byte = if byte_idx == 0 {
                    act_lo_word
                } else {
                    let c_shift = m.constant_u32(t_u32, byte_idx * 8);
                    m.shift_right_logical(t_u32, act_lo_word, c_shift)
                };
                let act_lo_byte_top = m.shift_left_logical(t_u32, act_lo_byte, c_u32_24);
                let act_lo_byte_top_i = m.bitcast(t_i32, act_lo_byte_top);
                let x_lo = m.shift_right_arithmetic(t_i32, act_lo_byte_top_i, c_u32_24);

                let act_hi_byte = if byte_idx == 0 {
                    act_hi_word
                } else {
                    let c_shift = m.constant_u32(t_u32, byte_idx * 8);
                    m.shift_right_logical(t_u32, act_hi_word, c_shift)
                };
                let act_hi_byte_top = m.shift_left_logical(t_u32, act_hi_byte, c_u32_24);
                let act_hi_byte_top_i = m.bitcast(t_i32, act_hi_byte_top);
                let x_hi = m.shift_right_arithmetic(t_i32, act_hi_byte_top_i, c_u32_24);

                let prod_lo = m.imul(t_i32, lo_nib_i, x_lo);
                let prod_hi = m.imul(t_i32, hi_nib_i, x_hi);

                isum0 = m.iadd(t_i32, isum0, prod_lo);
                isum1 = m.iadd(t_i32, isum1, prod_hi);
            }
        }

        // sumi += sc[is]*isum0 + sc[is+1]*isum1
        let term_a = m.imul(t_i32, scales_i[is], isum0);
        let term_b = m.imul(t_i32, scales_i[is + 1], isum1);
        let term_ab = m.iadd(t_i32, term_a, term_b);
        sumi = m.iadd(t_i32, sumi, term_ab);

        // Load bsums word for this group (word 65 + group), low half = bsums[2g], high half = bsums[2g+1].
        let bsum_word_addr = if group == 0 {
            act_bsums_base
        } else {
            let c_g = m.constant_u32(t_u32, group);
            m.iadd(t_u32, act_bsums_base, c_g)
        };
        let bsum_ptr = m.access_chain(t_ptr_sb_u32, gvar_input, &[c_u32_0, bsum_word_addr]);
        let bsum_word = m.load(t_u32, bsum_ptr);
        // sign-extend low 16 bits → i32
        let bsum_lo_top = m.shift_left_logical(t_u32, bsum_word, c_u32_16);
        let bsum_lo_top_i = m.bitcast(t_i32, bsum_lo_top);
        let bsum_lo = m.shift_right_arithmetic(t_i32, bsum_lo_top_i, c_u32_16);
        // sign-extend high 16 bits → i32
        let bsum_word_i = m.bitcast(t_i32, bsum_word);
        let bsum_hi = m.shift_right_arithmetic(t_i32, bsum_word_i, c_u32_16);

        // summ += mn[is]*bsum_lo + mn[is+1]*bsum_hi
        let term_m_a = m.imul(t_i32, mins_i[is], bsum_lo);
        let term_m_b = m.imul(t_i32, mins_i[is + 1], bsum_hi);
        let term_m_ab = m.iadd(t_i32, term_m_a, term_m_b);
        summ = m.iadd(t_i32, summ, term_m_ab);
    }

    // Block-level f32 finalization: acc += q8b.d * (d * sumi - dmin * summ)
    let sumi_f = m.convert_s_to_f(t_f32, sumi);
    let summ_f = m.convert_s_to_f(t_f32, summ);
    let d_sumi = m.fmul(t_f32, d_f32, sumi_f);
    let dmin_summ = m.fmul(t_f32, dmin_f32, summ_f);
    let inner = m.fsub(t_f32, d_sumi, dmin_summ);
    let block_term = m.fmul(t_f32, q8k_d, inner);

    let prev_total = m.load(t_f32, var_sum);
    let new_total = m.fadd(t_f32, prev_total, block_term);
    m.store(var_sum, new_total);
    m.branch(lbl_outer_continue);

    // --- Outer loop continue ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_continue.0]));
    let blk_next = m.iadd(t_u32, blk_cur, c_u32_1);
    m.store(var_blk, blk_next);
    m.branch(lbl_outer_header);

    // --- Outer loop merge ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_merge.0]));

    let final_sum = m.load(t_f32, var_sum);
    let out_ptr = m.access_chain(t_ptr_sb_f32, gvar_output, &[c_u32_0, row]);
    m.store(out_ptr, final_sum);
    m.branch(lbl_bounds_merge);

    // --- Bounds merge ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_merge.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// Emit the first pass for block-parallel Q4_K GEMV.
///
/// Dispatch shape: `(ceil(rows / local_size_x), cols / 256, 1)`.
/// Each invocation computes one `(row, block)` partial dot and writes:
/// `partial[block * rows + row]`.
pub fn emit_q4k_gemv_block_partial(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_arr_u32 = m.type_runtime_array(t_u32);
    let t_arr_f32 = m.type_runtime_array(t_f32);
    let t_struct_weight = m.type_struct(&[t_arr_u32]);
    let t_struct_input = m.type_struct(&[t_arr_f32]);
    let t_struct_partial = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);

    let t_ptr_sb_struct_weight = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_weight);
    let t_ptr_sb_struct_input = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_input);
    let t_ptr_sb_struct_partial = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_partial);
    let t_ptr_pc_struct = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_u32 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_4 = m.constant_u32(t_u32, 4);
    let c_u32_10 = m.constant_u32(t_u32, 10);
    let c_u32_13 = m.constant_u32(t_u32, 13);
    let c_u32_15 = m.constant_u32(t_u32, 15);
    let c_u32_16 = m.constant_u32(t_u32, 16);
    let c_u32_23 = m.constant_u32(t_u32, 23);
    let c_u32_31 = m.constant_u32(t_u32, 31);
    let c_u32_36 = m.constant_u32(t_u32, 36);
    let c_u32_ff = m.constant_u32(t_u32, 0xFF);
    let c_u32_0f = m.constant_u32(t_u32, 0x0F);
    let c_u32_3f = m.constant_u32(t_u32, 0x3F);
    let c_u32_ffff = m.constant_u32(t_u32, 0xFFFF);
    let c_u32_1f = m.constant_u32(t_u32, 0x1F);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3FF);
    let c_u32_112 = m.constant_u32(t_u32, 112);
    let c_u32_256 = m.constant_u32(t_u32, 256);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);
    let c_f32_2pow_neg24 = m.constant_f32(t_f32, 5.9604644775390625e-8);

    m.decorate(t_struct_weight, decoration::BLOCK, &[]);
    m.decorate(t_struct_input, decoration::BLOCK, &[]);
    m.decorate(t_struct_partial, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_weight, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_input, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_partial, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_u32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);

    let gvar_weight = m.variable(t_ptr_sb_struct_weight, storage_class::STORAGE_BUFFER);
    let gvar_input = m.variable(t_ptr_sb_struct_input, storage_class::STORAGE_BUFFER);
    let gvar_partial = m.variable(t_ptr_sb_struct_partial, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc_struct, storage_class::PUSH_CONSTANT);
    let gvar_glob_id = m.variable(t_ptr_input_u32, storage_class::INPUT);

    m.decorate(gvar_weight, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_weight, decoration::BINDING, &[0]);
    m.decorate(gvar_input, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_input, decoration::BINDING, &[1]);
    m.decorate(gvar_partial, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_partial, decoration::BINDING, &[2]);
    m.decorate(
        gvar_glob_id,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_glob_id]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    let lbl_entry = m.alloc_id();
    let lbl_merge = m.alloc_id();
    let lbl_row_true = m.alloc_id();
    let lbl_block_true = m.alloc_id();
    let lbl_block_merge = m.alloc_id();

    m.function(t_void, func_id, 0, t_fn_void);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let glob_id_vec = m.load(t_v3u32, gvar_glob_id);
    let row = m.composite_extract(t_u32, glob_id_vec, 0);
    let blk_cur = m.composite_extract(t_u32, glob_id_vec, 1);

    let pc_ptr_rows = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let pc_rows = m.load(t_u32, pc_ptr_rows);
    let pc_ptr_cols = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let pc_cols = m.load(t_u32, pc_ptr_cols);
    let num_blocks = m.udiv(t_u32, pc_cols, c_u32_256);

    let row_in_bounds = m.u_less_than(t_bool, row, pc_rows);
    m.selection_merge(lbl_merge, 0);
    m.branch_conditional(row_in_bounds, lbl_row_true, lbl_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_row_true.0]));
    let block_in_bounds = m.u_less_than(t_bool, blk_cur, num_blocks);
    m.selection_merge(lbl_block_merge, 0);
    m.branch_conditional(block_in_bounds, lbl_block_true, lbl_block_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_block_true.0]));

    let blk_x_36 = m.imul(t_u32, blk_cur, c_u32_36);
    let plane_base = m.imul(t_u32, blk_x_36, pc_rows);
    let packed_addr = m.iadd(t_u32, plane_base, row);
    let packed_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, packed_addr]);
    let packed_word = m.load(t_u32, packed_ptr);

    let d_f16_raw = m.bitwise_and(t_u32, packed_word, c_u32_ffff);
    let d_f32 = {
        let exp_raw = m.shift_right_logical(t_u32, d_f16_raw, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, d_f16_raw, c_u32_3ff);
        let sign = m.shift_right_logical(t_u32, d_f16_raw, c_u32_15);
        let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
        let s_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
        let e_adj = m.iadd(t_u32, exp, c_u32_112);
        let e_part = m.shift_left_logical(t_u32, e_adj, c_u32_23);
        let m_part = m.shift_left_logical(t_u32, mant, c_u32_13);
        let bits_mid = m.bitwise_or(t_u32, s_part, e_part);
        let bits = m.bitwise_or(t_u32, bits_mid, m_part);
        let normal = m.bitcast(t_f32, bits);
        let mant_f = m.convert_u_to_f(t_f32, mant);
        let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
        let denorm_neg = m.fnegate(t_f32, denorm_abs);
        let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
        let denormal = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
        let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
        m.select(t_f32, exp_nonzero, normal, denormal)
    };

    let dmin_f16_raw = m.shift_right_logical(t_u32, packed_word, c_u32_16);
    let dmin_f32 = {
        let exp_raw = m.shift_right_logical(t_u32, dmin_f16_raw, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, dmin_f16_raw, c_u32_3ff);
        let sign = m.shift_right_logical(t_u32, dmin_f16_raw, c_u32_15);
        let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
        let s_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
        let e_adj = m.iadd(t_u32, exp, c_u32_112);
        let e_part = m.shift_left_logical(t_u32, e_adj, c_u32_23);
        let m_part = m.shift_left_logical(t_u32, mant, c_u32_13);
        let bits_mid = m.bitwise_or(t_u32, s_part, e_part);
        let bits = m.bitwise_or(t_u32, bits_mid, m_part);
        let normal = m.bitcast(t_f32, bits);
        let mant_f = m.convert_u_to_f(t_f32, mant);
        let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
        let denorm_neg = m.fnegate(t_f32, denorm_abs);
        let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
        let denormal = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
        let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
        m.select(t_f32, exp_nonzero, normal, denormal)
    };

    let s0_addr = m.iadd(t_u32, plane_base, pc_rows);
    let s0_addr = m.iadd(t_u32, s0_addr, row);
    let s0_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, s0_addr]);
    let s0_word = m.load(t_u32, s0_ptr);

    let two_rows = m.imul(t_u32, c_u32_2, pc_rows);
    let s1_offset = m.iadd(t_u32, plane_base, two_rows);
    let s1_addr = m.iadd(t_u32, s1_offset, row);
    let s1_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, s1_addr]);
    let s1_word = m.load(t_u32, s1_ptr);

    let three_rows = m.imul(t_u32, c_u32_3, pc_rows);
    let s2_offset = m.iadd(t_u32, plane_base, three_rows);
    let s2_addr = m.iadd(t_u32, s2_offset, row);
    let s2_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, s2_addr]);
    let s2_word = m.load(t_u32, s2_ptr);

    let mut sb = [c_u32_0; 12];
    for i in 0..4u32 {
        if i == 0 {
            sb[0] = m.bitwise_and(t_u32, s0_word, c_u32_ff);
        } else {
            let shift = m.constant_u32(t_u32, i * 8);
            let shifted = m.shift_right_logical(t_u32, s0_word, shift);
            sb[i as usize] = m.bitwise_and(t_u32, shifted, c_u32_ff);
        }
    }
    for i in 0..4u32 {
        if i == 0 {
            sb[4] = m.bitwise_and(t_u32, s1_word, c_u32_ff);
        } else {
            let shift = m.constant_u32(t_u32, i * 8);
            let shifted = m.shift_right_logical(t_u32, s1_word, shift);
            sb[4 + i as usize] = m.bitwise_and(t_u32, shifted, c_u32_ff);
        }
    }
    for i in 0..4u32 {
        if i == 0 {
            sb[8] = m.bitwise_and(t_u32, s2_word, c_u32_ff);
        } else {
            let shift = m.constant_u32(t_u32, i * 8);
            let shifted = m.shift_right_logical(t_u32, s2_word, shift);
            sb[8 + i as usize] = m.bitwise_and(t_u32, shifted, c_u32_ff);
        }
    }

    let c_u32_6 = m.constant_u32(t_u32, 6);
    let mut scales = [c_u32_0; 8];
    let mut mins = [c_u32_0; 8];
    for j in 0..4usize {
        scales[j] = m.bitwise_and(t_u32, sb[j], c_u32_3f);
        mins[j] = m.bitwise_and(t_u32, sb[j + 4], c_u32_3f);
    }
    for j in 4..8usize {
        let lo = m.bitwise_and(t_u32, sb[j + 4], c_u32_0f);
        let hi_raw = m.shift_right_logical(t_u32, sb[j - 4], c_u32_6);
        let hi = m.shift_left_logical(t_u32, hi_raw, c_u32_4);
        scales[j] = m.bitwise_or(t_u32, lo, hi);

        let lo2 = m.shift_right_logical(t_u32, sb[j + 4], c_u32_4);
        let hi2_raw = m.shift_right_logical(t_u32, sb[j], c_u32_6);
        let hi2 = m.shift_left_logical(t_u32, hi2_raw, c_u32_4);
        mins[j] = m.bitwise_or(t_u32, lo2, hi2);
    }

    let blk_x_256 = m.imul(t_u32, blk_cur, c_u32_256);
    let mut total_sum = c_f32_0;
    for sb_idx in 0..8u32 {
        let sc_f = m.convert_u_to_f(t_f32, scales[sb_idx as usize]);
        let mn_f = m.convert_u_to_f(t_f32, mins[sb_idx as usize]);
        let qs_group = sb_idx / 2;
        let is_high = (sb_idx & 1) != 0;
        let qs_base_plane = 4 + qs_group * 8;
        let c_qs_bp = m.constant_u32(t_u32, qs_base_plane);
        let qs_plane_offset = m.imul(t_u32, c_qs_bp, pc_rows);
        let qs_plane_start = m.iadd(t_u32, plane_base, qs_plane_offset);
        let c_sb_offset = m.constant_u32(t_u32, sb_idx * 32);
        let inp_sb_base = m.iadd(t_u32, blk_x_256, c_sb_offset);
        let mut nibble_input_sum = c_f32_0;
        let mut input_sum = c_f32_0;

        for w in 0..8u32 {
            let w_offset = if w == 0 {
                qs_plane_start
            } else {
                let c_w = m.constant_u32(t_u32, w);
                let w_x_rows = m.imul(t_u32, c_w, pc_rows);
                m.iadd(t_u32, qs_plane_start, w_x_rows)
            };
            let qs_addr = m.iadd(t_u32, w_offset, row);
            let qs_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, qs_addr]);
            let qs_word = m.load(t_u32, qs_ptr);
            let inp_w_base = if w == 0 {
                inp_sb_base
            } else {
                let c_w4 = m.constant_u32(t_u32, w * 4);
                m.iadd(t_u32, inp_sb_base, c_w4)
            };
            for byte_idx in 0..4u32 {
                let shift_amt = if is_high {
                    byte_idx * 8 + 4
                } else {
                    byte_idx * 8
                };
                let nibble = if shift_amt == 0 {
                    m.bitwise_and(t_u32, qs_word, c_u32_0f)
                } else {
                    let c_shift = m.constant_u32(t_u32, shift_amt);
                    let shifted = m.shift_right_logical(t_u32, qs_word, c_shift);
                    m.bitwise_and(t_u32, shifted, c_u32_0f)
                };
                let nibble_f = m.convert_u_to_f(t_f32, nibble);
                let inp_idx = if byte_idx == 0 {
                    inp_w_base
                } else {
                    let c_bi = m.constant_u32(t_u32, byte_idx);
                    m.iadd(t_u32, inp_w_base, c_bi)
                };
                let inp_ptr = m.access_chain(t_ptr_sb_f32, gvar_input, &[c_u32_0, inp_idx]);
                let inp_val = m.load(t_f32, inp_ptr);
                let ni_prod = m.fmul(t_f32, nibble_f, inp_val);
                nibble_input_sum = m.fadd(t_f32, nibble_input_sum, ni_prod);
                input_sum = m.fadd(t_f32, input_sum, inp_val);
            }
        }

        let d_sc = m.fmul(t_f32, d_f32, sc_f);
        let term1 = m.fmul(t_f32, d_sc, nibble_input_sum);
        let dmin_mn = m.fmul(t_f32, dmin_f32, mn_f);
        let term2 = m.fmul(t_f32, dmin_mn, input_sum);
        let sb_result = m.fsub(t_f32, term1, term2);
        total_sum = m.fadd(t_f32, total_sum, sb_result);
    }

    let partial_base = m.imul(t_u32, blk_cur, pc_rows);
    let partial_addr = m.iadd(t_u32, partial_base, row);
    let partial_ptr = m.access_chain(t_ptr_sb_f32, gvar_partial, &[c_u32_0, partial_addr]);
    m.store(partial_ptr, total_sum);
    m.branch(lbl_block_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_block_merge.0]));
    m.branch(lbl_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_merge.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// Emit the second pass for block-parallel Q4_K GEMV.
///
/// Binding 0: partial buffer written by `emit_q4k_gemv_block_partial`
/// Binding 1: output buffer
/// Push constants: `{ rows, cols, rows_per_wg }`, where `cols / 256` is the
/// number of partials to reduce for each row.
pub fn emit_q4k_block_reduce(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_arr_f32 = m.type_runtime_array(t_f32);
    let t_struct_partial = m.type_struct(&[t_arr_f32]);
    let t_struct_output = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);
    let t_ptr_sb_partial = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_partial);
    let t_ptr_sb_output = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_output);
    let t_ptr_pc_struct = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_u32 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_256 = m.constant_u32(t_u32, 256);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);

    m.decorate(t_struct_partial, decoration::BLOCK, &[]);
    m.decorate(t_struct_output, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_partial, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_output, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);

    let gvar_partial = m.variable(t_ptr_sb_partial, storage_class::STORAGE_BUFFER);
    let gvar_output = m.variable(t_ptr_sb_output, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc_struct, storage_class::PUSH_CONSTANT);
    let gvar_glob_id = m.variable(t_ptr_input_u32, storage_class::INPUT);

    m.decorate(gvar_partial, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_partial, decoration::BINDING, &[0]);
    m.decorate(gvar_output, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_output, decoration::BINDING, &[1]);
    m.decorate(
        gvar_glob_id,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_glob_id]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    let lbl_entry = m.alloc_id();
    let lbl_bounds_true = m.alloc_id();
    let lbl_bounds_merge = m.alloc_id();
    let lbl_loop_header = m.alloc_id();
    let lbl_loop_cond = m.alloc_id();
    let lbl_loop_body = m.alloc_id();
    let lbl_loop_continue = m.alloc_id();
    let lbl_loop_merge = m.alloc_id();

    m.function(t_void, func_id, 0, t_fn_void);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let var_blk = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);

    let glob_id_vec = m.load(t_v3u32, gvar_glob_id);
    let row = m.composite_extract(t_u32, glob_id_vec, 0);
    let pc_ptr_rows = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let pc_rows = m.load(t_u32, pc_ptr_rows);
    let pc_ptr_cols = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let pc_cols = m.load(t_u32, pc_ptr_cols);
    let num_blocks = m.udiv(t_u32, pc_cols, c_u32_256);

    let in_bounds = m.u_less_than(t_bool, row, pc_rows);
    m.selection_merge(lbl_bounds_merge, 0);
    m.branch_conditional(in_bounds, lbl_bounds_true, lbl_bounds_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_true.0]));
    m.store(var_blk, c_u32_0);
    m.store(var_sum, c_f32_0);
    m.branch(lbl_loop_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_header.0]));
    m.loop_merge(lbl_loop_merge, lbl_loop_continue, 0);
    m.branch(lbl_loop_cond);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_cond.0]));
    let blk_cur = m.load(t_u32, var_blk);
    let keep_going = m.u_less_than(t_bool, blk_cur, num_blocks);
    m.branch_conditional(keep_going, lbl_loop_body, lbl_loop_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_body.0]));
    let partial_base = m.imul(t_u32, blk_cur, pc_rows);
    let partial_addr = m.iadd(t_u32, partial_base, row);
    let partial_ptr = m.access_chain(t_ptr_sb_f32, gvar_partial, &[c_u32_0, partial_addr]);
    let partial_val = m.load(t_f32, partial_ptr);
    let old_sum = m.load(t_f32, var_sum);
    let new_sum = m.fadd(t_f32, old_sum, partial_val);
    m.store(var_sum, new_sum);
    m.branch(lbl_loop_continue);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_continue.0]));
    let blk_next = m.iadd(t_u32, blk_cur, c_u32_1);
    m.store(var_blk, blk_next);
    m.branch(lbl_loop_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_merge.0]));
    let final_sum = m.load(t_f32, var_sum);
    let out_ptr = m.access_chain(t_ptr_sb_f32, gvar_output, &[c_u32_0, row]);
    m.store(out_ptr, final_sum);
    m.branch(lbl_bounds_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_merge.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// Emit a single-dispatch Q4_K GEMV shader where one workgroup computes one row.
///
/// Dispatch shape: `(rows, 1, 1)`, local size: `local_size_x`.
/// Each lane walks the column dimension with `i += local_size_x`, accumulates
/// a partial sum, then the workgroup reduces to `output[row]`.
pub fn emit_q4k_gemv_wg_reduce(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_arr_u32 = m.type_runtime_array(t_u32);
    let t_arr_f32 = m.type_runtime_array(t_f32);
    let t_struct_weight = m.type_struct(&[t_arr_u32]);
    let t_struct_input = m.type_struct(&[t_arr_f32]);
    let t_struct_output = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);
    let c_local_size = m.constant_u32(t_u32, local_size_x);
    let t_shared_arr = m.type_array(t_f32, c_local_size);

    let t_ptr_sb_weight = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_weight);
    let t_ptr_sb_input = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_input);
    let t_ptr_sb_output = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_output);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);
    let t_ptr_wg_arr = m.type_pointer(storage_class::WORKGROUP, t_shared_arr);
    let t_ptr_wg_f32 = m.type_pointer(storage_class::WORKGROUP, t_f32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_4 = m.constant_u32(t_u32, 4);
    let c_u32_6 = m.constant_u32(t_u32, 6);
    let c_u32_8 = m.constant_u32(t_u32, 8);
    let c_u32_10 = m.constant_u32(t_u32, 10);
    let c_u32_13 = m.constant_u32(t_u32, 13);
    let c_u32_15 = m.constant_u32(t_u32, 15);
    let c_u32_16 = m.constant_u32(t_u32, 16);
    let c_u32_23 = m.constant_u32(t_u32, 23);
    let c_u32_31 = m.constant_u32(t_u32, 31);
    let c_u32_32 = m.constant_u32(t_u32, 32);
    let c_u32_36 = m.constant_u32(t_u32, 36);
    let c_u32_0f = m.constant_u32(t_u32, 0x0F);
    let c_u32_3f = m.constant_u32(t_u32, 0x3F);
    let c_u32_ff = m.constant_u32(t_u32, 0xFF);
    let c_u32_ffff = m.constant_u32(t_u32, 0xFFFF);
    let c_u32_1f = m.constant_u32(t_u32, 0x1F);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3FF);
    let c_u32_112 = m.constant_u32(t_u32, 112);
    let c_u32_256 = m.constant_u32(t_u32, 256);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);
    let c_f32_2pow_neg24 = m.constant_f32(t_f32, 5.9604644775390625e-8);

    m.decorate(t_struct_weight, decoration::BLOCK, &[]);
    m.decorate(t_struct_input, decoration::BLOCK, &[]);
    m.decorate(t_struct_output, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_weight, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_input, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_output, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_u32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);

    let gvar_weight = m.variable(t_ptr_sb_weight, storage_class::STORAGE_BUFFER);
    let gvar_input = m.variable(t_ptr_sb_input, storage_class::STORAGE_BUFFER);
    let gvar_output = m.variable(t_ptr_sb_output, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let gvar_lid = m.variable(t_ptr_input_v3, storage_class::INPUT);
    let gvar_wgid = m.variable(t_ptr_input_v3, storage_class::INPUT);
    let gvar_shared = m.variable(t_ptr_wg_arr, storage_class::WORKGROUP);

    m.decorate(gvar_weight, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_weight, decoration::BINDING, &[0]);
    m.decorate(gvar_input, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_input, decoration::BINDING, &[1]);
    m.decorate(gvar_output, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_output, decoration::BINDING, &[2]);
    m.decorate(
        gvar_lid,
        decoration::BUILTIN,
        &[builtin::LOCAL_INVOCATION_ID],
    );
    m.decorate(gvar_wgid, decoration::BUILTIN, &[builtin::WORKGROUP_ID]);

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_lid, gvar_wgid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    let lbl_entry = m.alloc_id();
    let lbl_bounds_true = m.alloc_id();
    let lbl_bounds_merge = m.alloc_id();
    let lbl_loop_header = m.alloc_id();
    let lbl_loop_cond = m.alloc_id();
    let lbl_loop_body = m.alloc_id();
    let lbl_loop_continue = m.alloc_id();
    let lbl_loop_merge = m.alloc_id();

    m.function(t_void, func_id, 0, t_fn_void);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let var_i = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_s = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);

    let lid_vec = m.load(t_v3u32, gvar_lid);
    let lid = m.composite_extract(t_u32, lid_vec, 0);
    let wgid_vec = m.load(t_v3u32, gvar_wgid);
    let row = m.composite_extract(t_u32, wgid_vec, 0);
    let pc_ptr_rows = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let pc_rows = m.load(t_u32, pc_ptr_rows);
    let pc_ptr_cols = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let pc_cols = m.load(t_u32, pc_ptr_cols);

    let row_in_bounds = m.u_less_than(t_bool, row, pc_rows);
    m.selection_merge(lbl_bounds_merge, 0);
    m.branch_conditional(row_in_bounds, lbl_bounds_true, lbl_bounds_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_true.0]));
    m.store(var_i, lid);
    m.store(var_sum, c_f32_0);
    m.branch(lbl_loop_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_header.0]));
    m.loop_merge(lbl_loop_merge, lbl_loop_continue, 0);
    m.branch(lbl_loop_cond);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_cond.0]));
    let iv = m.load(t_u32, var_i);
    let cond = m.u_less_than(t_bool, iv, pc_cols);
    m.branch_conditional(cond, lbl_loop_body, lbl_loop_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_body.0]));

    let blk_cur = m.udiv(t_u32, iv, c_u32_256);
    let rem = m.umod(t_u32, iv, c_u32_256);
    let sb_idx = m.udiv(t_u32, rem, c_u32_32);
    let elem_in_sb = m.umod(t_u32, rem, c_u32_32);
    let qword = m.udiv(t_u32, elem_in_sb, c_u32_4);
    let byte_idx = m.umod(t_u32, elem_in_sb, c_u32_4);

    let blk_x_36 = m.imul(t_u32, blk_cur, c_u32_36);
    let plane_base = m.imul(t_u32, blk_x_36, pc_rows);
    let packed_addr = m.iadd(t_u32, plane_base, row);
    let packed_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, packed_addr]);
    let packed_word = m.load(t_u32, packed_ptr);

    let d_f16_raw = m.bitwise_and(t_u32, packed_word, c_u32_ffff);
    let d_f32 = {
        let exp_raw = m.shift_right_logical(t_u32, d_f16_raw, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, d_f16_raw, c_u32_3ff);
        let sign = m.shift_right_logical(t_u32, d_f16_raw, c_u32_15);
        let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
        let s_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
        let e_adj = m.iadd(t_u32, exp, c_u32_112);
        let e_part = m.shift_left_logical(t_u32, e_adj, c_u32_23);
        let m_part = m.shift_left_logical(t_u32, mant, c_u32_13);
        let bits_mid = m.bitwise_or(t_u32, s_part, e_part);
        let bits = m.bitwise_or(t_u32, bits_mid, m_part);
        let normal = m.bitcast(t_f32, bits);
        let mant_f = m.convert_u_to_f(t_f32, mant);
        let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
        let denorm_neg = m.fnegate(t_f32, denorm_abs);
        let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
        let denormal = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
        let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
        m.select(t_f32, exp_nonzero, normal, denormal)
    };
    let dmin_f16_raw = m.shift_right_logical(t_u32, packed_word, c_u32_16);
    let dmin_f32 = {
        let exp_raw = m.shift_right_logical(t_u32, dmin_f16_raw, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, dmin_f16_raw, c_u32_3ff);
        let sign = m.shift_right_logical(t_u32, dmin_f16_raw, c_u32_15);
        let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
        let s_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
        let e_adj = m.iadd(t_u32, exp, c_u32_112);
        let e_part = m.shift_left_logical(t_u32, e_adj, c_u32_23);
        let m_part = m.shift_left_logical(t_u32, mant, c_u32_13);
        let bits_mid = m.bitwise_or(t_u32, s_part, e_part);
        let bits = m.bitwise_or(t_u32, bits_mid, m_part);
        let normal = m.bitcast(t_f32, bits);
        let mant_f = m.convert_u_to_f(t_f32, mant);
        let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
        let denorm_neg = m.fnegate(t_f32, denorm_abs);
        let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
        let denormal = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
        let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
        m.select(t_f32, exp_nonzero, normal, denormal)
    };

    let s0_addr = m.iadd(t_u32, plane_base, pc_rows);
    let s0_addr = m.iadd(t_u32, s0_addr, row);
    let s0_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, s0_addr]);
    let s0_word = m.load(t_u32, s0_ptr);
    let two_rows = m.imul(t_u32, c_u32_2, pc_rows);
    let s1_offset = m.iadd(t_u32, plane_base, two_rows);
    let s1_addr = m.iadd(t_u32, s1_offset, row);
    let s1_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, s1_addr]);
    let s1_word = m.load(t_u32, s1_ptr);
    let three_rows = m.imul(t_u32, c_u32_3, pc_rows);
    let s2_offset = m.iadd(t_u32, plane_base, three_rows);
    let s2_addr = m.iadd(t_u32, s2_offset, row);
    let s2_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, s2_addr]);
    let s2_word = m.load(t_u32, s2_ptr);

    let is_low_scale = m.u_less_than(t_bool, sb_idx, c_u32_4);
    let sb_hi_raw = m.isub(t_u32, sb_idx, c_u32_4);
    let sb_low_idx = m.select(t_u32, is_low_scale, sb_idx, c_u32_0);
    let sb_hi_idx = m.select(t_u32, is_low_scale, c_u32_0, sb_hi_raw);
    let shift_low = m.imul(t_u32, sb_low_idx, c_u32_8);
    let shift_hi = m.imul(t_u32, sb_hi_idx, c_u32_8);

    let sc_lo_byte = {
        let shifted = m.shift_right_logical(t_u32, s0_word, shift_low);
        m.bitwise_and(t_u32, shifted, c_u32_ff)
    };
    let mn_lo_byte = {
        let shifted = m.shift_right_logical(t_u32, s1_word, shift_low);
        m.bitwise_and(t_u32, shifted, c_u32_ff)
    };
    let s2_byte = {
        let shifted = m.shift_right_logical(t_u32, s2_word, shift_hi);
        m.bitwise_and(t_u32, shifted, c_u32_ff)
    };
    let s0_hi_byte = {
        let shifted = m.shift_right_logical(t_u32, s0_word, shift_hi);
        m.bitwise_and(t_u32, shifted, c_u32_ff)
    };
    let s1_hi_byte = {
        let shifted = m.shift_right_logical(t_u32, s1_word, shift_hi);
        m.bitwise_and(t_u32, shifted, c_u32_ff)
    };
    let sc_lo = m.bitwise_and(t_u32, sc_lo_byte, c_u32_3f);
    let mn_lo = m.bitwise_and(t_u32, mn_lo_byte, c_u32_3f);
    let sc_hi_lo = m.bitwise_and(t_u32, s2_byte, c_u32_0f);
    let sc_hi_hi_raw = m.shift_right_logical(t_u32, s0_hi_byte, c_u32_6);
    let sc_hi_hi = m.shift_left_logical(t_u32, sc_hi_hi_raw, c_u32_4);
    let sc_hi = m.bitwise_or(t_u32, sc_hi_lo, sc_hi_hi);
    let mn_hi_lo = m.shift_right_logical(t_u32, s2_byte, c_u32_4);
    let mn_hi_hi_raw = m.shift_right_logical(t_u32, s1_hi_byte, c_u32_6);
    let mn_hi_hi = m.shift_left_logical(t_u32, mn_hi_hi_raw, c_u32_4);
    let mn_hi = m.bitwise_or(t_u32, mn_hi_lo, mn_hi_hi);
    let sc_u = m.select(t_u32, is_low_scale, sc_lo, sc_hi);
    let mn_u = m.select(t_u32, is_low_scale, mn_lo, mn_hi);

    let qs_group = m.udiv(t_u32, sb_idx, c_u32_2);
    let qs_group_x8 = m.imul(t_u32, qs_group, c_u32_8);
    let qs_plane = m.iadd(t_u32, c_u32_4, qs_group_x8);
    let qs_plane = m.iadd(t_u32, qs_plane, qword);
    let qs_plane_offset = m.imul(t_u32, qs_plane, pc_rows);
    let qs_addr = m.iadd(t_u32, plane_base, qs_plane_offset);
    let qs_addr = m.iadd(t_u32, qs_addr, row);
    let qs_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, qs_addr]);
    let qs_word = m.load(t_u32, qs_ptr);

    let sb_mod2 = m.umod(t_u32, sb_idx, c_u32_2);
    let is_high = m.i_not_equal(t_bool, sb_mod2, c_u32_0);
    let high_add = m.select(t_u32, is_high, c_u32_4, c_u32_0);
    let shift_byte = m.imul(t_u32, byte_idx, c_u32_8);
    let shift_nibble = m.iadd(t_u32, shift_byte, high_add);
    let shifted_qs = m.shift_right_logical(t_u32, qs_word, shift_nibble);
    let nibble = m.bitwise_and(t_u32, shifted_qs, c_u32_0f);

    let input_ptr = m.access_chain(t_ptr_sb_f32, gvar_input, &[c_u32_0, iv]);
    let input_val = m.load(t_f32, input_ptr);
    let sc_f = m.convert_u_to_f(t_f32, sc_u);
    let mn_f = m.convert_u_to_f(t_f32, mn_u);
    let nibble_f = m.convert_u_to_f(t_f32, nibble);
    let d_sc = m.fmul(t_f32, d_f32, sc_f);
    let weighted_nibble = m.fmul(t_f32, d_sc, nibble_f);
    let dmin_mn = m.fmul(t_f32, dmin_f32, mn_f);
    let deq = m.fsub(t_f32, weighted_nibble, dmin_mn);
    let contrib = m.fmul(t_f32, deq, input_val);
    let old_sum = m.load(t_f32, var_sum);
    let new_sum = m.fadd(t_f32, old_sum, contrib);
    m.store(var_sum, new_sum);
    m.branch(lbl_loop_continue);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_continue.0]));
    let next_i = m.iadd(t_u32, iv, c_local_size);
    m.store(var_i, next_i);
    m.branch(lbl_loop_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_merge.0]));
    let shared_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared, &[lid]);
    let lane_sum = m.load(t_f32, var_sum);
    m.store(shared_ptr, lane_sum);
    {
        let c_scope_wg = m.constant_u32(t_u32, scope::WORKGROUP);
        let c_sem = m.constant_u32(
            t_u32,
            memory_semantics::WORKGROUP_MEMORY | memory_semantics::ACQUIRE_RELEASE,
        );
        m.control_barrier(c_scope_wg, c_scope_wg, c_sem);
    }

    let c_half = m.constant_u32(t_u32, local_size_x / 2);
    m.store(var_s, c_half);
    let lbl_r_h = m.alloc_id();
    let lbl_r_c = m.alloc_id();
    let lbl_r_b = m.alloc_id();
    let lbl_r_cont = m.alloc_id();
    let lbl_r_m = m.alloc_id();
    m.branch(lbl_r_h);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_h.0]));
    m.loop_merge(lbl_r_m, lbl_r_cont, 0);
    m.branch(lbl_r_c);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_c.0]));
    let sv = m.load(t_u32, var_s);
    let sg = m.u_less_than(t_bool, c_u32_0, sv);
    m.branch_conditional(sg, lbl_r_b, lbl_r_m);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_b.0]));
    let lane_active = m.u_less_than(t_bool, lid, sv);
    let lbl_add = m.alloc_id();
    let lbl_add_merge = m.alloc_id();
    m.selection_merge(lbl_add_merge, 0);
    m.branch_conditional(lane_active, lbl_add, lbl_add_merge);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_add.0]));
    let rhs_idx = m.iadd(t_u32, lid, sv);
    let lhs_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared, &[lid]);
    let rhs_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared, &[rhs_idx]);
    let lhs = m.load(t_f32, lhs_ptr);
    let rhs = m.load(t_f32, rhs_ptr);
    let reduced = m.fadd(t_f32, lhs, rhs);
    m.store(lhs_ptr, reduced);
    m.branch(lbl_add_merge);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_add_merge.0]));
    {
        let c_scope_wg = m.constant_u32(t_u32, scope::WORKGROUP);
        let c_sem = m.constant_u32(
            t_u32,
            memory_semantics::WORKGROUP_MEMORY | memory_semantics::ACQUIRE_RELEASE,
        );
        m.control_barrier(c_scope_wg, c_scope_wg, c_sem);
    }
    m.branch(lbl_r_cont);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_cont.0]));
    let sv_next = m.shift_right_logical(t_u32, sv, c_u32_1);
    m.store(var_s, sv_next);
    m.branch(lbl_r_h);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_m.0]));
    let is_lane_zero = m.u_less_than(t_bool, lid, c_u32_1);
    let lbl_store = m.alloc_id();
    let lbl_store_merge = m.alloc_id();
    m.selection_merge(lbl_store_merge, 0);
    m.branch_conditional(is_lane_zero, lbl_store, lbl_store_merge);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_store.0]));
    let s0_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared, &[c_u32_0]);
    let final_sum = m.load(t_f32, s0_ptr);
    let out_ptr = m.access_chain(t_ptr_sb_f32, gvar_output, &[c_u32_0, row]);
    m.store(out_ptr, final_sum);
    m.branch(lbl_store_merge);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_store_merge.0]));
    m.branch(lbl_bounds_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_merge.0]));
    m.ret();
    m.function_end();

    m.encode()
}

pub fn emit_q5k_gemv(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    // Capabilities & memory model
    m.capability(1); // Shader
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1); // Logical, GLSL450

    // --- Types ---
    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let _t_i32 = m.type_int(32, 1);
    let t_f32 = m.type_float(32);

    let t_v3u32 = m.type_vector(t_u32, 3);

    let t_arr_u32 = m.type_runtime_array(t_u32);
    let t_arr_f32 = m.type_runtime_array(t_f32);

    let t_struct_weight = m.type_struct(&[t_arr_u32]);
    let t_struct_input = m.type_struct(&[t_arr_f32]);
    let t_struct_output = m.type_struct(&[t_arr_f32]);

    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);

    // Pointer types
    let t_ptr_sb_struct_weight = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_weight);
    let t_ptr_sb_struct_input = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_input);
    let t_ptr_sb_struct_output = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_output);
    let t_ptr_pc_struct = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_u32 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);

    let t_fn_void = m.type_function(t_void, &[]);

    // --- Constants ---
    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_4 = m.constant_u32(t_u32, 4);
    let c_u32_6 = m.constant_u32(t_u32, 6);
    let c_u32_10 = m.constant_u32(t_u32, 10);
    let c_u32_13 = m.constant_u32(t_u32, 13);
    let c_u32_15 = m.constant_u32(t_u32, 15);
    let c_u32_16 = m.constant_u32(t_u32, 16);
    let c_u32_23 = m.constant_u32(t_u32, 23);
    let c_u32_31 = m.constant_u32(t_u32, 31);
    let c_u32_44 = m.constant_u32(t_u32, 44);
    let c_u32_ff = m.constant_u32(t_u32, 0xFF);
    let c_u32_0f = m.constant_u32(t_u32, 0x0F);
    let c_u32_3f = m.constant_u32(t_u32, 0x3F);
    let c_u32_ffff = m.constant_u32(t_u32, 0xFFFF);
    let c_u32_1f = m.constant_u32(t_u32, 0x1F);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3FF);
    let c_u32_112 = m.constant_u32(t_u32, 112);
    let c_u32_256 = m.constant_u32(t_u32, 256);

    let c_f32_0 = m.constant_f32(t_f32, 0.0);

    // --- Decorations ---
    m.decorate(t_struct_weight, decoration::BLOCK, &[]);
    m.decorate(t_struct_input, decoration::BLOCK, &[]);
    m.decorate(t_struct_output, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_weight, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_input, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_output, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_u32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);

    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);

    // --- Global variables ---
    let gvar_weight = m.variable(t_ptr_sb_struct_weight, storage_class::STORAGE_BUFFER);
    let gvar_input = m.variable(t_ptr_sb_struct_input, storage_class::STORAGE_BUFFER);
    let gvar_output = m.variable(t_ptr_sb_struct_output, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc_struct, storage_class::PUSH_CONSTANT);
    let gvar_glob_id = m.variable(t_ptr_input_u32, storage_class::INPUT);

    m.decorate(gvar_weight, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_weight, decoration::BINDING, &[0]);
    m.decorate(gvar_input, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_input, decoration::BINDING, &[1]);
    m.decorate(gvar_output, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_output, decoration::BINDING, &[2]);
    m.decorate(
        gvar_glob_id,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    // --- Entry point ---
    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_glob_id]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    // Pre-allocate labels
    let lbl_entry = m.alloc_id();
    let lbl_bounds_merge = m.alloc_id();
    let lbl_bounds_true = m.alloc_id();
    let lbl_outer_header = m.alloc_id();
    let lbl_outer_body = m.alloc_id();
    let lbl_outer_continue = m.alloc_id();
    let lbl_outer_merge = m.alloc_id();

    // --- Function ---
    m.function(t_void, func_id, 0, t_fn_void);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    // Function variables
    let var_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_blk = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);

    // Load GlobalInvocationID.x → row
    let glob_id_vec = m.load(t_v3u32, gvar_glob_id);
    let row = m.composite_extract(t_u32, glob_id_vec, 0);

    // Load push constants: rows, cols
    let pc_ptr_rows = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let pc_rows = m.load(t_u32, pc_ptr_rows);
    let pc_ptr_cols = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let pc_cols = m.load(t_u32, pc_ptr_cols);

    // Bounds check: row < rows
    let in_bounds = m.u_less_than(t_bool, row, pc_rows);
    m.selection_merge(lbl_bounds_merge, 0);
    m.branch_conditional(in_bounds, lbl_bounds_true, lbl_bounds_merge);

    // --- In-bounds block ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_true.0]));

    m.store(var_sum, c_f32_0);

    // num_blocks = cols / 256
    let num_blocks = m.udiv(t_u32, pc_cols, c_u32_256);

    m.store(var_blk, c_u32_0);
    m.branch(lbl_outer_header);

    // --- Outer loop header (block loop) ---
    let lbl_outer_cond = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_header.0]));
    m.loop_merge(lbl_outer_merge, lbl_outer_continue, 0);
    m.branch(lbl_outer_cond);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_cond.0]));
    let blk_cur = m.load(t_u32, var_blk);
    let outer_cond = m.u_less_than(t_bool, blk_cur, num_blocks);
    m.branch_conditional(outer_cond, lbl_outer_body, lbl_outer_merge);

    // --- Outer loop body ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_body.0]));

    let blk_x_44 = m.imul(t_u32, blk_cur, c_u32_44);
    let plane_base = m.imul(t_u32, blk_x_44, pc_rows);

    // ============================================================
    // Load d and dmin from plane 0
    // packed = weight[plane_base + row]
    // d    = f16_to_f32(packed & 0xFFFF)
    // dmin = f16_to_f32(packed >> 16)
    // ============================================================
    let packed_addr = m.iadd(t_u32, plane_base, row);
    let packed_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, packed_addr]);
    let packed_word = m.load(t_u32, packed_ptr);

    // f16→f32 helper constant: 2^(-24) = 5.9604644775390625e-8
    let c_f32_2pow_neg24 = m.constant_f32(t_f32, 5.9604644775390625e-8);

    // d = f16_to_f32(packed & 0xFFFF) with denormal + sign handling
    let d_f16_raw = m.bitwise_and(t_u32, packed_word, c_u32_ffff);
    let d_f32 = {
        let exp_raw = m.shift_right_logical(t_u32, d_f16_raw, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, d_f16_raw, c_u32_3ff);
        let sign = m.shift_right_logical(t_u32, d_f16_raw, c_u32_15);
        let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
        let s_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
        let e_adj = m.iadd(t_u32, exp, c_u32_112);
        let e_part = m.shift_left_logical(t_u32, e_adj, c_u32_23);
        let m_part = m.shift_left_logical(t_u32, mant, c_u32_13);
        let bits_mid = m.bitwise_or(t_u32, s_part, e_part);
        let bits = m.bitwise_or(t_u32, bits_mid, m_part);
        let normal = m.bitcast(t_f32, bits);
        let mant_f = m.convert_u_to_f(t_f32, mant);
        let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
        let denorm_neg = m.fnegate(t_f32, denorm_abs);
        let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
        let denormal = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
        let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
        m.select(t_f32, exp_nonzero, normal, denormal)
    };

    let dmin_f16_raw = m.shift_right_logical(t_u32, packed_word, c_u32_16);
    let dmin_f32 = {
        let exp_raw = m.shift_right_logical(t_u32, dmin_f16_raw, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, dmin_f16_raw, c_u32_3ff);
        let sign = m.shift_right_logical(t_u32, dmin_f16_raw, c_u32_15);
        let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
        let s_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
        let e_adj = m.iadd(t_u32, exp, c_u32_112);
        let e_part = m.shift_left_logical(t_u32, e_adj, c_u32_23);
        let m_part = m.shift_left_logical(t_u32, mant, c_u32_13);
        let bits_mid = m.bitwise_or(t_u32, s_part, e_part);
        let bits = m.bitwise_or(t_u32, bits_mid, m_part);
        let normal = m.bitcast(t_f32, bits);
        let mant_f = m.convert_u_to_f(t_f32, mant);
        let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
        let denorm_neg = m.fnegate(t_f32, denorm_abs);
        let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
        let denormal = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
        let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
        m.select(t_f32, exp_nonzero, normal, denormal)
    };

    // ============================================================
    // Load scales bytes from planes 1-3
    // ============================================================
    let s0_addr = m.iadd(t_u32, plane_base, pc_rows);
    let s0_addr = m.iadd(t_u32, s0_addr, row);
    let s0_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, s0_addr]);
    let s0_word = m.load(t_u32, s0_ptr);

    let s1_offset = {
        let two_rows = m.imul(t_u32, c_u32_2, pc_rows);
        m.iadd(t_u32, plane_base, two_rows)
    };
    let s1_addr = m.iadd(t_u32, s1_offset, row);
    let s1_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, s1_addr]);
    let s1_word = m.load(t_u32, s1_ptr);

    let s2_offset = {
        let three_rows = m.imul(t_u32, c_u32_3, pc_rows);
        m.iadd(t_u32, plane_base, three_rows)
    };
    let s2_addr = m.iadd(t_u32, s2_offset, row);
    let s2_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, s2_addr]);
    let s2_word = m.load(t_u32, s2_ptr);

    // Extract 12 individual bytes: sb[0..11]
    let mut sb = [c_u32_0; 12];
    for i in 0..4u32 {
        if i == 0 {
            sb[i as usize] = m.bitwise_and(t_u32, s0_word, c_u32_ff);
            sb[4 + i as usize] = m.bitwise_and(t_u32, s1_word, c_u32_ff);
            sb[8 + i as usize] = m.bitwise_and(t_u32, s2_word, c_u32_ff);
        } else {
            let shift = m.constant_u32(t_u32, i * 8);
            let s0_shifted = m.shift_right_logical(t_u32, s0_word, shift);
            let s1_shifted = m.shift_right_logical(t_u32, s1_word, shift);
            let s2_shifted = m.shift_right_logical(t_u32, s2_word, shift);
            sb[i as usize] = m.bitwise_and(t_u32, s0_shifted, c_u32_ff);
            sb[4 + i as usize] = m.bitwise_and(t_u32, s1_shifted, c_u32_ff);
            sb[8 + i as usize] = m.bitwise_and(t_u32, s2_shifted, c_u32_ff);
        }
    }

    // Extract 8 scales and 8 mins using 6-bit packing
    let mut scales = [c_u32_0; 8];
    let mut mins = [c_u32_0; 8];

    for j in 0..4usize {
        scales[j] = m.bitwise_and(t_u32, sb[j], c_u32_3f);
        mins[j] = m.bitwise_and(t_u32, sb[j + 4], c_u32_3f);
    }

    for j in 4..8usize {
        let lo = m.bitwise_and(t_u32, sb[j + 4], c_u32_0f);
        let hi_raw = m.shift_right_logical(t_u32, sb[j - 4], c_u32_6);
        let hi = m.shift_left_logical(t_u32, hi_raw, c_u32_4);
        scales[j] = m.bitwise_or(t_u32, lo, hi);

        let lo2 = m.shift_right_logical(t_u32, sb[j + 4], c_u32_4);
        let hi2_raw = m.shift_right_logical(t_u32, sb[j], c_u32_6);
        let hi2 = m.shift_left_logical(t_u32, hi2_raw, c_u32_4);
        mins[j] = m.bitwise_or(t_u32, lo2, hi2);
    }

    let blk_x_256 = m.imul(t_u32, blk_cur, c_u32_256);
    let mut total_sum = c_f32_0;
    let mut qh_words = [c_u32_0; 8];
    for w in 0..8u32 {
        let c_qh_plane = m.constant_u32(t_u32, 36 + w);
        let qh_plane_offset = m.imul(t_u32, c_qh_plane, pc_rows);
        let qh_plane_start = m.iadd(t_u32, plane_base, qh_plane_offset);
        let qh_addr = m.iadd(t_u32, qh_plane_start, row);
        let qh_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, qh_addr]);
        qh_words[w as usize] = m.load(t_u32, qh_ptr);
    }

    for qs_group in 0..4u32 {
        let sb_lo = (qs_group * 2) as usize;
        let sb_hi = sb_lo + 1;

        let sc_lo_f = {
            let sc_u = scales[sb_lo];
            m.convert_u_to_f(t_f32, sc_u)
        };
        let mn_lo_f = {
            let mn_u = mins[sb_lo];
            m.convert_u_to_f(t_f32, mn_u)
        };
        let sc_hi_f = {
            let sc_u = scales[sb_hi];
            m.convert_u_to_f(t_f32, sc_u)
        };
        let mn_hi_f = {
            let mn_u = mins[sb_hi];
            m.convert_u_to_f(t_f32, mn_u)
        };

        let qs_base_plane = 4 + qs_group * 8;

        let c_qs_bp = m.constant_u32(t_u32, qs_base_plane);
        let qs_plane_offset = m.imul(t_u32, c_qs_bp, pc_rows);
        let qs_plane_start = m.iadd(t_u32, plane_base, qs_plane_offset);

        // input base for this sub-block: blk*256 + sb*32
        let c_sb_offset = m.constant_u32(t_u32, qs_group * 64);
        let inp_sb_base = m.iadd(t_u32, blk_x_256, c_sb_offset);

        let mut q_input_sum_lo = c_f32_0;
        let mut q_input_sum_hi = c_f32_0;
        let mut input_sum = c_f32_0;

        for w in 0..8u32 {
            // Load qs word: weight[(qs_plane_start + w * rows) + row]
            let w_offset = if w == 0 {
                qs_plane_start
            } else {
                let c_w = m.constant_u32(t_u32, w);
                let w_x_rows = m.imul(t_u32, c_w, pc_rows);
                m.iadd(t_u32, qs_plane_start, w_x_rows)
            };
            let qs_addr = m.iadd(t_u32, w_offset, row);
            let qs_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, qs_addr]);
            let qs_word = m.load(t_u32, qs_ptr);
            let qh_word = qh_words[w as usize];

            // Input base for this word: inp_sb_base + w*4
            let inp_w_base = if w == 0 {
                inp_sb_base
            } else {
                let c_w4 = m.constant_u32(t_u32, w * 4);
                m.iadd(t_u32, inp_sb_base, c_w4)
            };

            // Process 4 bytes from this u32
            for byte_idx in 0..4u32 {
                let lo_shift_amt = byte_idx * 8;
                let nibble_lo = if lo_shift_amt == 0 {
                    m.bitwise_and(t_u32, qs_word, c_u32_0f)
                } else {
                    let c_shift = m.constant_u32(t_u32, lo_shift_amt);
                    let shifted = m.shift_right_logical(t_u32, qs_word, c_shift);
                    m.bitwise_and(t_u32, shifted, c_u32_0f)
                };
                let hi_shift_amt = byte_idx * 8 + 4;
                let nibble_hi = {
                    let c_shift = m.constant_u32(t_u32, hi_shift_amt);
                    let shifted = m.shift_right_logical(t_u32, qs_word, c_shift);
                    m.bitwise_and(t_u32, shifted, c_u32_0f)
                };

                let qh_shift_lo = byte_idx * 8 + qs_group * 2;
                let high_bit_lo = if qh_shift_lo == 0 {
                    m.bitwise_and(t_u32, qh_word, c_u32_1)
                } else {
                    let c_shift = m.constant_u32(t_u32, qh_shift_lo);
                    let shifted = m.shift_right_logical(t_u32, qh_word, c_shift);
                    m.bitwise_and(t_u32, shifted, c_u32_1)
                };
                let qh_shift_hi = qh_shift_lo + 1;
                let high_bit_hi = {
                    let c_shift = m.constant_u32(t_u32, qh_shift_hi);
                    let shifted = m.shift_right_logical(t_u32, qh_word, c_shift);
                    m.bitwise_and(t_u32, shifted, c_u32_1)
                };
                let high_lo = m.shift_left_logical(t_u32, high_bit_lo, c_u32_4);
                let high_hi = m.shift_left_logical(t_u32, high_bit_hi, c_u32_4);

                let q5_lo = m.bitwise_or(t_u32, nibble_lo, high_lo);
                let q5_hi = m.bitwise_or(t_u32, nibble_hi, high_hi);
                let q5_lo_f = m.convert_u_to_f(t_f32, q5_lo);
                let q5_hi_f = m.convert_u_to_f(t_f32, q5_hi);

                // Load input[inp_w_base + byte_idx]
                let inp_idx = if byte_idx == 0 {
                    inp_w_base
                } else {
                    let c_bi = m.constant_u32(t_u32, byte_idx);
                    m.iadd(t_u32, inp_w_base, c_bi)
                };
                let inp_ptr = m.access_chain(t_ptr_sb_f32, gvar_input, &[c_u32_0, inp_idx]);
                let inp_val = m.load(t_f32, inp_ptr);

                let qi_prod_lo = m.fmul(t_f32, q5_lo_f, inp_val);
                let qi_prod_hi = m.fmul(t_f32, q5_hi_f, inp_val);
                q_input_sum_lo = m.fadd(t_f32, q_input_sum_lo, qi_prod_lo);
                q_input_sum_hi = m.fadd(t_f32, q_input_sum_hi, qi_prod_hi);

                // input_sum += input
                input_sum = m.fadd(t_f32, input_sum, inp_val);
            }
        }

        let d_sc_lo = m.fmul(t_f32, d_f32, sc_lo_f);
        let term1_lo = m.fmul(t_f32, d_sc_lo, q_input_sum_lo);
        let dmin_mn_lo = m.fmul(t_f32, dmin_f32, mn_lo_f);
        let term2_lo = m.fmul(t_f32, dmin_mn_lo, input_sum);
        let sb_result_lo = m.fsub(t_f32, term1_lo, term2_lo);
        total_sum = m.fadd(t_f32, total_sum, sb_result_lo);

        let d_sc_hi = m.fmul(t_f32, d_f32, sc_hi_f);
        let term1_hi = m.fmul(t_f32, d_sc_hi, q_input_sum_hi);
        let dmin_mn_hi = m.fmul(t_f32, dmin_f32, mn_hi_f);
        let term2_hi = m.fmul(t_f32, dmin_mn_hi, input_sum);
        let sb_result_hi = m.fsub(t_f32, term1_hi, term2_hi);
        total_sum = m.fadd(t_f32, total_sum, sb_result_hi);
    }

    // Accumulate block result
    let prev_total = m.load(t_f32, var_sum);
    let new_total = m.fadd(t_f32, prev_total, total_sum);
    m.store(var_sum, new_total);
    m.branch(lbl_outer_continue);

    // --- Outer loop continue ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_continue.0]));
    let blk_next = m.iadd(t_u32, blk_cur, c_u32_1);
    m.store(var_blk, blk_next);
    m.branch(lbl_outer_header);

    // --- Outer loop merge ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_merge.0]));

    let final_sum = m.load(t_f32, var_sum);
    let out_ptr = m.access_chain(t_ptr_sb_f32, gvar_output, &[c_u32_0, row]);
    m.store(out_ptr, final_sum);
    m.branch(lbl_bounds_merge);

    // --- Bounds merge ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_merge.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// Emit a complete SPIR-V compute shader for **Q4_K GEMV (row-major layout)**.
///
/// Same dequantization logic as `emit_q4k_gemv`, but reads weights in their
/// original GGUF row-major layout (144 bytes per block = 36 u32 words):
///   weight[row * num_blocks * 36 + block * 36 + plane]
///
/// This avoids the SoA repack step, enabling zero-copy from mmap at the cost
/// of non-coalesced GPU memory access.
pub fn emit_q4k_gemv_rowmajor(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1); // Shader
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1); // Logical, GLSL450

    // --- Types ---
    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let _t_i32 = m.type_int(32, 1);
    let t_f32 = m.type_float(32);

    let t_v3u32 = m.type_vector(t_u32, 3);

    let t_arr_u32 = m.type_runtime_array(t_u32);
    let t_arr_f32 = m.type_runtime_array(t_f32);

    let t_struct_weight = m.type_struct(&[t_arr_u32]);
    let t_struct_input = m.type_struct(&[t_arr_f32]);
    let t_struct_output = m.type_struct(&[t_arr_f32]);

    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);

    let t_ptr_sb_struct_weight = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_weight);
    let t_ptr_sb_struct_input = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_input);
    let t_ptr_sb_struct_output = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_output);
    let t_ptr_pc_struct = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_u32 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);

    let t_fn_void = m.type_function(t_void, &[]);

    // --- Constants ---
    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_4 = m.constant_u32(t_u32, 4);
    let c_u32_10 = m.constant_u32(t_u32, 10);
    let c_u32_13 = m.constant_u32(t_u32, 13);
    let c_u32_15 = m.constant_u32(t_u32, 15);
    let c_u32_16 = m.constant_u32(t_u32, 16);
    let c_u32_23 = m.constant_u32(t_u32, 23);
    let c_u32_31 = m.constant_u32(t_u32, 31);
    let c_u32_36 = m.constant_u32(t_u32, 36);
    let c_u32_ff = m.constant_u32(t_u32, 0xFF);
    let c_u32_0f = m.constant_u32(t_u32, 0x0F);
    let c_u32_3f = m.constant_u32(t_u32, 0x3F);
    let c_u32_ffff = m.constant_u32(t_u32, 0xFFFF);
    let c_u32_1f = m.constant_u32(t_u32, 0x1F);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3FF);
    let c_u32_112 = m.constant_u32(t_u32, 112);
    let c_u32_256 = m.constant_u32(t_u32, 256);

    let c_f32_0 = m.constant_f32(t_f32, 0.0);

    // --- Decorations ---
    m.decorate(t_struct_weight, decoration::BLOCK, &[]);
    m.decorate(t_struct_input, decoration::BLOCK, &[]);
    m.decorate(t_struct_output, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_weight, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_input, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_output, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_u32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);

    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);

    // --- Global variables ---
    let gvar_weight = m.variable(t_ptr_sb_struct_weight, storage_class::STORAGE_BUFFER);
    let gvar_input = m.variable(t_ptr_sb_struct_input, storage_class::STORAGE_BUFFER);
    let gvar_output = m.variable(t_ptr_sb_struct_output, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc_struct, storage_class::PUSH_CONSTANT);
    let gvar_glob_id = m.variable(t_ptr_input_u32, storage_class::INPUT);

    m.decorate(gvar_weight, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_weight, decoration::BINDING, &[0]);
    m.decorate(gvar_input, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_input, decoration::BINDING, &[1]);
    m.decorate(gvar_output, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_output, decoration::BINDING, &[2]);
    m.decorate(
        gvar_glob_id,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    // --- Entry point ---
    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_glob_id]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    let lbl_entry = m.alloc_id();
    let lbl_bounds_merge = m.alloc_id();
    let lbl_bounds_true = m.alloc_id();
    let lbl_outer_header = m.alloc_id();
    let lbl_outer_body = m.alloc_id();
    let lbl_outer_continue = m.alloc_id();
    let lbl_outer_merge = m.alloc_id();

    // --- Function ---
    m.function(t_void, func_id, 0, t_fn_void);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let var_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_blk = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);

    let glob_id_vec = m.load(t_v3u32, gvar_glob_id);
    let row = m.composite_extract(t_u32, glob_id_vec, 0);

    let pc_ptr_rows = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let pc_rows = m.load(t_u32, pc_ptr_rows);
    let pc_ptr_cols = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let pc_cols = m.load(t_u32, pc_ptr_cols);

    let in_bounds = m.u_less_than(t_bool, row, pc_rows);
    m.selection_merge(lbl_bounds_merge, 0);
    m.branch_conditional(in_bounds, lbl_bounds_true, lbl_bounds_merge);

    // --- In-bounds block ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_true.0]));

    m.store(var_sum, c_f32_0);

    // num_blocks = cols / 256
    let num_blocks = m.udiv(t_u32, pc_cols, c_u32_256);

    m.store(var_blk, c_u32_0);
    m.branch(lbl_outer_header);

    // --- Outer loop header ---
    let lbl_outer_cond = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_header.0]));
    m.loop_merge(lbl_outer_merge, lbl_outer_continue, 0);
    m.branch(lbl_outer_cond);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_cond.0]));
    let blk_cur = m.load(t_u32, var_blk);
    let outer_cond = m.u_less_than(t_bool, blk_cur, num_blocks);
    m.branch_conditional(outer_cond, lbl_outer_body, lbl_outer_merge);

    // --- Outer loop body ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_body.0]));

    // === ROW-MAJOR weight layout ===
    // weight[row * num_blocks * 36 + block * 36 + plane]
    // row_base = row * num_blocks * 36
    let nb_x_36 = m.imul(t_u32, num_blocks, c_u32_36);
    let row_base = m.imul(t_u32, row, nb_x_36);
    // blk_base = row_base + block * 36
    let blk_x_36 = m.imul(t_u32, blk_cur, c_u32_36);
    let blk_base = m.iadd(t_u32, row_base, blk_x_36);

    // Load d and dmin from plane 0: weight[blk_base + 0]
    let packed_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, blk_base]);
    let packed_word = m.load(t_u32, packed_ptr);

    let c_f32_2pow_neg24 = m.constant_f32(t_f32, 5.9604644775390625e-8);

    // f16→f32: d
    let d_f16_raw = m.bitwise_and(t_u32, packed_word, c_u32_ffff);
    let d_f32 = {
        let exp_raw = m.shift_right_logical(t_u32, d_f16_raw, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, d_f16_raw, c_u32_3ff);
        let sign = m.shift_right_logical(t_u32, d_f16_raw, c_u32_15);
        let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
        let s_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
        let e_adj = m.iadd(t_u32, exp, c_u32_112);
        let e_part = m.shift_left_logical(t_u32, e_adj, c_u32_23);
        let m_part = m.shift_left_logical(t_u32, mant, c_u32_13);
        let bits_mid = m.bitwise_or(t_u32, s_part, e_part);
        let bits = m.bitwise_or(t_u32, bits_mid, m_part);
        let normal = m.bitcast(t_f32, bits);
        let mant_f = m.convert_u_to_f(t_f32, mant);
        let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
        let denorm_neg = m.fnegate(t_f32, denorm_abs);
        let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
        let denormal = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
        let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
        m.select(t_f32, exp_nonzero, normal, denormal)
    };

    // f16→f32: dmin
    let dmin_f16_raw = m.shift_right_logical(t_u32, packed_word, c_u32_16);
    let dmin_f32 = {
        let exp_raw = m.shift_right_logical(t_u32, dmin_f16_raw, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, dmin_f16_raw, c_u32_3ff);
        let sign = m.shift_right_logical(t_u32, dmin_f16_raw, c_u32_15);
        let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
        let s_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
        let e_adj = m.iadd(t_u32, exp, c_u32_112);
        let e_part = m.shift_left_logical(t_u32, e_adj, c_u32_23);
        let m_part = m.shift_left_logical(t_u32, mant, c_u32_13);
        let bits_mid = m.bitwise_or(t_u32, s_part, e_part);
        let bits = m.bitwise_or(t_u32, bits_mid, m_part);
        let normal = m.bitcast(t_f32, bits);
        let mant_f = m.convert_u_to_f(t_f32, mant);
        let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
        let denorm_neg = m.fnegate(t_f32, denorm_abs);
        let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
        let denormal = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
        let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
        m.select(t_f32, exp_nonzero, normal, denormal)
    };

    // Load scales from planes 1-3: weight[blk_base + 1..3]
    let s0_addr = m.iadd(t_u32, blk_base, c_u32_1);
    let s0_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, s0_addr]);
    let s0_word = m.load(t_u32, s0_ptr);

    let s1_addr = m.iadd(t_u32, blk_base, c_u32_2);
    let s1_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, s1_addr]);
    let s1_word = m.load(t_u32, s1_ptr);

    let s2_addr = m.iadd(t_u32, blk_base, c_u32_3);
    let s2_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, s2_addr]);
    let s2_word = m.load(t_u32, s2_ptr);

    // Extract 12 individual bytes: sb[0..11]
    let mut sb = [c_u32_0; 12];
    for i in 0..4u32 {
        if i == 0 {
            sb[0] = m.bitwise_and(t_u32, s0_word, c_u32_ff);
        } else {
            let shift = m.constant_u32(t_u32, i * 8);
            let shifted = m.shift_right_logical(t_u32, s0_word, shift);
            sb[i as usize] = m.bitwise_and(t_u32, shifted, c_u32_ff);
        }
    }
    for i in 0..4u32 {
        if i == 0 {
            sb[4] = m.bitwise_and(t_u32, s1_word, c_u32_ff);
        } else {
            let shift = m.constant_u32(t_u32, i * 8);
            let shifted = m.shift_right_logical(t_u32, s1_word, shift);
            sb[4 + i as usize] = m.bitwise_and(t_u32, shifted, c_u32_ff);
        }
    }
    for i in 0..4u32 {
        if i == 0 {
            sb[8] = m.bitwise_and(t_u32, s2_word, c_u32_ff);
        } else {
            let shift = m.constant_u32(t_u32, i * 8);
            let shifted = m.shift_right_logical(t_u32, s2_word, shift);
            sb[8 + i as usize] = m.bitwise_and(t_u32, shifted, c_u32_ff);
        }
    }

    // Extract 8 scales and 8 mins using 6-bit packing
    let c_u32_6 = m.constant_u32(t_u32, 6);

    let mut scales = [c_u32_0; 8];
    let mut mins = [c_u32_0; 8];

    for j in 0..4usize {
        scales[j] = m.bitwise_and(t_u32, sb[j], c_u32_3f);
        mins[j] = m.bitwise_and(t_u32, sb[j + 4], c_u32_3f);
    }

    for j in 4..8usize {
        let lo = m.bitwise_and(t_u32, sb[j + 4], c_u32_0f);
        let hi_raw = m.shift_right_logical(t_u32, sb[j - 4], c_u32_6);
        let hi = m.shift_left_logical(t_u32, hi_raw, c_u32_4);
        scales[j] = m.bitwise_or(t_u32, lo, hi);

        let lo2 = m.shift_right_logical(t_u32, sb[j + 4], c_u32_4);
        let hi2_raw = m.shift_right_logical(t_u32, sb[j], c_u32_6);
        let hi2 = m.shift_left_logical(t_u32, hi2_raw, c_u32_4);
        mins[j] = m.bitwise_or(t_u32, lo2, hi2);
    }

    // Process 8 sub-blocks
    let blk_x_256 = m.imul(t_u32, blk_cur, c_u32_256);
    let mut total_sum = c_f32_0;

    for sb_idx in 0..8u32 {
        let sc_f = {
            let sc_u = scales[sb_idx as usize];
            m.convert_u_to_f(t_f32, sc_u)
        };
        let mn_f = {
            let mn_u = mins[sb_idx as usize];
            m.convert_u_to_f(t_f32, mn_u)
        };

        let qs_group = sb_idx / 2;
        let is_high = (sb_idx & 1) != 0;

        // qs_base_plane = 4 + qs_group * 8 (same plane offset within block)
        let qs_base_plane = 4 + qs_group * 8;

        // ROW-MAJOR: qs at weight[blk_base + qs_base_plane + w]
        let c_qs_bp = m.constant_u32(t_u32, qs_base_plane);
        let qs_start = m.iadd(t_u32, blk_base, c_qs_bp);

        let c_sb_offset = m.constant_u32(t_u32, sb_idx * 32);
        let inp_sb_base = m.iadd(t_u32, blk_x_256, c_sb_offset);

        let mut nibble_input_sum = c_f32_0;
        let mut input_sum = c_f32_0;

        for w in 0..8u32 {
            // ROW-MAJOR: weight[qs_start + w]
            let qs_addr = if w == 0 {
                qs_start
            } else {
                let c_w = m.constant_u32(t_u32, w);
                m.iadd(t_u32, qs_start, c_w)
            };
            let qs_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, qs_addr]);
            let qs_word = m.load(t_u32, qs_ptr);

            let inp_w_base = if w == 0 {
                inp_sb_base
            } else {
                let c_w4 = m.constant_u32(t_u32, w * 4);
                m.iadd(t_u32, inp_sb_base, c_w4)
            };

            for byte_idx in 0..4u32 {
                let shift_amt = if is_high {
                    byte_idx * 8 + 4
                } else {
                    byte_idx * 8
                };
                let nibble = if shift_amt == 0 {
                    m.bitwise_and(t_u32, qs_word, c_u32_0f)
                } else {
                    let c_shift = m.constant_u32(t_u32, shift_amt);
                    let shifted = m.shift_right_logical(t_u32, qs_word, c_shift);
                    m.bitwise_and(t_u32, shifted, c_u32_0f)
                };
                let nibble_f = m.convert_u_to_f(t_f32, nibble);

                let inp_idx = if byte_idx == 0 {
                    inp_w_base
                } else {
                    let c_bi = m.constant_u32(t_u32, byte_idx);
                    m.iadd(t_u32, inp_w_base, c_bi)
                };
                let inp_ptr = m.access_chain(t_ptr_sb_f32, gvar_input, &[c_u32_0, inp_idx]);
                let inp_val = m.load(t_f32, inp_ptr);

                let ni_prod = m.fmul(t_f32, nibble_f, inp_val);
                nibble_input_sum = m.fadd(t_f32, nibble_input_sum, ni_prod);

                input_sum = m.fadd(t_f32, input_sum, inp_val);
            }
        }

        let d_sc = m.fmul(t_f32, d_f32, sc_f);
        let term1 = m.fmul(t_f32, d_sc, nibble_input_sum);
        let dmin_mn = m.fmul(t_f32, dmin_f32, mn_f);
        let term2 = m.fmul(t_f32, dmin_mn, input_sum);
        let sb_result = m.fsub(t_f32, term1, term2);
        total_sum = m.fadd(t_f32, total_sum, sb_result);
    }

    let prev_total = m.load(t_f32, var_sum);
    let new_total = m.fadd(t_f32, prev_total, total_sum);
    m.store(var_sum, new_total);
    m.branch(lbl_outer_continue);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_continue.0]));
    let blk_next = m.iadd(t_u32, blk_cur, c_u32_1);
    m.store(var_blk, blk_next);
    m.branch(lbl_outer_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_merge.0]));

    let final_sum = m.load(t_f32, var_sum);
    let out_ptr = m.access_chain(t_ptr_sb_f32, gvar_output, &[c_u32_0, row]);
    m.store(out_ptr, final_sum);
    m.branch(lbl_bounds_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_merge.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// Emit a complete SPIR-V compute shader for **Q6_K GEMV**.
///
/// Weight buffer uses transposed SoA layout for coalesced access:
///   weight[(block * 53 + plane) * rows + row]
///
/// Q6_K block structure (256 elements, 53 planes):
///   Plane 0:     d (f16, lower 16 bits)
///   Planes 1-4:  scales[16] as signed i8 packed in 4 u32 words
///   Planes 5-36: ql[128] as 32 u32 words (low 4 bits per element)
///   Planes 37-52: qh[64] as 16 u32 words (high 2 bits packed)
///
/// Storage buffers (set=0):
///   binding=0: weight (uint array, transposed SoA)
///   binding=1: input  (f32 array)
///   binding=2: output (f32 array)
///
/// Push constants: { rows: u32, cols: u32, rows_per_wg: u32 }
pub fn emit_q6k_gemv(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    // Capabilities & memory model
    m.capability(1); // Shader
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1); // Logical, GLSL450

    // --- Types ---
    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_i32 = m.type_int(32, 1);
    let t_f32 = m.type_float(32);

    let t_v3u32 = m.type_vector(t_u32, 3);

    let t_arr_u32 = m.type_runtime_array(t_u32);
    let t_arr_f32 = m.type_runtime_array(t_f32);

    let t_struct_weight = m.type_struct(&[t_arr_u32]);
    let t_struct_input = m.type_struct(&[t_arr_f32]);
    let t_struct_output = m.type_struct(&[t_arr_f32]);

    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);

    // Pointer types
    let t_ptr_sb_struct_weight = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_weight);
    let t_ptr_sb_struct_input = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_input);
    let t_ptr_sb_struct_output = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_output);
    let t_ptr_pc_struct = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_u32 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);

    let t_fn_void = m.type_function(t_void, &[]);

    // --- Constants ---
    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let _c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_4 = m.constant_u32(t_u32, 4);
    let c_u32_10 = m.constant_u32(t_u32, 10);
    let c_u32_13 = m.constant_u32(t_u32, 13);
    let c_u32_15 = m.constant_u32(t_u32, 15);
    let _c_u32_16 = m.constant_u32(t_u32, 16);
    let c_u32_23 = m.constant_u32(t_u32, 23);
    let c_u32_24 = m.constant_u32(t_u32, 24);
    let c_u32_31 = m.constant_u32(t_u32, 31);
    let c_u32_32 = m.constant_u32(t_u32, 32);
    let c_u32_53 = m.constant_u32(t_u32, 53);
    let c_u32_ff = m.constant_u32(t_u32, 0xFF);
    let c_u32_0f = m.constant_u32(t_u32, 0x0F);
    let c_u32_ffff = m.constant_u32(t_u32, 0xFFFF);
    let c_u32_1f = m.constant_u32(t_u32, 0x1F);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3FF);
    let c_u32_112 = m.constant_u32(t_u32, 112);
    let c_u32_256 = m.constant_u32(t_u32, 256);

    let c_f32_0 = m.constant_f32(t_f32, 0.0);

    // --- Decorations ---
    m.decorate(t_struct_weight, decoration::BLOCK, &[]);
    m.decorate(t_struct_input, decoration::BLOCK, &[]);
    m.decorate(t_struct_output, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_weight, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_input, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_output, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_u32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);

    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);

    // --- Global variables ---
    let gvar_weight = m.variable(t_ptr_sb_struct_weight, storage_class::STORAGE_BUFFER);
    let gvar_input = m.variable(t_ptr_sb_struct_input, storage_class::STORAGE_BUFFER);
    let gvar_output = m.variable(t_ptr_sb_struct_output, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc_struct, storage_class::PUSH_CONSTANT);
    let gvar_glob_id = m.variable(t_ptr_input_u32, storage_class::INPUT);

    m.decorate(gvar_weight, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_weight, decoration::BINDING, &[0]);
    m.decorate(gvar_input, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_input, decoration::BINDING, &[1]);
    m.decorate(gvar_output, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_output, decoration::BINDING, &[2]);
    m.decorate(
        gvar_glob_id,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    // --- Entry point ---
    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_glob_id]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    // Pre-allocate labels
    let lbl_entry = m.alloc_id();
    let lbl_bounds_merge = m.alloc_id();
    let lbl_bounds_true = m.alloc_id();
    let lbl_outer_header = m.alloc_id();
    let lbl_outer_body = m.alloc_id();
    let lbl_outer_continue = m.alloc_id();
    let lbl_outer_merge = m.alloc_id();

    // --- Function ---
    m.function(t_void, func_id, 0, t_fn_void);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    // Function variables
    let var_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_blk = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);

    // Load GlobalInvocationID.x → row
    let glob_id_vec = m.load(t_v3u32, gvar_glob_id);
    let row = m.composite_extract(t_u32, glob_id_vec, 0);

    // Load push constants: rows, cols
    let pc_ptr_rows = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let pc_rows = m.load(t_u32, pc_ptr_rows);
    let pc_ptr_cols = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let pc_cols = m.load(t_u32, pc_ptr_cols);

    // Bounds check: row < rows
    let in_bounds = m.u_less_than(t_bool, row, pc_rows);
    m.selection_merge(lbl_bounds_merge, 0);
    m.branch_conditional(in_bounds, lbl_bounds_true, lbl_bounds_merge);

    // --- In-bounds block ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_true.0]));

    m.store(var_sum, c_f32_0);

    // num_blocks = cols / 256
    let num_blocks = m.udiv(t_u32, pc_cols, c_u32_256);

    m.store(var_blk, c_u32_0);
    m.branch(lbl_outer_header);

    // --- Outer loop header (block loop) ---
    let lbl_outer_cond = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_header.0]));
    m.loop_merge(lbl_outer_merge, lbl_outer_continue, 0);
    m.branch(lbl_outer_cond);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_cond.0]));
    let blk_cur = m.load(t_u32, var_blk);
    let outer_cond = m.u_less_than(t_bool, blk_cur, num_blocks);
    m.branch_conditional(outer_cond, lbl_outer_body, lbl_outer_merge);

    // --- Outer loop body ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_body.0]));

    // === TRANSPOSED SoA weight layout ===
    // weight[(block * 53 + plane) * rows + row]
    // plane_base_offset = block * 53 * rows
    let blk_x_53 = m.imul(t_u32, blk_cur, c_u32_53);
    let plane_base = m.imul(t_u32, blk_x_53, pc_rows);

    // ============================================================
    // Load d from plane 0
    // packed = weight[plane_base + row]
    // d = f16_to_f32(packed & 0xFFFF)
    // ============================================================
    let packed_addr = m.iadd(t_u32, plane_base, row);
    let packed_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, packed_addr]);
    let packed_word = m.load(t_u32, packed_ptr);

    // f16->f32 helper constant: 2^(-24) for denormal conversion
    let c_f32_2pow_neg24 = m.constant_f32(t_f32, 5.9604644775390625e-8);

    // d = f16_to_f32(packed & 0xFFFF) with denormal + sign handling
    let d_f16_raw = m.bitwise_and(t_u32, packed_word, c_u32_ffff);
    let d_f32 = {
        let exp_raw = m.shift_right_logical(t_u32, d_f16_raw, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, d_f16_raw, c_u32_3ff);
        let sign = m.shift_right_logical(t_u32, d_f16_raw, c_u32_15);
        let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
        let s_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
        let e_adj = m.iadd(t_u32, exp, c_u32_112);
        let e_part = m.shift_left_logical(t_u32, e_adj, c_u32_23);
        let m_part = m.shift_left_logical(t_u32, mant, c_u32_13);
        let bits_mid = m.bitwise_or(t_u32, s_part, e_part);
        let bits = m.bitwise_or(t_u32, bits_mid, m_part);
        let normal = m.bitcast(t_f32, bits);
        // Denormal path with sign handling
        let mant_f = m.convert_u_to_f(t_f32, mant);
        let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
        let denorm_neg = m.fnegate(t_f32, denorm_abs);
        let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
        let denormal = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
        let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
        m.select(t_f32, exp_nonzero, normal, denormal)
    };

    // ============================================================
    // Load scale words from planes 1-4
    // scale_word[i] = weight[(plane_base + (1+i)*rows) + row]
    // ============================================================
    let mut scale_words = [c_u32_0; 4];
    for i in 0..4u32 {
        let c_plane = m.constant_u32(t_u32, i + 1);
        let plane_offset = m.imul(t_u32, c_plane, pc_rows);
        let addr = m.iadd(t_u32, plane_base, plane_offset);
        let addr = m.iadd(t_u32, addr, row);
        let ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, addr]);
        scale_words[i as usize] = m.load(t_u32, ptr);
    }

    // ============================================================
    // Process 16 sub-blocks, fully unrolled
    //
    // Sub-block table:
    // sb  ql_off  hi?   qh_off  qh_shift  elem_off  scale_idx
    // 0   5       false 37      0         0         0
    // 1   9       false 41      0         16        1
    // 2   13      false 37      2         32        2
    // 3   17      false 41      2         48        3
    // 4   5       true  37      4         64        4
    // 5   9       true  41      4         80        5
    // 6   13      true  37      6         96        6
    // 7   17      true  41      6         112       7
    // 8   21      false 45      0         128       8
    // 9   25      false 49      0         144       9
    // 10  29      false 45      2         160       10
    // 11  33      false 49      2         176       11
    // 12  21      true  45      4         192       12
    // 13  25      true  49      4         208       13
    // 14  29      true  45      6         224       14
    // 15  33      true  49      6         240       15
    // ============================================================

    struct SubBlockParams {
        ql_off: u32,
        is_high: bool,
        qh_off: u32,
        qh_shift: u32,
        elem_off: u32,
        scale_idx: u32,
    }

    let sub_blocks = [
        SubBlockParams {
            ql_off: 5,
            is_high: false,
            qh_off: 37,
            qh_shift: 0,
            elem_off: 0,
            scale_idx: 0,
        },
        SubBlockParams {
            ql_off: 9,
            is_high: false,
            qh_off: 41,
            qh_shift: 0,
            elem_off: 16,
            scale_idx: 1,
        },
        SubBlockParams {
            ql_off: 13,
            is_high: false,
            qh_off: 37,
            qh_shift: 2,
            elem_off: 32,
            scale_idx: 2,
        },
        SubBlockParams {
            ql_off: 17,
            is_high: false,
            qh_off: 41,
            qh_shift: 2,
            elem_off: 48,
            scale_idx: 3,
        },
        SubBlockParams {
            ql_off: 5,
            is_high: true,
            qh_off: 37,
            qh_shift: 4,
            elem_off: 64,
            scale_idx: 4,
        },
        SubBlockParams {
            ql_off: 9,
            is_high: true,
            qh_off: 41,
            qh_shift: 4,
            elem_off: 80,
            scale_idx: 5,
        },
        SubBlockParams {
            ql_off: 13,
            is_high: true,
            qh_off: 37,
            qh_shift: 6,
            elem_off: 96,
            scale_idx: 6,
        },
        SubBlockParams {
            ql_off: 17,
            is_high: true,
            qh_off: 41,
            qh_shift: 6,
            elem_off: 112,
            scale_idx: 7,
        },
        SubBlockParams {
            ql_off: 21,
            is_high: false,
            qh_off: 45,
            qh_shift: 0,
            elem_off: 128,
            scale_idx: 8,
        },
        SubBlockParams {
            ql_off: 25,
            is_high: false,
            qh_off: 49,
            qh_shift: 0,
            elem_off: 144,
            scale_idx: 9,
        },
        SubBlockParams {
            ql_off: 29,
            is_high: false,
            qh_off: 45,
            qh_shift: 2,
            elem_off: 160,
            scale_idx: 10,
        },
        SubBlockParams {
            ql_off: 33,
            is_high: false,
            qh_off: 49,
            qh_shift: 2,
            elem_off: 176,
            scale_idx: 11,
        },
        SubBlockParams {
            ql_off: 21,
            is_high: true,
            qh_off: 45,
            qh_shift: 4,
            elem_off: 192,
            scale_idx: 12,
        },
        SubBlockParams {
            ql_off: 25,
            is_high: true,
            qh_off: 49,
            qh_shift: 4,
            elem_off: 208,
            scale_idx: 13,
        },
        SubBlockParams {
            ql_off: 29,
            is_high: true,
            qh_off: 45,
            qh_shift: 6,
            elem_off: 224,
            scale_idx: 14,
        },
        SubBlockParams {
            ql_off: 33,
            is_high: true,
            qh_off: 49,
            qh_shift: 6,
            elem_off: 240,
            scale_idx: 15,
        },
    ];

    let blk_x_256 = m.imul(t_u32, blk_cur, c_u32_256);
    let mut total_sum = c_f32_0;

    for sb_param in &sub_blocks {
        // Extract signed i8 scale for this sub-block
        // scale_word = scale_words[sb_idx / 4]
        // byte = (scale_word >> ((sb_idx % 4) * 8)) & 0xFF
        // sign-extend: shift left 24, arithmetic shift right 24
        let scale_word = scale_words[(sb_param.scale_idx / 4) as usize];
        let byte_shift = (sb_param.scale_idx % 4) * 8;
        let scale_byte = if byte_shift == 0 {
            m.bitwise_and(t_u32, scale_word, c_u32_ff)
        } else {
            let c_shift = m.constant_u32(t_u32, byte_shift);
            let shifted = m.shift_right_logical(t_u32, scale_word, c_shift);
            m.bitwise_and(t_u32, shifted, c_u32_ff)
        };
        // Sign-extend u8 -> i32: shift left 24, arithmetic shift right 24
        let shl24 = m.shift_left_logical(t_u32, scale_byte, c_u32_24);
        let shl24_i32 = m.bitcast(t_i32, shl24);
        let sign_ext = m.shift_right_arithmetic(t_i32, shl24_i32, c_u32_24);
        let scale_f = m.convert_s_to_f(t_f32, sign_ext);

        // Compute plane offsets for ql and qh
        let c_ql_off = m.constant_u32(t_u32, sb_param.ql_off);
        let ql_plane_offset = m.imul(t_u32, c_ql_off, pc_rows);
        let ql_plane_start = m.iadd(t_u32, plane_base, ql_plane_offset);

        let c_qh_off = m.constant_u32(t_u32, sb_param.qh_off);
        let qh_plane_offset = m.imul(t_u32, c_qh_off, pc_rows);
        let qh_plane_start = m.iadd(t_u32, plane_base, qh_plane_offset);

        // Input base for this sub-block: blk*256 + elem_off
        let inp_sb_base = if sb_param.elem_off == 0 {
            blk_x_256
        } else {
            let c_elem_off = m.constant_u32(t_u32, sb_param.elem_off);
            m.iadd(t_u32, blk_x_256, c_elem_off)
        };

        let d_scale = m.fmul(t_f32, d_f32, scale_f);

        // 4 u32 ql words x 4 bytes = 16 elements per sub-block
        for w in 0..4u32 {
            // Load ql word: weight[(ql_plane_start + w * rows) + row]
            let ql_w_offset = if w == 0 {
                ql_plane_start
            } else {
                let c_w = m.constant_u32(t_u32, w);
                let w_x_rows = m.imul(t_u32, c_w, pc_rows);
                m.iadd(t_u32, ql_plane_start, w_x_rows)
            };
            let ql_addr = m.iadd(t_u32, ql_w_offset, row);
            let ql_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, ql_addr]);
            let ql_word = m.load(t_u32, ql_ptr);

            // Load qh word: weight[(qh_plane_start + w * rows) + row]
            let qh_w_offset = if w == 0 {
                qh_plane_start
            } else {
                let c_w = m.constant_u32(t_u32, w);
                let w_x_rows = m.imul(t_u32, c_w, pc_rows);
                m.iadd(t_u32, qh_plane_start, w_x_rows)
            };
            let qh_addr = m.iadd(t_u32, qh_w_offset, row);
            let qh_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, qh_addr]);
            let qh_word = m.load(t_u32, qh_ptr);

            // Input base for this word: inp_sb_base + w*4
            let inp_w_base = if w == 0 {
                inp_sb_base
            } else {
                let c_w4 = m.constant_u32(t_u32, w * 4);
                m.iadd(t_u32, inp_sb_base, c_w4)
            };

            // Process 4 bytes from this u32
            for byte_idx in 0..4u32 {
                // Extract ql nibble (low 4 bits or high 4 bits)
                let ql_shift_amt = if sb_param.is_high {
                    byte_idx * 8 + 4
                } else {
                    byte_idx * 8
                };
                let ql_nibble = if ql_shift_amt == 0 {
                    m.bitwise_and(t_u32, ql_word, c_u32_0f)
                } else {
                    let c_shift = m.constant_u32(t_u32, ql_shift_amt);
                    let shifted = m.shift_right_logical(t_u32, ql_word, c_shift);
                    m.bitwise_and(t_u32, shifted, c_u32_0f)
                };

                // Extract qh 2 bits
                let c_u32_qh_mask = c_u32_3;
                let qh_shift_amt = byte_idx * 8 + sb_param.qh_shift;
                let qh_bits = if qh_shift_amt == 0 {
                    m.bitwise_and(t_u32, qh_word, c_u32_qh_mask)
                } else {
                    let c_shift = m.constant_u32(t_u32, qh_shift_amt);
                    let shifted = m.shift_right_logical(t_u32, qh_word, c_shift);
                    m.bitwise_and(t_u32, shifted, c_u32_qh_mask)
                };

                // q6 = ql_nibble | (qh_bits << 4)
                let qh_shifted = m.shift_left_logical(t_u32, qh_bits, c_u32_4);
                let q6 = m.bitwise_or(t_u32, ql_nibble, qh_shifted);

                // q6_centered = q6 - 32 (signed: -32 to +31)
                // Use i32 subtraction for signed result
                let q6_i32 = m.bitcast(t_i32, q6);
                let c_32_i32 = m.bitcast(t_i32, c_u32_32);
                let q6_centered = m.isub(t_i32, q6_i32, c_32_i32);
                let q6_f = m.convert_s_to_f(t_f32, q6_centered);

                // Load input[inp_w_base + byte_idx]
                let inp_idx = if byte_idx == 0 {
                    inp_w_base
                } else {
                    let c_bi = m.constant_u32(t_u32, byte_idx);
                    m.iadd(t_u32, inp_w_base, c_bi)
                };
                let inp_ptr = m.access_chain(t_ptr_sb_f32, gvar_input, &[c_u32_0, inp_idx]);
                let inp_val = m.load(t_f32, inp_ptr);

                let dequant = m.fmul(t_f32, d_scale, q6_f);
                let qi_prod = m.fmul(t_f32, dequant, inp_val);
                total_sum = m.fadd(t_f32, total_sum, qi_prod);
            }
        }
    }

    // Accumulate block result
    let prev_total = m.load(t_f32, var_sum);
    let new_total = m.fadd(t_f32, prev_total, total_sum);
    m.store(var_sum, new_total);
    m.branch(lbl_outer_continue);

    // --- Outer loop continue ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_continue.0]));
    let blk_next = m.iadd(t_u32, blk_cur, c_u32_1);
    m.store(var_blk, blk_next);
    m.branch(lbl_outer_header);

    // --- Outer loop merge ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_merge.0]));

    let final_sum = m.load(t_f32, var_sum);
    let out_ptr = m.access_chain(t_ptr_sb_f32, gvar_output, &[c_u32_0, row]);
    m.store(out_ptr, final_sum);
    m.branch(lbl_bounds_merge);

    // --- Bounds merge ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_merge.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// Emit a Q6_K weight × Q8K activation integer-dot GEMV compute shader.
///
/// Mirrors CPU strict `dot_q6_k_q8k_scalar` semantics:
///   per sub-block (16 elements): sumi += scale * (ql|qh<<4 - 32) * q8k.qs[byte]
///   per block: acc += d * q8k.d * float(sumi)
///
/// Weight layout (binding 0): identical to `emit_q6k_gemv` (transposed SoA, 53 planes).
///
/// Activation layout (binding 1): packed u32 array, per-block stride = 69 u32 (276 B):
///   words 0..64  : qs (256 i8, 4 bytes per u32, little-endian)
///   word  64     : d (f32 bitcast)
///   words 65..69 : bsums (8 i16, unused for Q6_K but included for layout parity with Q4_K Q8K)
///
/// Output (binding 2): f32 array, one element per row.
/// Push constants: { rows, cols, rows_per_wg }.
pub fn emit_q6k_q8k_gemv(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1); // Shader
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1); // Logical, GLSL450

    // --- Types ---
    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_i32 = m.type_int(32, 1);
    let t_f32 = m.type_float(32);

    let t_v3u32 = m.type_vector(t_u32, 3);

    let t_arr_u32 = m.type_runtime_array(t_u32);
    let t_arr_f32 = m.type_runtime_array(t_f32);

    let t_struct_weight = m.type_struct(&[t_arr_u32]);
    let t_struct_input = m.type_struct(&[t_arr_u32]); // Q8K packed
    let t_struct_output = m.type_struct(&[t_arr_f32]);

    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);

    let t_ptr_sb_struct_weight = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_weight);
    let t_ptr_sb_struct_input = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_input);
    let t_ptr_sb_struct_output = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_output);
    let t_ptr_pc_struct = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_u32 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);

    let t_fn_void = m.type_function(t_void, &[]);

    // --- Constants ---
    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_4 = m.constant_u32(t_u32, 4);
    let c_u32_10 = m.constant_u32(t_u32, 10);
    let c_u32_13 = m.constant_u32(t_u32, 13);
    let c_u32_15 = m.constant_u32(t_u32, 15);
    let c_u32_23 = m.constant_u32(t_u32, 23);
    let c_u32_24 = m.constant_u32(t_u32, 24);
    let c_u32_31 = m.constant_u32(t_u32, 31);
    let c_u32_32 = m.constant_u32(t_u32, 32);
    let c_u32_53 = m.constant_u32(t_u32, 53);
    let c_u32_64 = m.constant_u32(t_u32, 64);
    let c_u32_69 = m.constant_u32(t_u32, 69);
    let c_u32_256 = m.constant_u32(t_u32, 256);
    let c_u32_ff = m.constant_u32(t_u32, 0xFF);
    let c_u32_0f = m.constant_u32(t_u32, 0x0F);
    let c_u32_ffff = m.constant_u32(t_u32, 0xFFFF);
    let c_u32_1f = m.constant_u32(t_u32, 0x1F);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3FF);
    let c_u32_112 = m.constant_u32(t_u32, 112);

    let c_i32_0 = m.constant_u32(t_i32, 0);

    let c_f32_0 = m.constant_f32(t_f32, 0.0);

    // --- Decorations ---
    m.decorate(t_struct_weight, decoration::BLOCK, &[]);
    m.decorate(t_struct_input, decoration::BLOCK, &[]);
    m.decorate(t_struct_output, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_weight, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_input, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_output, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_u32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);

    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);

    // --- Global variables ---
    let gvar_weight = m.variable(t_ptr_sb_struct_weight, storage_class::STORAGE_BUFFER);
    let gvar_input = m.variable(t_ptr_sb_struct_input, storage_class::STORAGE_BUFFER);
    let gvar_output = m.variable(t_ptr_sb_struct_output, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc_struct, storage_class::PUSH_CONSTANT);
    let gvar_glob_id = m.variable(t_ptr_input_u32, storage_class::INPUT);

    m.decorate(gvar_weight, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_weight, decoration::BINDING, &[0]);
    m.decorate(gvar_input, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_input, decoration::BINDING, &[1]);
    m.decorate(gvar_output, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_output, decoration::BINDING, &[2]);
    m.decorate(
        gvar_glob_id,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    // --- Entry point ---
    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_glob_id]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    let lbl_entry = m.alloc_id();
    let lbl_bounds_merge = m.alloc_id();
    let lbl_bounds_true = m.alloc_id();
    let lbl_outer_header = m.alloc_id();
    let lbl_outer_body = m.alloc_id();
    let lbl_outer_continue = m.alloc_id();
    let lbl_outer_merge = m.alloc_id();

    // --- Function ---
    m.function(t_void, func_id, 0, t_fn_void);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let var_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_blk = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);

    let glob_id_vec = m.load(t_v3u32, gvar_glob_id);
    let row = m.composite_extract(t_u32, glob_id_vec, 0);

    let pc_ptr_rows = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let pc_rows = m.load(t_u32, pc_ptr_rows);
    let pc_ptr_cols = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let pc_cols = m.load(t_u32, pc_ptr_cols);

    let in_bounds = m.u_less_than(t_bool, row, pc_rows);
    m.selection_merge(lbl_bounds_merge, 0);
    m.branch_conditional(in_bounds, lbl_bounds_true, lbl_bounds_merge);

    // --- In-bounds block ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_true.0]));

    m.store(var_sum, c_f32_0);

    let num_blocks = m.udiv(t_u32, pc_cols, c_u32_256);

    m.store(var_blk, c_u32_0);
    m.branch(lbl_outer_header);

    let lbl_outer_cond = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_header.0]));
    m.loop_merge(lbl_outer_merge, lbl_outer_continue, 0);
    m.branch(lbl_outer_cond);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_cond.0]));
    let blk_cur = m.load(t_u32, var_blk);
    let outer_cond = m.u_less_than(t_bool, blk_cur, num_blocks);
    m.branch_conditional(outer_cond, lbl_outer_body, lbl_outer_merge);

    // --- Outer loop body ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_body.0]));

    // Weight: transposed SoA, plane_base = block * 53 * rows
    let blk_x_53 = m.imul(t_u32, blk_cur, c_u32_53);
    let plane_base = m.imul(t_u32, blk_x_53, pc_rows);

    // Activation: per-block stride 69 u32
    let act_base = m.imul(t_u32, blk_cur, c_u32_69);

    // ---- Load weight d (plane 0, low 16 bits) ----
    let packed_addr = m.iadd(t_u32, plane_base, row);
    let packed_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, packed_addr]);
    let packed_word = m.load(t_u32, packed_ptr);

    let c_f32_2pow_neg24 = m.constant_f32(t_f32, 5.9604644775390625e-8);

    let d_f16_raw = m.bitwise_and(t_u32, packed_word, c_u32_ffff);
    let d_f32 = {
        let exp_raw = m.shift_right_logical(t_u32, d_f16_raw, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, d_f16_raw, c_u32_3ff);
        let sign = m.shift_right_logical(t_u32, d_f16_raw, c_u32_15);
        let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
        let s_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
        let e_adj = m.iadd(t_u32, exp, c_u32_112);
        let e_part = m.shift_left_logical(t_u32, e_adj, c_u32_23);
        let m_part = m.shift_left_logical(t_u32, mant, c_u32_13);
        let bits_mid = m.bitwise_or(t_u32, s_part, e_part);
        let bits = m.bitwise_or(t_u32, bits_mid, m_part);
        let normal = m.bitcast(t_f32, bits);
        let mant_f = m.convert_u_to_f(t_f32, mant);
        let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
        let denorm_neg = m.fnegate(t_f32, denorm_abs);
        let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
        let denormal = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
        let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
        m.select(t_f32, exp_nonzero, normal, denormal)
    };

    // ---- Load 4 scale words (planes 1..4) ----
    let mut scale_words = [c_u32_0; 4];
    for i in 0..4u32 {
        let c_plane = m.constant_u32(t_u32, i + 1);
        let plane_offset = m.imul(t_u32, c_plane, pc_rows);
        let addr = m.iadd(t_u32, plane_base, plane_offset);
        let addr = m.iadd(t_u32, addr, row);
        let ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, addr]);
        scale_words[i as usize] = m.load(t_u32, ptr);
    }

    // ---- Load activation block scale d (word 64) ----
    let q8k_d_addr = m.iadd(t_u32, act_base, c_u32_64);
    let q8k_d_ptr = m.access_chain(t_ptr_sb_u32, gvar_input, &[c_u32_0, q8k_d_addr]);
    let q8k_d_word = m.load(t_u32, q8k_d_ptr);
    let q8k_d = m.bitcast(t_f32, q8k_d_word);

    // ---- Per sub-block params (16 sub-blocks of 16 elements each) ----
    struct SubBlockParams {
        ql_off: u32,
        is_high: bool,
        qh_off: u32,
        qh_shift: u32,
        elem_off: u32,
        scale_idx: u32,
    }
    let sub_blocks = [
        SubBlockParams {
            ql_off: 5,
            is_high: false,
            qh_off: 37,
            qh_shift: 0,
            elem_off: 0,
            scale_idx: 0,
        },
        SubBlockParams {
            ql_off: 9,
            is_high: false,
            qh_off: 41,
            qh_shift: 0,
            elem_off: 16,
            scale_idx: 1,
        },
        SubBlockParams {
            ql_off: 13,
            is_high: false,
            qh_off: 37,
            qh_shift: 2,
            elem_off: 32,
            scale_idx: 2,
        },
        SubBlockParams {
            ql_off: 17,
            is_high: false,
            qh_off: 41,
            qh_shift: 2,
            elem_off: 48,
            scale_idx: 3,
        },
        SubBlockParams {
            ql_off: 5,
            is_high: true,
            qh_off: 37,
            qh_shift: 4,
            elem_off: 64,
            scale_idx: 4,
        },
        SubBlockParams {
            ql_off: 9,
            is_high: true,
            qh_off: 41,
            qh_shift: 4,
            elem_off: 80,
            scale_idx: 5,
        },
        SubBlockParams {
            ql_off: 13,
            is_high: true,
            qh_off: 37,
            qh_shift: 6,
            elem_off: 96,
            scale_idx: 6,
        },
        SubBlockParams {
            ql_off: 17,
            is_high: true,
            qh_off: 41,
            qh_shift: 6,
            elem_off: 112,
            scale_idx: 7,
        },
        SubBlockParams {
            ql_off: 21,
            is_high: false,
            qh_off: 45,
            qh_shift: 0,
            elem_off: 128,
            scale_idx: 8,
        },
        SubBlockParams {
            ql_off: 25,
            is_high: false,
            qh_off: 49,
            qh_shift: 0,
            elem_off: 144,
            scale_idx: 9,
        },
        SubBlockParams {
            ql_off: 29,
            is_high: false,
            qh_off: 45,
            qh_shift: 2,
            elem_off: 160,
            scale_idx: 10,
        },
        SubBlockParams {
            ql_off: 33,
            is_high: false,
            qh_off: 49,
            qh_shift: 2,
            elem_off: 176,
            scale_idx: 11,
        },
        SubBlockParams {
            ql_off: 21,
            is_high: true,
            qh_off: 45,
            qh_shift: 4,
            elem_off: 192,
            scale_idx: 12,
        },
        SubBlockParams {
            ql_off: 25,
            is_high: true,
            qh_off: 49,
            qh_shift: 4,
            elem_off: 208,
            scale_idx: 13,
        },
        SubBlockParams {
            ql_off: 29,
            is_high: true,
            qh_off: 45,
            qh_shift: 6,
            elem_off: 224,
            scale_idx: 14,
        },
        SubBlockParams {
            ql_off: 33,
            is_high: true,
            qh_off: 49,
            qh_shift: 6,
            elem_off: 240,
            scale_idx: 15,
        },
    ];

    // Block-level integer accumulator: sumi += sc * (q6-32) * q8k_byte (for all 256 elements).
    let mut sumi = c_i32_0;

    for sb_param in &sub_blocks {
        // ---- Extract signed i8 scale ----
        let scale_word = scale_words[(sb_param.scale_idx / 4) as usize];
        let byte_shift = (sb_param.scale_idx % 4) * 8;
        let scale_byte = if byte_shift == 0 {
            m.bitwise_and(t_u32, scale_word, c_u32_ff)
        } else {
            let c_shift = m.constant_u32(t_u32, byte_shift);
            let shifted = m.shift_right_logical(t_u32, scale_word, c_shift);
            m.bitwise_and(t_u32, shifted, c_u32_ff)
        };
        let shl24 = m.shift_left_logical(t_u32, scale_byte, c_u32_24);
        let shl24_i32 = m.bitcast(t_i32, shl24);
        let scale_i = m.shift_right_arithmetic(t_i32, shl24_i32, c_u32_24);

        // Plane offsets
        let c_ql_off = m.constant_u32(t_u32, sb_param.ql_off);
        let ql_plane_offset = m.imul(t_u32, c_ql_off, pc_rows);
        let ql_plane_start = m.iadd(t_u32, plane_base, ql_plane_offset);

        let c_qh_off = m.constant_u32(t_u32, sb_param.qh_off);
        let qh_plane_offset = m.imul(t_u32, c_qh_off, pc_rows);
        let qh_plane_start = m.iadd(t_u32, plane_base, qh_plane_offset);

        // Activation u32 base for this sub-block (16 elements = 4 u32 words at qs offset elem_off)
        let act_qs_word_off = sb_param.elem_off / 4; // u32 offset within block's qs
        let c_act_qs_off = m.constant_u32(t_u32, act_qs_word_off);
        let act_qs_base = if act_qs_word_off == 0 {
            act_base
        } else {
            m.iadd(t_u32, act_base, c_act_qs_off)
        };

        // 4 ql/qh u32 words × 4 bytes = 16 elements
        for w in 0..4u32 {
            let ql_w_offset = if w == 0 {
                ql_plane_start
            } else {
                let c_w = m.constant_u32(t_u32, w);
                let w_x_rows = m.imul(t_u32, c_w, pc_rows);
                m.iadd(t_u32, ql_plane_start, w_x_rows)
            };
            let ql_addr = m.iadd(t_u32, ql_w_offset, row);
            let ql_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, ql_addr]);
            let ql_word = m.load(t_u32, ql_ptr);

            let qh_w_offset = if w == 0 {
                qh_plane_start
            } else {
                let c_w = m.constant_u32(t_u32, w);
                let w_x_rows = m.imul(t_u32, c_w, pc_rows);
                m.iadd(t_u32, qh_plane_start, w_x_rows)
            };
            let qh_addr = m.iadd(t_u32, qh_w_offset, row);
            let qh_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, qh_addr]);
            let qh_word = m.load(t_u32, qh_ptr);

            let act_w_addr = if w == 0 {
                act_qs_base
            } else {
                let c_w = m.constant_u32(t_u32, w);
                m.iadd(t_u32, act_qs_base, c_w)
            };
            let act_ptr = m.access_chain(t_ptr_sb_u32, gvar_input, &[c_u32_0, act_w_addr]);
            let act_word = m.load(t_u32, act_ptr);

            for byte_idx in 0..4u32 {
                // ql nibble (4-bit unsigned)
                let ql_shift_amt = if sb_param.is_high {
                    byte_idx * 8 + 4
                } else {
                    byte_idx * 8
                };
                let ql_nibble = if ql_shift_amt == 0 {
                    m.bitwise_and(t_u32, ql_word, c_u32_0f)
                } else {
                    let c_shift = m.constant_u32(t_u32, ql_shift_amt);
                    let shifted = m.shift_right_logical(t_u32, ql_word, c_shift);
                    m.bitwise_and(t_u32, shifted, c_u32_0f)
                };

                // qh 2 bits
                let qh_shift_amt = byte_idx * 8 + sb_param.qh_shift;
                let qh_bits = if qh_shift_amt == 0 {
                    m.bitwise_and(t_u32, qh_word, c_u32_3)
                } else {
                    let c_shift = m.constant_u32(t_u32, qh_shift_amt);
                    let shifted = m.shift_right_logical(t_u32, qh_word, c_shift);
                    m.bitwise_and(t_u32, shifted, c_u32_3)
                };

                // q6 = ql_nibble | (qh_bits << 4); q6_centered = q6 - 32 (signed i32)
                let qh_shifted = m.shift_left_logical(t_u32, qh_bits, c_u32_4);
                let q6 = m.bitwise_or(t_u32, ql_nibble, qh_shifted);
                let q6_i32 = m.bitcast(t_i32, q6);
                let c_32_i32 = m.bitcast(t_i32, c_u32_32);
                let q6_centered = m.isub(t_i32, q6_i32, c_32_i32);

                // q8k byte (signed i8) at same byte_idx within act_word
                let act_byte = if byte_idx == 0 {
                    act_word
                } else {
                    let c_shift = m.constant_u32(t_u32, byte_idx * 8);
                    m.shift_right_logical(t_u32, act_word, c_shift)
                };
                let act_byte_top = m.shift_left_logical(t_u32, act_byte, c_u32_24);
                let act_byte_top_i = m.bitcast(t_i32, act_byte_top);
                let q8k_signed = m.shift_right_arithmetic(t_i32, act_byte_top_i, c_u32_24);

                // sumi += scale_i * q6_centered * q8k_signed
                let sc_q6 = m.imul(t_i32, scale_i, q6_centered);
                let prod = m.imul(t_i32, sc_q6, q8k_signed);
                sumi = m.iadd(t_i32, sumi, prod);
            }
        }
    }

    // Block-level f32 finalization: acc += d * q8k_d * float(sumi)
    let sumi_f = m.convert_s_to_f(t_f32, sumi);
    let d_q8k = m.fmul(t_f32, d_f32, q8k_d);
    let block_term = m.fmul(t_f32, d_q8k, sumi_f);

    let prev_total = m.load(t_f32, var_sum);
    let new_total = m.fadd(t_f32, prev_total, block_term);
    m.store(var_sum, new_total);
    m.branch(lbl_outer_continue);

    // --- Outer loop continue ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_continue.0]));
    let blk_next = m.iadd(t_u32, blk_cur, c_u32_1);
    m.store(var_blk, blk_next);
    m.branch(lbl_outer_header);

    // --- Outer loop merge ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_merge.0]));

    let final_sum = m.load(t_f32, var_sum);
    let out_ptr = m.access_chain(t_ptr_sb_f32, gvar_output, &[c_u32_0, row]);
    m.store(out_ptr, final_sum);
    m.branch(lbl_bounds_merge);

    // --- Bounds merge ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_merge.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// Emit a compute shader that quantizes f32 activations into Q8K packed blocks.
///
/// Mirrors CPU `quantize_input_q8k` (rnb-cpu/src/gemm/activation_q8.rs) for one
/// 256-element block per workgroup invocation. Used by Vulkan fullpath to feed
/// `emit_q4k_q8k_gemv` / `emit_q6k_q8k_gemv` without a host roundtrip.
///
/// Algorithm (per block):
///   amax = max(|x_i|) for i in 0..256
///   d = amax / 127
///   id = (d != 0) ? 1/d : 0
///   q_i = clamp(ties_away_from_zero_round(x_i * id), -128, 127)
///   bsums[g] = sum(q_i for i in g*32..(g+1)*32) as i16
///
/// Layout:
///   binding 0: input f32 array (n_blocks * 256 elements)
///   binding 1: output u32 array (n_blocks * 69 words per Q8K block packing)
///   push constant: n_blocks (u32)
///   local_size_x = 1, dispatch ceil(n_blocks / 1) workgroups
///
/// Output packing per block (69 u32 = 276 B), matching `pack_q8k_for_shader`:
///   words 0..64  : qs (256 i8, 4 per u32, little-endian)
///   word  64     : d (f32 bitcast)
///   words 65..69 : bsums (8 i16, 2 per u32, low half then high half)
pub fn emit_quantize_to_q8k(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1); // Shader
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1);

    // --- Types ---
    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_i32 = m.type_int(32, 1);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);

    let t_arr_in_f32 = m.type_runtime_array(t_f32);
    let t_arr_out_u32 = m.type_runtime_array(t_u32);

    let t_struct_in = m.type_struct(&[t_arr_in_f32]);
    let t_struct_out = m.type_struct(&[t_arr_out_u32]);
    let t_struct_pc = m.type_struct(&[t_u32]);

    let t_ptr_sb_in = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_in);
    let t_ptr_sb_out = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_out);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);
    let t_ptr_fn_i32 = m.type_pointer(storage_class::FUNCTION, t_i32);

    let t_fn_void = m.type_function(t_void, &[]);

    // --- Constants ---
    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_16 = m.constant_u32(t_u32, 16);
    let c_u32_24 = m.constant_u32(t_u32, 24);
    let c_u32_64 = m.constant_u32(t_u32, 64);
    let c_u32_69 = m.constant_u32(t_u32, 69);
    let c_u32_256 = m.constant_u32(t_u32, 256);
    let c_u32_ff = m.constant_u32(t_u32, 0xFF);
    let c_u32_ffff = m.constant_u32(t_u32, 0xFFFF);

    let c_i32_0 = m.constant_u32(t_i32, 0);

    let c_f32_0 = m.constant_f32(t_f32, 0.0);
    let c_f32_127 = m.constant_f32(t_f32, 127.0);
    let c_f32_128 = m.constant_f32(t_f32, 128.0);
    let c_f32_255 = m.constant_f32(t_f32, 255.0);
    let c_u32_128 = m.constant_u32(t_u32, 128);

    // 8 individual function-local i32 vars for bsums (one per group of 32
    // elements). Dynamic-indexed function-storage arrays were producing wrong
    // values on Adreno; using compile-time-resolved scalars sidesteps that.

    // --- Decorations ---
    m.decorate(t_struct_in, decoration::BLOCK, &[]);
    m.decorate(t_struct_out, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_in, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_out, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_in_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_out_u32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);

    // --- Globals ---
    let gvar_in = m.variable(t_ptr_sb_in, storage_class::STORAGE_BUFFER);
    let gvar_out = m.variable(t_ptr_sb_out, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let gvar_gid = m.variable(t_ptr_input_v3, storage_class::INPUT);

    m.decorate(gvar_in, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_in, decoration::BINDING, &[0]);
    m.decorate(gvar_out, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_out, decoration::BINDING, &[1]);
    m.decorate(
        gvar_gid,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    // --- Entry point ---
    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_gid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    let glsl = m.ext_inst_import("GLSL.std.450");

    // --- Function body ---
    let lbl_entry = m.alloc_id();
    let lbl_bounds_merge = m.alloc_id();
    let lbl_bounds_true = m.alloc_id();

    m.function(t_void, func_id, 0, t_fn_void);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let var_amax = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_idx = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_bsum_0 = m.function_variable(t_ptr_fn_i32, storage_class::FUNCTION);
    let var_bsum_1 = m.function_variable(t_ptr_fn_i32, storage_class::FUNCTION);
    let var_bsum_2 = m.function_variable(t_ptr_fn_i32, storage_class::FUNCTION);
    let var_bsum_3 = m.function_variable(t_ptr_fn_i32, storage_class::FUNCTION);
    let var_bsum_4 = m.function_variable(t_ptr_fn_i32, storage_class::FUNCTION);
    let var_bsum_5 = m.function_variable(t_ptr_fn_i32, storage_class::FUNCTION);
    let var_bsum_6 = m.function_variable(t_ptr_fn_i32, storage_class::FUNCTION);
    let var_bsum_7 = m.function_variable(t_ptr_fn_i32, storage_class::FUNCTION);
    let bsum_vars = [
        var_bsum_0, var_bsum_1, var_bsum_2, var_bsum_3, var_bsum_4, var_bsum_5, var_bsum_6,
        var_bsum_7,
    ];

    // Bounds check
    let gid_vec = m.load(t_v3u32, gvar_gid);
    let blk = m.composite_extract(t_u32, gid_vec, 0);

    let pc_n_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let n_blocks = m.load(t_u32, pc_n_ptr);

    let in_bounds = m.u_less_than(t_bool, blk, n_blocks);
    m.selection_merge(lbl_bounds_merge, 0);
    m.branch_conditional(in_bounds, lbl_bounds_true, lbl_bounds_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_true.0]));

    // input_base = blk * 256
    let in_base = m.imul(t_u32, blk, c_u32_256);
    // output_base = blk * 69
    let out_base = m.imul(t_u32, blk, c_u32_69);

    // ---- amax loop: 256 elements ----
    m.store(var_amax, c_f32_0);
    m.store(var_idx, c_u32_0);

    let lbl_amax_header = m.alloc_id();
    let lbl_amax_cond = m.alloc_id();
    let lbl_amax_body = m.alloc_id();
    let lbl_amax_continue = m.alloc_id();
    let lbl_amax_merge = m.alloc_id();

    m.branch(lbl_amax_header);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_amax_header.0]));
    m.loop_merge(lbl_amax_merge, lbl_amax_continue, 0);
    m.branch(lbl_amax_cond);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_amax_cond.0]));
    let i_amax = m.load(t_u32, var_idx);
    let cont_amax = m.u_less_than(t_bool, i_amax, c_u32_256);
    m.branch_conditional(cont_amax, lbl_amax_body, lbl_amax_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_amax_body.0]));
    let in_offset = m.iadd(t_u32, in_base, i_amax);
    let in_ptr = m.access_chain(t_ptr_sb_f32, gvar_in, &[c_u32_0, in_offset]);
    let x = m.load(t_f32, in_ptr);
    let abs_x = m.ext_inst(t_f32, glsl, 4, &[x]); // FAbs
    let cur_amax = m.load(t_f32, var_amax);
    let new_amax = m.ext_inst(t_f32, glsl, 40, &[cur_amax, abs_x]); // FMax
    m.store(var_amax, new_amax);
    m.branch(lbl_amax_continue);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_amax_continue.0]));
    let i_next_amax = m.iadd(t_u32, i_amax, c_u32_1);
    m.store(var_idx, i_next_amax);
    m.branch(lbl_amax_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_amax_merge.0]));

    // ---- d, id_inv ----
    let amax = m.load(t_f32, var_amax);
    let d = m.fdiv(t_f32, amax, c_f32_127);
    let nonzero = m.f_ord_greater_than(t_bool, d, c_f32_0);
    let one_f = m.constant_f32(t_f32, 1.0);
    let id_div = m.fdiv(t_f32, one_f, d);
    let id_inv = m.select(t_f32, nonzero, id_div, c_f32_0);

    // ---- Init 8 bsums vars = 0 ----
    for &v in bsum_vars.iter() {
        m.store(v, c_i32_0);
    }

    // ---- Pack qs words: 64 words × 4 bytes — fully UNROLLED.
    // Adreno SPIR-V loop processing exhibits early-exit on the prior loop form
    // (only word 0 was being written). Unrolling the 64-word outer loop and
    // 4-byte inner loop sidesteps the loop CFG entirely.
    for w in 0..64u32 {
        let group = (w / 8) as usize;
        let bsum_var = bsum_vars[group];
        let mut word_acc = c_u32_0;
        let mut sum_q_word = c_i32_0;
        let c_w = m.constant_u32(t_u32, w);
        for byte in 0..4u32 {
            let elem = w * 4 + byte;
            let c_elem = m.constant_u32(t_u32, elem);
            let in_off = m.iadd(t_u32, in_base, c_elem);
            let elem_ptr = m.access_chain(t_ptr_sb_f32, gvar_in, &[c_u32_0, in_off]);
            let x_e = m.load(t_f32, elem_ptr);

            let f = m.fmul(t_f32, x_e, id_inv);
            let rounded = m.ext_inst(t_f32, glsl, 1, &[f]); // GLSL Round (banker's)
                                                            // Adreno workaround: ConvertFToS zeros negative inputs. Shift to
                                                            // unsigned domain [0, 255], convert, then subtract 128 in u32 space
                                                            // to recover the i8 bit pattern (mod 256) and the signed value.
            let shifted = m.fadd(t_f32, rounded, c_f32_128);
            let clamped = m.ext_inst(t_f32, glsl, 43, &[shifted, c_f32_0, c_f32_255]);
            let q_unsigned = m.convert_f_to_u(t_u32, clamped);
            let q_signed_bits = m.isub(t_u32, q_unsigned, c_u32_128);
            let q_byte = m.bitwise_and(t_u32, q_signed_bits, c_u32_ff);
            let c_byte_shift = m.constant_u32(t_u32, byte * 8);
            let q_byte_shifted = m.shift_left_logical(t_u32, q_byte, c_byte_shift);
            word_acc = m.bitwise_or(t_u32, word_acc, q_byte_shifted);

            // Sign-extend low byte to i32 for bsums accumulation.
            let q_byte_top = m.shift_left_logical(t_u32, q_byte, c_u32_24);
            let q_byte_top_i = m.bitcast(t_i32, q_byte_top);
            let q_signed_i32 = m.shift_right_arithmetic(t_i32, q_byte_top_i, c_u32_24);
            sum_q_word = m.iadd(t_i32, sum_q_word, q_signed_i32);
        }
        let out_off = m.iadd(t_u32, out_base, c_w);
        let out_ptr_w = m.access_chain(t_ptr_sb_u32, gvar_out, &[c_u32_0, out_off]);
        m.store(out_ptr_w, word_acc);

        let bsum_cur = m.load(t_i32, bsum_var);
        let bsum_new = m.iadd(t_i32, bsum_cur, sum_q_word);
        m.store(bsum_var, bsum_new);
    }

    // ---- Store d (f32 bits) to output[out_base + 64] ----
    let d_bits = m.bitcast(t_u32, d);
    let out_off_d = m.iadd(t_u32, out_base, c_u32_64);
    let out_ptr_d = m.access_chain(t_ptr_sb_u32, gvar_out, &[c_u32_0, out_off_d]);
    m.store(out_ptr_d, d_bits);

    // ---- Pack bsums (8 i16) into 4 u32 words at out_base + 65..69 ----
    for g in 0..4u32 {
        let lo = m.load(t_i32, bsum_vars[(g * 2) as usize]);
        let hi = m.load(t_i32, bsum_vars[(g * 2 + 1) as usize]);
        let lo_u = m.bitcast(t_u32, lo);
        let hi_u = m.bitcast(t_u32, hi);
        let lo_masked = m.bitwise_and(t_u32, lo_u, c_u32_ffff);
        let hi_masked = m.bitwise_and(t_u32, hi_u, c_u32_ffff);
        let hi_shifted = m.shift_left_logical(t_u32, hi_masked, c_u32_16);
        let packed = m.bitwise_or(t_u32, lo_masked, hi_shifted);
        let c_g_off = m.constant_u32(t_u32, 65 + g);
        let out_off_b = m.iadd(t_u32, out_base, c_g_off);
        let out_ptr_b = m.access_chain(t_ptr_sb_u32, gvar_out, &[c_u32_0, out_off_b]);
        m.store(out_ptr_b, packed);
    }

    m.branch(lbl_bounds_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_merge.0]));
    m.ret();
    m.function_end();

    m.encode()
}

// ============================================================================
// Elementwise shaders for GPU-resident layer execution
// ============================================================================

/// Emit SPIR-V for fused SiLU+Mul: gate[i] = silu(gate[i]) * up[i]
///
/// Layout:
///   binding 0: gate buffer (f32, read/write)
///   binding 1: up buffer (f32, read)
///   push constant: count (u32)
pub fn emit_silu_mul(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    let glsl = m.ext_inst_import("GLSL.std.450");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_u32 = m.type_int(32, 0);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_arr_f32 = m.type_runtime_array(t_f32);

    let t_struct_gate = m.type_struct(&[t_arr_f32]);
    let t_struct_up = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32]);

    let t_ptr_sb_gate = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_gate);
    let t_ptr_sb_up = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_up);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_f32_1 = m.constant_f32(t_f32, 1.0);

    m.decorate(t_struct_gate, decoration::BLOCK, &[]);
    m.decorate(t_struct_up, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_gate, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_up, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);

    let gvar_gate = m.variable(t_ptr_sb_gate, storage_class::STORAGE_BUFFER);
    let gvar_up = m.variable(t_ptr_sb_up, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let gvar_gid = m.variable(t_ptr_input_v3, storage_class::INPUT);

    m.decorate(gvar_gate, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_gate, decoration::BINDING, &[0]);
    m.decorate(gvar_up, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_up, decoration::BINDING, &[1]);
    m.decorate(
        gvar_gid,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_gid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    m.function(t_void, func_id, 0, t_fn_void);
    let lbl_entry = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let gid_vec = m.load(t_v3u32, gvar_gid);
    let gid = m.composite_extract(t_u32, gid_vec, 0);

    let pc_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let count = m.load(t_u32, pc_ptr);

    let t_bool = m.type_bool();
    let in_bounds = m.u_less_than(t_bool, gid, count);
    let lbl_body = m.alloc_id();
    let lbl_end = m.alloc_id();
    m.selection_merge(lbl_end, 0);
    m.branch_conditional(in_bounds, lbl_body, lbl_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_body.0]));

    let gate_ptr = m.access_chain(t_ptr_sb_f32, gvar_gate, &[c_u32_0, gid]);
    let g = m.load(t_f32, gate_ptr);
    let up_ptr = m.access_chain(t_ptr_sb_f32, gvar_up, &[c_u32_0, gid]);
    let u = m.load(t_f32, up_ptr);

    // silu(g) = g / (1 + exp(-g))
    let neg_g = m.fnegate(t_f32, g);
    let exp_neg_g = m.ext_inst(t_f32, glsl, 27, &[neg_g]); // GLSL Exp = 27
    let one_plus = m.fadd(t_f32, c_f32_1, exp_neg_g);
    let silu_g = m.fdiv(t_f32, g, one_plus);
    let result = m.fmul(t_f32, silu_g, u);
    m.store(gate_ptr, result);

    m.branch(lbl_end);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_end.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// Emit SPIR-V for attention gate application: target[i] *= sigmoid(gate[i]).
///
/// Layout:
///   binding 0: gate buffer (f32, read)
///   binding 1: target buffer (f32, read/write)
///   push constant: count (u32)
pub fn emit_sigmoid_mul(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    let glsl = m.ext_inst_import("GLSL.std.450");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_u32 = m.type_int(32, 0);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_arr_f32 = m.type_runtime_array(t_f32);

    let t_struct_gate = m.type_struct(&[t_arr_f32]);
    let t_struct_target = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32]);

    let t_ptr_sb_gate = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_gate);
    let t_ptr_sb_target = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_target);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_f32_1 = m.constant_f32(t_f32, 1.0);

    m.decorate(t_struct_gate, decoration::BLOCK, &[]);
    m.decorate(t_struct_target, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_gate, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_target, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);

    let gvar_gate = m.variable(t_ptr_sb_gate, storage_class::STORAGE_BUFFER);
    let gvar_target = m.variable(t_ptr_sb_target, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let gvar_gid = m.variable(t_ptr_input_v3, storage_class::INPUT);

    m.decorate(gvar_gate, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_gate, decoration::BINDING, &[0]);
    m.decorate(gvar_target, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_target, decoration::BINDING, &[1]);
    m.decorate(
        gvar_gid,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_gid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    m.function(t_void, func_id, 0, t_fn_void);
    let lbl_entry = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let gid_vec = m.load(t_v3u32, gvar_gid);
    let gid = m.composite_extract(t_u32, gid_vec, 0);

    let pc_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let count = m.load(t_u32, pc_ptr);

    let t_bool = m.type_bool();
    let in_bounds = m.u_less_than(t_bool, gid, count);
    let lbl_body = m.alloc_id();
    let lbl_end = m.alloc_id();
    m.selection_merge(lbl_end, 0);
    m.branch_conditional(in_bounds, lbl_body, lbl_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_body.0]));

    let gate_ptr = m.access_chain(t_ptr_sb_f32, gvar_gate, &[c_u32_0, gid]);
    let g = m.load(t_f32, gate_ptr);
    let target_ptr = m.access_chain(t_ptr_sb_f32, gvar_target, &[c_u32_0, gid]);
    let y = m.load(t_f32, target_ptr);

    let neg_g = m.fnegate(t_f32, g);
    let exp_neg_g = m.ext_inst(t_f32, glsl, 27, &[neg_g]);
    let denom = m.fadd(t_f32, c_f32_1, exp_neg_g);
    let sig = m.fdiv(t_f32, c_f32_1, denom);
    let result = m.fmul(t_f32, y, sig);
    m.store(target_ptr, result);

    m.branch(lbl_end);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_end.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// Emit SPIR-V for Qwen-style gated-Q split.
///
/// Source layout per token/head is `[q(head_dim), gate(head_dim)]`.
/// Destination Q and gate layouts are both packed `[token, head, dim]`.
///
/// Layout:
///   binding 0: q_full buffer (f32, read)
///   binding 1: q_out buffer (f32, write)
///   binding 2: gate_out buffer (f32, write)
///   push constants: count, q_dim, head_dim (u32)
pub fn emit_split_gated_q(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_arr_f32 = m.type_runtime_array(t_f32);

    let t_struct_src = m.type_struct(&[t_arr_f32]);
    let t_struct_q = m.type_struct(&[t_arr_f32]);
    let t_struct_gate = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);

    let t_ptr_sb_src = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_src);
    let t_ptr_sb_q = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_q);
    let t_ptr_sb_gate = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_gate);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);

    m.decorate(t_struct_src, decoration::BLOCK, &[]);
    m.decorate(t_struct_q, decoration::BLOCK, &[]);
    m.decorate(t_struct_gate, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_src, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_q, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_gate, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);

    let gvar_src = m.variable(t_ptr_sb_src, storage_class::STORAGE_BUFFER);
    let gvar_q = m.variable(t_ptr_sb_q, storage_class::STORAGE_BUFFER);
    let gvar_gate = m.variable(t_ptr_sb_gate, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let gvar_gid = m.variable(t_ptr_input_v3, storage_class::INPUT);

    m.decorate(gvar_src, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_src, decoration::BINDING, &[0]);
    m.decorate(gvar_q, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_q, decoration::BINDING, &[1]);
    m.decorate(gvar_gate, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_gate, decoration::BINDING, &[2]);
    m.decorate(
        gvar_gid,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_gid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    m.function(t_void, func_id, 0, t_fn_void);
    let lbl_entry = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let gid_vec = m.load(t_v3u32, gvar_gid);
    let gid = m.composite_extract(t_u32, gid_vec, 0);

    let pc_count_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let count = m.load(t_u32, pc_count_ptr);
    let pc_q_dim_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let q_dim = m.load(t_u32, pc_q_dim_ptr);
    let pc_head_dim_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_2]);
    let head_dim = m.load(t_u32, pc_head_dim_ptr);

    let in_bounds = m.u_less_than(t_bool, gid, count);
    let lbl_body = m.alloc_id();
    let lbl_end = m.alloc_id();
    m.selection_merge(lbl_end, 0);
    m.branch_conditional(in_bounds, lbl_body, lbl_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_body.0]));

    let token = m.udiv(t_u32, gid, q_dim);
    let within_q = m.umod(t_u32, gid, q_dim);
    let head = m.udiv(t_u32, within_q, head_dim);
    let dim = m.umod(t_u32, within_q, head_dim);
    let q_rows = m.imul(t_u32, q_dim, c_u32_2);
    let token_base = m.imul(t_u32, token, q_rows);
    let per_head = m.imul(t_u32, head_dim, c_u32_2);
    let head_base = m.imul(t_u32, head, per_head);
    let src_head_base = m.iadd(t_u32, token_base, head_base);
    let src_q_idx = m.iadd(t_u32, src_head_base, dim);
    let src_gate_idx = m.iadd(t_u32, src_q_idx, head_dim);

    let src_q_ptr = m.access_chain(t_ptr_sb_f32, gvar_src, &[c_u32_0, src_q_idx]);
    let q_val = m.load(t_f32, src_q_ptr);
    let src_gate_ptr = m.access_chain(t_ptr_sb_f32, gvar_src, &[c_u32_0, src_gate_idx]);
    let gate_val = m.load(t_f32, src_gate_ptr);

    let q_out_ptr = m.access_chain(t_ptr_sb_f32, gvar_q, &[c_u32_0, gid]);
    m.store(q_out_ptr, q_val);
    let gate_out_ptr = m.access_chain(t_ptr_sb_f32, gvar_gate, &[c_u32_0, gid]);
    m.store(gate_out_ptr, gate_val);

    m.branch(lbl_end);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_end.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// Emit SPIR-V for elementwise add: a[i] += b[i]
///
/// Layout:
///   binding 0: a buffer (f32, read/write)
///   binding 1: b buffer (f32, read)
///   push constant: count (u32)
pub fn emit_elem_add(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_u32 = m.type_int(32, 0);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_arr_f32 = m.type_runtime_array(t_f32);

    let t_struct_a = m.type_struct(&[t_arr_f32]);
    let t_struct_b = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32]);

    let t_ptr_sb_a = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_a);
    let t_ptr_sb_b = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_b);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);

    m.decorate(t_struct_a, decoration::BLOCK, &[]);
    m.decorate(t_struct_b, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_a, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_b, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);

    let gvar_a = m.variable(t_ptr_sb_a, storage_class::STORAGE_BUFFER);
    let gvar_b = m.variable(t_ptr_sb_b, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let gvar_gid = m.variable(t_ptr_input_v3, storage_class::INPUT);

    m.decorate(gvar_a, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_a, decoration::BINDING, &[0]);
    m.decorate(gvar_b, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_b, decoration::BINDING, &[1]);
    m.decorate(
        gvar_gid,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_gid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    m.function(t_void, func_id, 0, t_fn_void);
    let lbl_entry = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let gid_vec = m.load(t_v3u32, gvar_gid);
    let gid = m.composite_extract(t_u32, gid_vec, 0);

    let pc_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let count = m.load(t_u32, pc_ptr);

    let t_bool = m.type_bool();
    let in_bounds = m.u_less_than(t_bool, gid, count);
    let lbl_body = m.alloc_id();
    let lbl_end = m.alloc_id();
    m.selection_merge(lbl_end, 0);
    m.branch_conditional(in_bounds, lbl_body, lbl_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_body.0]));
    let a_ptr = m.access_chain(t_ptr_sb_f32, gvar_a, &[c_u32_0, gid]);
    let a_val = m.load(t_f32, a_ptr);
    let b_ptr = m.access_chain(t_ptr_sb_f32, gvar_b, &[c_u32_0, gid]);
    let b_val = m.load(t_f32, b_ptr);
    let sum = m.fadd(t_f32, a_val, b_val);
    m.store(a_ptr, sum);

    m.branch(lbl_end);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_end.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// Emit SPIR-V for elementwise add into a separate output:
/// out[i] = a[i] + b[i]
///
/// Layout:
///   binding 0: a buffer (f32, read)
///   binding 1: b buffer (f32, read)
///   binding 2: out buffer (f32, write)
///   push constant: count (u32)
pub fn emit_elem_add_out(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_u32 = m.type_int(32, 0);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_arr_f32 = m.type_runtime_array(t_f32);

    let t_struct_a = m.type_struct(&[t_arr_f32]);
    let t_struct_b = m.type_struct(&[t_arr_f32]);
    let t_struct_out = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32]);

    let t_ptr_sb_a = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_a);
    let t_ptr_sb_b = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_b);
    let t_ptr_sb_out = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_out);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);

    m.decorate(t_struct_a, decoration::BLOCK, &[]);
    m.decorate(t_struct_b, decoration::BLOCK, &[]);
    m.decorate(t_struct_out, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_a, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_b, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_out, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);

    let gvar_a = m.variable(t_ptr_sb_a, storage_class::STORAGE_BUFFER);
    let gvar_b = m.variable(t_ptr_sb_b, storage_class::STORAGE_BUFFER);
    let gvar_out = m.variable(t_ptr_sb_out, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let gvar_gid = m.variable(t_ptr_input_v3, storage_class::INPUT);

    m.decorate(gvar_a, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_a, decoration::BINDING, &[0]);
    m.decorate(gvar_b, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_b, decoration::BINDING, &[1]);
    m.decorate(gvar_out, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_out, decoration::BINDING, &[2]);
    m.decorate(
        gvar_gid,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_gid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    m.function(t_void, func_id, 0, t_fn_void);
    let lbl_entry = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let gid_vec = m.load(t_v3u32, gvar_gid);
    let gid = m.composite_extract(t_u32, gid_vec, 0);

    let pc_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let count = m.load(t_u32, pc_ptr);

    let t_bool = m.type_bool();
    let in_bounds = m.u_less_than(t_bool, gid, count);
    let lbl_body = m.alloc_id();
    let lbl_end = m.alloc_id();
    m.selection_merge(lbl_end, 0);
    m.branch_conditional(in_bounds, lbl_body, lbl_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_body.0]));
    let a_ptr = m.access_chain(t_ptr_sb_f32, gvar_a, &[c_u32_0, gid]);
    let a_val = m.load(t_f32, a_ptr);
    let b_ptr = m.access_chain(t_ptr_sb_f32, gvar_b, &[c_u32_0, gid]);
    let b_val = m.load(t_f32, b_ptr);
    let out_ptr = m.access_chain(t_ptr_sb_f32, gvar_out, &[c_u32_0, gid]);
    let sum = m.fadd(t_f32, a_val, b_val);
    m.store(out_ptr, sum);

    m.branch(lbl_end);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_end.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// Emit SPIR-V for GDN gated norm:
/// out[i] = rms_norm(delta_head)[i] * ssm_norm[i % head_v_dim] * silu(z[i]).
///
/// Layout:
///   binding 0: delta_out (f32, read)
///   binding 1: z gate    (f32, read)
///   binding 2: ssm_norm  (f32, read, length=head_v_dim)
///   binding 3: output    (f32, write)
///   push constants: d_inner (u32), head_v_dim (u32), eps_bits (u32)
pub fn emit_gdn_gated_norm_silu(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    let glsl = m.ext_inst_import("GLSL.std.450");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_arr_f32 = m.type_runtime_array(t_f32);

    let t_struct_delta = m.type_struct(&[t_arr_f32]);
    let t_struct_z = m.type_struct(&[t_arr_f32]);
    let t_struct_norm = m.type_struct(&[t_arr_f32]);
    let t_struct_out = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);

    let t_ptr_delta = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_delta);
    let t_ptr_z = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_z);
    let t_ptr_norm = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_norm);
    let t_ptr_out = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_out);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);
    let c_f32_1 = m.constant_f32(t_f32, 1.0);

    for s in [t_struct_delta, t_struct_z, t_struct_norm, t_struct_out] {
        m.decorate(s, decoration::BLOCK, &[]);
        m.member_decorate(s, 0, decoration::OFFSET, &[0]);
    }
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);

    let gvar_delta = m.variable(t_ptr_delta, storage_class::STORAGE_BUFFER);
    let gvar_z = m.variable(t_ptr_z, storage_class::STORAGE_BUFFER);
    let gvar_norm = m.variable(t_ptr_norm, storage_class::STORAGE_BUFFER);
    let gvar_out = m.variable(t_ptr_out, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let gvar_gid = m.variable(t_ptr_input_v3, storage_class::INPUT);

    m.decorate(gvar_delta, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_delta, decoration::BINDING, &[0]);
    m.decorate(gvar_z, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_z, decoration::BINDING, &[1]);
    m.decorate(gvar_norm, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_norm, decoration::BINDING, &[2]);
    m.decorate(gvar_out, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_out, decoration::BINDING, &[3]);
    m.decorate(
        gvar_gid,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_gid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    let lbl_entry = m.alloc_id();
    let lbl_body = m.alloc_id();
    let lbl_end = m.alloc_id();
    let lbl_loop_header = m.alloc_id();
    let lbl_loop_cond = m.alloc_id();
    let lbl_loop_body = m.alloc_id();
    let lbl_loop_continue = m.alloc_id();
    let lbl_loop_merge = m.alloc_id();

    m.function(t_void, func_id, 0, t_fn_void);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));
    let var_j = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);

    let gid_vec = m.load(t_v3u32, gvar_gid);
    let gid = m.composite_extract(t_u32, gid_vec, 0);
    let pc_d_inner_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let d_inner = m.load(t_u32, pc_d_inner_ptr);
    let pc_head_v_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let head_v_dim = m.load(t_u32, pc_head_v_ptr);
    let pc_eps_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_2]);
    let eps_bits = m.load(t_u32, pc_eps_ptr);
    let eps = m.bitcast(t_f32, eps_bits);

    let in_bounds = m.u_less_than(t_bool, gid, d_inner);
    m.selection_merge(lbl_end, 0);
    m.branch_conditional(in_bounds, lbl_body, lbl_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_body.0]));
    let local_i = m.umod(t_u32, gid, head_v_dim);
    let head_idx = m.udiv(t_u32, gid, head_v_dim);
    let head_base = m.imul(t_u32, head_idx, head_v_dim);
    m.store(var_j, c_u32_0);
    m.store(var_sum, c_f32_0);
    m.branch(lbl_loop_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_header.0]));
    m.loop_merge(lbl_loop_merge, lbl_loop_continue, 0);
    m.branch(lbl_loop_cond);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_cond.0]));
    let j_cur = m.load(t_u32, var_j);
    let loop_cond = m.u_less_than(t_bool, j_cur, head_v_dim);
    m.branch_conditional(loop_cond, lbl_loop_body, lbl_loop_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_body.0]));
    let delta_idx_j = m.iadd(t_u32, head_base, j_cur);
    let delta_ptr_j = m.access_chain(t_ptr_sb_f32, gvar_delta, &[c_u32_0, delta_idx_j]);
    let delta_j = m.load(t_f32, delta_ptr_j);
    let sq = m.fmul(t_f32, delta_j, delta_j);
    let sum_old = m.load(t_f32, var_sum);
    let sum_new = m.fadd(t_f32, sum_old, sq);
    m.store(var_sum, sum_new);
    m.branch(lbl_loop_continue);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_continue.0]));
    let j_for_next = m.load(t_u32, var_j);
    let j_next = m.iadd(t_u32, j_for_next, c_u32_1);
    m.store(var_j, j_next);
    m.branch(lbl_loop_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_merge.0]));
    let sum_final = m.load(t_f32, var_sum);
    let head_v_f = m.convert_u_to_f(t_f32, head_v_dim);
    let mean = m.fdiv(t_f32, sum_final, head_v_f);
    let denom_sq = m.fadd(t_f32, mean, eps);
    let denom = m.ext_inst(t_f32, glsl, 31, &[denom_sq]); // GLSL Sqrt = 31
    let inv = m.fdiv(t_f32, c_f32_1, denom);

    let delta_ptr = m.access_chain(t_ptr_sb_f32, gvar_delta, &[c_u32_0, gid]);
    let delta_val = m.load(t_f32, delta_ptr);
    let norm_ptr = m.access_chain(t_ptr_sb_f32, gvar_norm, &[c_u32_0, local_i]);
    let norm_weight = m.load(t_f32, norm_ptr);
    let z_ptr = m.access_chain(t_ptr_sb_f32, gvar_z, &[c_u32_0, gid]);
    let z = m.load(t_f32, z_ptr);
    let neg_z = m.fnegate(t_f32, z);
    let exp_neg_z = m.ext_inst(t_f32, glsl, 27, &[neg_z]); // GLSL Exp = 27
    let one_plus = m.fadd(t_f32, c_f32_1, exp_neg_z);
    let silu_z = m.fdiv(t_f32, z, one_plus);
    let normed = m.fmul(t_f32, delta_val, inv);
    let weighted = m.fmul(t_f32, normed, norm_weight);
    let out_val = m.fmul(t_f32, weighted, silu_z);
    let out_ptr = m.access_chain(t_ptr_sb_f32, gvar_out, &[c_u32_0, gid]);
    m.store(out_ptr, out_val);
    m.branch(lbl_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_end.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// Emit SPIR-V for one GDN DeltaNet token step.
///
/// Layout:
///   binding 0: conv_out  (f32, read, one token q/k/v after conv+SiLU)
///   binding 1: alpha_raw (f32, read, one token, length=num_v_heads)
///   binding 2: beta_raw  (f32, read, one token, length=num_v_heads)
///   binding 3: ssm_a     (f32, read, length=num_v_heads)
///   binding 4: dt_bias   (f32, read, length=num_v_heads)
///   binding 5: state     (f32, read/write, num_v_heads*head_v_dim*head_k_dim)
///   binding 6: output    (f32, write, length=d_inner)
///   push constants:
///     conv_channels, d_inner, num_k_heads, num_v_heads,
///     head_k_dim, head_v_dim, norm_eps_bits
pub fn emit_gdn_delta_step(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    let glsl = m.ext_inst_import("GLSL.std.450");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_arr_f32 = m.type_runtime_array(t_f32);

    let t_struct_conv = m.type_struct(&[t_arr_f32]);
    let t_struct_alpha = m.type_struct(&[t_arr_f32]);
    let t_struct_beta = m.type_struct(&[t_arr_f32]);
    let t_struct_ssm_a = m.type_struct(&[t_arr_f32]);
    let t_struct_dt_bias = m.type_struct(&[t_arr_f32]);
    let t_struct_state = m.type_struct(&[t_arr_f32]);
    let t_struct_out = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32, t_u32, t_u32, t_u32, t_u32]);

    let t_ptr_conv = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_conv);
    let t_ptr_alpha = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_alpha);
    let t_ptr_beta = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_beta);
    let t_ptr_ssm_a = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_ssm_a);
    let t_ptr_dt_bias = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_dt_bias);
    let t_ptr_state = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_state);
    let t_ptr_out = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_out);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_4 = m.constant_u32(t_u32, 4);
    let c_u32_5 = m.constant_u32(t_u32, 5);
    let c_u32_6 = m.constant_u32(t_u32, 6);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);
    let c_f32_1 = m.constant_f32(t_f32, 1.0);

    for s in [
        t_struct_conv,
        t_struct_alpha,
        t_struct_beta,
        t_struct_ssm_a,
        t_struct_dt_bias,
        t_struct_state,
        t_struct_out,
    ] {
        m.decorate(s, decoration::BLOCK, &[]);
        m.member_decorate(s, 0, decoration::OFFSET, &[0]);
    }
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    for member in 0..7 {
        m.member_decorate(t_struct_pc, member, decoration::OFFSET, &[member * 4]);
    }

    let gvar_conv = m.variable(t_ptr_conv, storage_class::STORAGE_BUFFER);
    let gvar_alpha = m.variable(t_ptr_alpha, storage_class::STORAGE_BUFFER);
    let gvar_beta = m.variable(t_ptr_beta, storage_class::STORAGE_BUFFER);
    let gvar_ssm_a = m.variable(t_ptr_ssm_a, storage_class::STORAGE_BUFFER);
    let gvar_dt_bias = m.variable(t_ptr_dt_bias, storage_class::STORAGE_BUFFER);
    let gvar_state = m.variable(t_ptr_state, storage_class::STORAGE_BUFFER);
    let gvar_out = m.variable(t_ptr_out, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let gvar_gid = m.variable(t_ptr_input_v3, storage_class::INPUT);

    for (binding, var) in [
        gvar_conv,
        gvar_alpha,
        gvar_beta,
        gvar_ssm_a,
        gvar_dt_bias,
        gvar_state,
        gvar_out,
    ]
    .into_iter()
    .enumerate()
    {
        m.decorate(var, decoration::DESCRIPTOR_SET, &[0]);
        m.decorate(var, decoration::BINDING, &[binding as u32]);
    }
    m.decorate(
        gvar_gid,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_gid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    let lbl_entry = m.alloc_id();
    let lbl_body = m.alloc_id();
    let lbl_end = m.alloc_id();
    let lbl_qk_header = m.alloc_id();
    let lbl_qk_cond = m.alloc_id();
    let lbl_qk_body = m.alloc_id();
    let lbl_qk_continue = m.alloc_id();
    let lbl_qk_merge = m.alloc_id();
    let lbl_sk_header = m.alloc_id();
    let lbl_sk_cond = m.alloc_id();
    let lbl_sk_body = m.alloc_id();
    let lbl_sk_continue = m.alloc_id();
    let lbl_sk_merge = m.alloc_id();
    let lbl_update_header = m.alloc_id();
    let lbl_update_cond = m.alloc_id();
    let lbl_update_body = m.alloc_id();
    let lbl_update_continue = m.alloc_id();
    let lbl_update_merge = m.alloc_id();

    m.function(t_void, func_id, 0, t_fn_void);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));
    let var_j = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_q_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_k_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_sk = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_out = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);

    let gid_vec = m.load(t_v3u32, gvar_gid);
    let gid = m.composite_extract(t_u32, gid_vec, 0);
    let pc_conv_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let conv_channels = m.load(t_u32, pc_conv_ptr);
    let pc_d_inner_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let d_inner = m.load(t_u32, pc_d_inner_ptr);
    let pc_num_k_heads_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_2]);
    let num_k_heads = m.load(t_u32, pc_num_k_heads_ptr);
    let pc_num_v_heads_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_3]);
    let _num_v_heads = m.load(t_u32, pc_num_v_heads_ptr);
    let pc_head_k_dim_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_4]);
    let head_k_dim = m.load(t_u32, pc_head_k_dim_ptr);
    let pc_head_v_dim_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_5]);
    let head_v_dim = m.load(t_u32, pc_head_v_dim_ptr);
    let pc_eps_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_6]);
    let eps_bits = m.load(t_u32, pc_eps_ptr);
    let eps = m.bitcast(t_f32, eps_bits);

    let in_bounds = m.u_less_than(t_bool, gid, d_inner);
    m.selection_merge(lbl_end, 0);
    m.branch_conditional(in_bounds, lbl_body, lbl_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_body.0]));
    let vi = m.umod(t_u32, gid, head_v_dim);
    let h = m.udiv(t_u32, gid, head_v_dim);
    let kh = m.umod(t_u32, h, num_k_heads);
    let q_dim = m.imul(t_u32, num_k_heads, head_k_dim);
    let q_base = m.imul(t_u32, kh, head_k_dim);
    let k_base = m.iadd(t_u32, q_dim, q_base);
    let v_start = m.isub(t_u32, conv_channels, d_inner);
    let v_head_off = m.imul(t_u32, h, head_v_dim);
    let v_base = m.iadd(t_u32, v_start, v_head_off);
    let state_head_stride = m.imul(t_u32, head_v_dim, head_k_dim);
    let state_head_base = m.imul(t_u32, h, state_head_stride);
    let state_row_off = m.imul(t_u32, vi, head_k_dim);
    let state_base = m.iadd(t_u32, state_head_base, state_row_off);

    m.store(var_j, c_u32_0);
    m.store(var_q_sum, c_f32_0);
    m.store(var_k_sum, c_f32_0);
    m.branch(lbl_qk_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_qk_header.0]));
    m.loop_merge(lbl_qk_merge, lbl_qk_continue, 0);
    m.branch(lbl_qk_cond);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_qk_cond.0]));
    let j_qk = m.load(t_u32, var_j);
    let qk_cond = m.u_less_than(t_bool, j_qk, head_k_dim);
    m.branch_conditional(qk_cond, lbl_qk_body, lbl_qk_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_qk_body.0]));
    let q_idx = m.iadd(t_u32, q_base, j_qk);
    let k_idx = m.iadd(t_u32, k_base, j_qk);
    let q_ptr = m.access_chain(t_ptr_sb_f32, gvar_conv, &[c_u32_0, q_idx]);
    let k_ptr = m.access_chain(t_ptr_sb_f32, gvar_conv, &[c_u32_0, k_idx]);
    let q_raw = m.load(t_f32, q_ptr);
    let k_raw = m.load(t_f32, k_ptr);
    let q_sq = m.fmul(t_f32, q_raw, q_raw);
    let k_sq = m.fmul(t_f32, k_raw, k_raw);
    let q_sum_old = m.load(t_f32, var_q_sum);
    let k_sum_old = m.load(t_f32, var_k_sum);
    let q_sum_new = m.fadd(t_f32, q_sum_old, q_sq);
    let k_sum_new = m.fadd(t_f32, k_sum_old, k_sq);
    m.store(var_q_sum, q_sum_new);
    m.store(var_k_sum, k_sum_new);
    m.branch(lbl_qk_continue);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_qk_continue.0]));
    let j_qk_next_cur = m.load(t_u32, var_j);
    let j_qk_next = m.iadd(t_u32, j_qk_next_cur, c_u32_1);
    m.store(var_j, j_qk_next);
    m.branch(lbl_qk_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_qk_merge.0]));
    let q_sum_final = m.load(t_f32, var_q_sum);
    let k_sum_final = m.load(t_f32, var_k_sum);
    let q_denom_sq = m.fadd(t_f32, q_sum_final, eps);
    let k_denom_sq = m.fadd(t_f32, k_sum_final, eps);
    let q_denom = m.ext_inst(t_f32, glsl, 31, &[q_denom_sq]);
    let k_denom = m.ext_inst(t_f32, glsl, 31, &[k_denom_sq]);
    let q_inv_l2 = m.fdiv(t_f32, c_f32_1, q_denom);
    let k_inv = m.fdiv(t_f32, c_f32_1, k_denom);
    let head_k_f = m.convert_u_to_f(t_f32, head_k_dim);
    let head_k_sqrt = m.ext_inst(t_f32, glsl, 31, &[head_k_f]);
    let head_k_scale = m.fdiv(t_f32, c_f32_1, head_k_sqrt);
    let q_inv = m.fmul(t_f32, q_inv_l2, head_k_scale);

    let beta_ptr = m.access_chain(t_ptr_sb_f32, gvar_beta, &[c_u32_0, h]);
    let beta_raw = m.load(t_f32, beta_ptr);
    let neg_beta = m.fnegate(t_f32, beta_raw);
    let exp_neg_beta = m.ext_inst(t_f32, glsl, 27, &[neg_beta]);
    let beta_den = m.fadd(t_f32, c_f32_1, exp_neg_beta);
    let beta = m.fdiv(t_f32, c_f32_1, beta_den);

    let alpha_ptr = m.access_chain(t_ptr_sb_f32, gvar_alpha, &[c_u32_0, h]);
    let dt_bias_ptr = m.access_chain(t_ptr_sb_f32, gvar_dt_bias, &[c_u32_0, h]);
    let ssm_a_ptr = m.access_chain(t_ptr_sb_f32, gvar_ssm_a, &[c_u32_0, h]);
    let alpha_raw = m.load(t_f32, alpha_ptr);
    let dt_bias = m.load(t_f32, dt_bias_ptr);
    let ssm_a = m.load(t_f32, ssm_a_ptr);
    let alpha_biased = m.fadd(t_f32, alpha_raw, dt_bias);
    let exp_alpha = m.ext_inst(t_f32, glsl, 27, &[alpha_biased]);
    let softplus_arg = m.fadd(t_f32, c_f32_1, exp_alpha);
    let softplus = m.ext_inst(t_f32, glsl, 28, &[softplus_arg]);
    let gate = m.fmul(t_f32, softplus, ssm_a);
    let decay = m.ext_inst(t_f32, glsl, 27, &[gate]);

    m.store(var_j, c_u32_0);
    m.store(var_sk, c_f32_0);
    m.branch(lbl_sk_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_sk_header.0]));
    m.loop_merge(lbl_sk_merge, lbl_sk_continue, 0);
    m.branch(lbl_sk_cond);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_sk_cond.0]));
    let j_sk = m.load(t_u32, var_j);
    let sk_cond = m.u_less_than(t_bool, j_sk, head_k_dim);
    m.branch_conditional(sk_cond, lbl_sk_body, lbl_sk_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_sk_body.0]));
    let state_idx_sk = m.iadd(t_u32, state_base, j_sk);
    let state_ptr_sk = m.access_chain(t_ptr_sb_f32, gvar_state, &[c_u32_0, state_idx_sk]);
    let state_val_sk = m.load(t_f32, state_ptr_sk);
    let k_idx_sk = m.iadd(t_u32, k_base, j_sk);
    let k_ptr_sk = m.access_chain(t_ptr_sb_f32, gvar_conv, &[c_u32_0, k_idx_sk]);
    let k_raw_sk = m.load(t_f32, k_ptr_sk);
    let k_norm_sk = m.fmul(t_f32, k_raw_sk, k_inv);
    let sk_mul = m.fmul(t_f32, state_val_sk, k_norm_sk);
    let sk_old = m.load(t_f32, var_sk);
    let sk_new = m.fadd(t_f32, sk_old, sk_mul);
    m.store(var_sk, sk_new);
    m.branch(lbl_sk_continue);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_sk_continue.0]));
    let j_sk_next_cur = m.load(t_u32, var_j);
    let j_sk_next = m.iadd(t_u32, j_sk_next_cur, c_u32_1);
    m.store(var_j, j_sk_next);
    m.branch(lbl_sk_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_sk_merge.0]));
    let v_idx = m.iadd(t_u32, v_base, vi);
    let v_ptr = m.access_chain(t_ptr_sb_f32, gvar_conv, &[c_u32_0, v_idx]);
    let v_val = m.load(t_f32, v_ptr);
    let sk_final = m.load(t_f32, var_sk);
    let v_minus_sk = m.fsub(t_f32, v_val, sk_final);
    let d_val = m.fmul(t_f32, v_minus_sk, beta);

    m.store(var_j, c_u32_0);
    m.store(var_out, c_f32_0);
    m.branch(lbl_update_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_update_header.0]));
    m.loop_merge(lbl_update_merge, lbl_update_continue, 0);
    m.branch(lbl_update_cond);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_update_cond.0]));
    let j_up = m.load(t_u32, var_j);
    let update_cond = m.u_less_than(t_bool, j_up, head_k_dim);
    m.branch_conditional(update_cond, lbl_update_body, lbl_update_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_update_body.0]));
    let state_idx_up = m.iadd(t_u32, state_base, j_up);
    let state_ptr_up = m.access_chain(t_ptr_sb_f32, gvar_state, &[c_u32_0, state_idx_up]);
    let state_old = m.load(t_f32, state_ptr_up);
    let k_idx_up = m.iadd(t_u32, k_base, j_up);
    let q_idx_up = m.iadd(t_u32, q_base, j_up);
    let k_ptr_up = m.access_chain(t_ptr_sb_f32, gvar_conv, &[c_u32_0, k_idx_up]);
    let q_ptr_up = m.access_chain(t_ptr_sb_f32, gvar_conv, &[c_u32_0, q_idx_up]);
    let k_raw_up = m.load(t_f32, k_ptr_up);
    let q_raw_up = m.load(t_f32, q_ptr_up);
    let k_norm_up = m.fmul(t_f32, k_raw_up, k_inv);
    let q_scaled = m.fmul(t_f32, q_raw_up, q_inv);
    let decay_state = m.fmul(t_f32, decay, state_old);
    let kd = m.fmul(t_f32, k_norm_up, d_val);
    let state_new = m.fadd(t_f32, decay_state, kd);
    m.store(state_ptr_up, state_new);
    let out_part = m.fmul(t_f32, state_new, q_scaled);
    let out_old = m.load(t_f32, var_out);
    let out_new = m.fadd(t_f32, out_old, out_part);
    m.store(var_out, out_new);
    m.branch(lbl_update_continue);

    m.functions.push(SpirvModule::encode_inst(
        op::LABEL,
        &[lbl_update_continue.0],
    ));
    let j_up_next_cur = m.load(t_u32, var_j);
    let j_up_next = m.iadd(t_u32, j_up_next_cur, c_u32_1);
    m.store(var_j, j_up_next);
    m.branch(lbl_update_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_update_merge.0]));
    let out_final = m.load(t_f32, var_out);
    let out_ptr = m.access_chain(t_ptr_sb_f32, gvar_out, &[c_u32_0, gid]);
    m.store(out_ptr, out_final);
    m.branch(lbl_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_end.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// Emit SPIR-V for RMSNorm: out[i] = (x[i] / sqrt(mean_sq + eps)) * weight[i]
///
/// Uses workgroup shared memory for parallel reduction.
/// Dispatch: 1 workgroup of local_size_x threads.
///
/// Layout:
///   binding 0: input (f32, read), binding 1: weight (f32, read), binding 2: output (f32, write)
///   push constant: dim (u32), eps_bits (u32 bitcast of f32)
pub fn emit_rms_norm(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    let glsl = m.ext_inst_import("GLSL.std.450");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_arr_f32 = m.type_runtime_array(t_f32);

    let t_struct_in = m.type_struct(&[t_arr_f32]);
    let t_struct_wt = m.type_struct(&[t_arr_f32]);
    let t_struct_out = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32]);

    let c_local_size = m.constant_u32(t_u32, local_size_x);
    let t_shared_arr = m.type_array(t_f32, c_local_size);
    let t_ptr_wg_arr = m.type_pointer(storage_class::WORKGROUP, t_shared_arr);
    let t_ptr_wg_f32 = m.type_pointer(storage_class::WORKGROUP, t_f32);

    let t_ptr_sb_in = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_in);
    let t_ptr_sb_wt = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_wt);
    let t_ptr_sb_out = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_out);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);
    let c_f32_1 = m.constant_f32(t_f32, 1.0);

    m.decorate(t_struct_in, decoration::BLOCK, &[]);
    m.decorate(t_struct_wt, decoration::BLOCK, &[]);
    m.decorate(t_struct_out, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_in, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_wt, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_out, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);

    let gvar_in = m.variable(t_ptr_sb_in, storage_class::STORAGE_BUFFER);
    let gvar_wt = m.variable(t_ptr_sb_wt, storage_class::STORAGE_BUFFER);
    let gvar_out = m.variable(t_ptr_sb_out, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let gvar_lid = m.variable(t_ptr_input_v3, storage_class::INPUT);
    let gvar_shared = m.variable(t_ptr_wg_arr, storage_class::WORKGROUP);

    m.decorate(gvar_in, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_in, decoration::BINDING, &[0]);
    m.decorate(gvar_wt, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_wt, decoration::BINDING, &[1]);
    m.decorate(gvar_out, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_out, decoration::BINDING, &[2]);
    m.decorate(
        gvar_lid,
        decoration::BUILTIN,
        &[builtin::LOCAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_lid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    m.function(t_void, func_id, 0, t_fn_void);
    let lbl_entry = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    // All function variables at top of first block
    let var_i = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_s = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);

    let lid_vec = m.load(t_v3u32, gvar_lid);
    let lid = m.composite_extract(t_u32, lid_vec, 0);

    let pc_dim_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let dim = m.load(t_u32, pc_dim_ptr);
    let pc_eps_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let eps_bits = m.load(t_u32, pc_eps_ptr);
    let eps = m.bitcast(t_f32, eps_bits);

    // ---- Loop 1: accumulate x² ----
    m.store(var_i, lid);
    m.store(var_sum, c_f32_0);

    let lbl_l1h = m.alloc_id(); // header
    let lbl_l1c_blk = m.alloc_id(); // condition block
    let lbl_l1b = m.alloc_id(); // body
    let lbl_l1cont = m.alloc_id(); // continue
    let lbl_l1m = m.alloc_id(); // merge

    m.branch(lbl_l1h);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_l1h.0]));
    m.loop_merge(lbl_l1m, lbl_l1cont, 0);
    m.branch(lbl_l1c_blk); // unconditional to condition

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_l1c_blk.0]));
    let iv = m.load(t_u32, var_i);
    let cond1 = m.u_less_than(t_bool, iv, dim);
    m.branch_conditional(cond1, lbl_l1b, lbl_l1m);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_l1b.0]));
    let xp = m.access_chain(t_ptr_sb_f32, gvar_in, &[c_u32_0, iv]);
    let xv = m.load(t_f32, xp);
    let xsq = m.fmul(t_f32, xv, xv);
    let os = m.load(t_f32, var_sum);
    let ns = m.fadd(t_f32, os, xsq);
    m.store(var_sum, ns);
    m.branch(lbl_l1cont);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_l1cont.0]));
    let inx = m.iadd(t_u32, iv, c_local_size);
    m.store(var_i, inx);
    m.branch(lbl_l1h);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_l1m.0]));

    // Store to shared + barrier
    let sp = m.access_chain(t_ptr_wg_f32, gvar_shared, &[lid]);
    let fp = m.load(t_f32, var_sum);
    m.store(sp, fp);
    {
        let c_scope_wg = m.constant_u32(t_u32, scope::WORKGROUP);
        let c_sem = m.constant_u32(
            t_u32,
            memory_semantics::WORKGROUP_MEMORY | memory_semantics::ACQUIRE_RELEASE,
        );
        m.control_barrier(c_scope_wg, c_scope_wg, c_sem);
    }

    // ---- Loop 2: parallel reduction ----
    let c_half = m.constant_u32(t_u32, local_size_x / 2);
    m.store(var_s, c_half);

    let lbl_r_h = m.alloc_id();
    let lbl_r_c = m.alloc_id(); // condition
    let lbl_r_b = m.alloc_id();
    let lbl_r_cont = m.alloc_id();
    let lbl_r_m = m.alloc_id();

    m.branch(lbl_r_h);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_h.0]));
    m.loop_merge(lbl_r_m, lbl_r_cont, 0);
    m.branch(lbl_r_c);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_c.0]));
    let sv = m.load(t_u32, var_s);
    let sg = m.u_less_than(t_bool, c_u32_0, sv);
    m.branch_conditional(sg, lbl_r_b, lbl_r_m);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_b.0]));
    let ls = m.u_less_than(t_bool, lid, sv);
    let lbl_a = m.alloc_id();
    let lbl_am = m.alloc_id();
    m.selection_merge(lbl_am, 0);
    m.branch_conditional(ls, lbl_a, lbl_am);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_a.0]));
    let lps = m.iadd(t_u32, lid, sv);
    let sap = m.access_chain(t_ptr_wg_f32, gvar_shared, &[lid]);
    let sbp = m.access_chain(t_ptr_wg_f32, gvar_shared, &[lps]);
    let sav = m.load(t_f32, sap);
    let sbv = m.load(t_f32, sbp);
    let ssm = m.fadd(t_f32, sav, sbv);
    m.store(sap, ssm);
    m.branch(lbl_am);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_am.0]));
    {
        let c_scope_wg = m.constant_u32(t_u32, scope::WORKGROUP);
        let c_sem = m.constant_u32(
            t_u32,
            memory_semantics::WORKGROUP_MEMORY | memory_semantics::ACQUIRE_RELEASE,
        );
        m.control_barrier(c_scope_wg, c_scope_wg, c_sem);
    }
    m.branch(lbl_r_cont);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_cont.0]));
    let sh = m.shift_right_logical(t_u32, sv, c_u32_1);
    m.store(var_s, sh);
    m.branch(lbl_r_h);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_m.0]));

    // ---- Step 3: inv_rms ----
    let s0p = m.access_chain(t_ptr_wg_f32, gvar_shared, &[c_u32_0]);
    let tsq = m.load(t_f32, s0p);
    let df = m.convert_u_to_f(t_f32, dim);
    let msq = m.fdiv(t_f32, tsq, df);
    let mse = m.fadd(t_f32, msq, eps);
    let sqr = m.ext_inst(t_f32, glsl, 31, &[mse]); // Sqrt
    let irm = m.fdiv(t_f32, c_f32_1, sqr);

    // ---- Loop 3: normalize ----
    m.store(var_i, lid);

    let lbl_n_h = m.alloc_id();
    let lbl_n_c = m.alloc_id();
    let lbl_n_b = m.alloc_id();
    let lbl_n_cont = m.alloc_id();
    let lbl_n_m = m.alloc_id();

    m.branch(lbl_n_h);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_n_h.0]));
    m.loop_merge(lbl_n_m, lbl_n_cont, 0);
    m.branch(lbl_n_c);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_n_c.0]));
    let i2 = m.load(t_u32, var_i);
    let c2 = m.u_less_than(t_bool, i2, dim);
    m.branch_conditional(c2, lbl_n_b, lbl_n_m);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_n_b.0]));
    let ip = m.access_chain(t_ptr_sb_f32, gvar_in, &[c_u32_0, i2]);
    let iv2 = m.load(t_f32, ip);
    let wp = m.access_chain(t_ptr_sb_f32, gvar_wt, &[c_u32_0, i2]);
    let wv = m.load(t_f32, wp);
    let sc = m.fmul(t_f32, iv2, irm);
    let nm = m.fmul(t_f32, sc, wv);
    let op2 = m.access_chain(t_ptr_sb_f32, gvar_out, &[c_u32_0, i2]);
    m.store(op2, nm);
    m.branch(lbl_n_cont);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_n_cont.0]));
    let i2n = m.iadd(t_u32, i2, c_local_size);
    m.store(var_i, i2n);
    m.branch(lbl_n_h);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_n_m.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// Emit SPIR-V for depthwise conv1d + SiLU over a single token window.
///
/// Layout:
///   binding 0: kernel buffer (f32, read)  [kernel_size, channels]
///   binding 1: input window (f32, read)   [kernel_size, channels]
///   binding 2: output buffer (f32, write) [channels]
///   push constants: channels (u32), kernel_size (u32)
pub fn emit_depthwise_conv1d_silu(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    let glsl = m.ext_inst_import("GLSL.std.450");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_arr_f32 = m.type_runtime_array(t_f32);

    let t_struct_kernel = m.type_struct(&[t_arr_f32]);
    let t_struct_input = m.type_struct(&[t_arr_f32]);
    let t_struct_output = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32]);

    let t_ptr_sb_kernel = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_kernel);
    let t_ptr_sb_input = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_input);
    let t_ptr_sb_output = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_output);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);
    let c_f32_1 = m.constant_f32(t_f32, 1.0);

    for t in [t_struct_kernel, t_struct_input, t_struct_output] {
        m.decorate(t, decoration::BLOCK, &[]);
        m.member_decorate(t, 0, decoration::OFFSET, &[0]);
    }
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);

    let gvar_kernel = m.variable(t_ptr_sb_kernel, storage_class::STORAGE_BUFFER);
    let gvar_input = m.variable(t_ptr_sb_input, storage_class::STORAGE_BUFFER);
    let gvar_output = m.variable(t_ptr_sb_output, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let gvar_gid = m.variable(t_ptr_input_v3, storage_class::INPUT);

    m.decorate(gvar_kernel, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_kernel, decoration::BINDING, &[0]);
    m.decorate(gvar_input, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_input, decoration::BINDING, &[1]);
    m.decorate(gvar_output, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_output, decoration::BINDING, &[2]);
    m.decorate(
        gvar_gid,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_gid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    m.function(t_void, func_id, 0, t_fn_void);
    let lbl_entry = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let var_k = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);

    let gid_vec = m.load(t_v3u32, gvar_gid);
    let gid = m.composite_extract(t_u32, gid_vec, 0);

    let pc_channels_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let channels = m.load(t_u32, pc_channels_ptr);
    let pc_kernel_size_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let kernel_size = m.load(t_u32, pc_kernel_size_ptr);

    let in_bounds = m.u_less_than(t_bool, gid, channels);
    let lbl_body = m.alloc_id();
    let lbl_end = m.alloc_id();
    m.selection_merge(lbl_end, 0);
    m.branch_conditional(in_bounds, lbl_body, lbl_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_body.0]));

    m.store(var_k, c_u32_0);
    m.store(var_sum, c_f32_0);

    let lbl_loop_h = m.alloc_id();
    let lbl_loop_c = m.alloc_id();
    let lbl_loop_b = m.alloc_id();
    let lbl_loop_cont = m.alloc_id();
    let lbl_loop_m = m.alloc_id();

    m.branch(lbl_loop_h);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_h.0]));
    m.loop_merge(lbl_loop_m, lbl_loop_cont, 0);
    m.branch(lbl_loop_c);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_c.0]));
    let k = m.load(t_u32, var_k);
    let cond = m.u_less_than(t_bool, k, kernel_size);
    m.branch_conditional(cond, lbl_loop_b, lbl_loop_m);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_b.0]));
    let base = m.imul(t_u32, k, channels);
    let idx = m.iadd(t_u32, base, gid);
    let k_ptr = m.access_chain(t_ptr_sb_f32, gvar_kernel, &[c_u32_0, idx]);
    let w = m.load(t_f32, k_ptr);
    let x_ptr = m.access_chain(t_ptr_sb_f32, gvar_input, &[c_u32_0, idx]);
    let x = m.load(t_f32, x_ptr);
    let prod = m.fmul(t_f32, w, x);
    let acc = m.load(t_f32, var_sum);
    let acc_new = m.fadd(t_f32, acc, prod);
    m.store(var_sum, acc_new);
    m.branch(lbl_loop_cont);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_cont.0]));
    let k_next = m.iadd(t_u32, k, c_u32_1);
    m.store(var_k, k_next);
    m.branch(lbl_loop_h);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_loop_m.0]));

    let sum = m.load(t_f32, var_sum);
    let neg_sum = m.fnegate(t_f32, sum);
    let exp_neg = m.ext_inst(t_f32, glsl, 27, &[neg_sum]); // GLSL Exp = 27
    let denom = m.fadd(t_f32, c_f32_1, exp_neg);
    let silu = m.fdiv(t_f32, sum, denom);

    let out_ptr = m.access_chain(t_ptr_sb_f32, gvar_output, &[c_u32_0, gid]);
    m.store(out_ptr, silu);
    m.branch(lbl_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_end.0]));
    m.ret();
    m.function_end();

    m.encode()
}

pub fn emit_attention_decode(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    let glsl = m.ext_inst_import("GLSL.std.450");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_arr_f32 = m.type_runtime_array(t_f32);

    let t_struct_q = m.type_struct(&[t_arr_f32]);
    let t_struct_k = m.type_struct(&[t_arr_f32]);
    let t_struct_v = m.type_struct(&[t_arr_f32]);
    let t_struct_out = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);

    let t_ptr_sb_q = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_q);
    let t_ptr_sb_k = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_k);
    let t_ptr_sb_v = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_v);
    let t_ptr_sb_out = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_out);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_4 = m.constant_u32(t_u32, 4);
    // mv39: NEON 8-elem STEP 정확 emulate 위해 c_u32_5..8 추가.
    let c_u32_5 = m.constant_u32(t_u32, 5);
    let c_u32_6 = m.constant_u32(t_u32, 6);
    let c_u32_7 = m.constant_u32(t_u32, 7);
    let c_u32_8 = m.constant_u32(t_u32, 8);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);

    for t in [t_struct_q, t_struct_k, t_struct_v, t_struct_out] {
        m.decorate(t, decoration::BLOCK, &[]);
        m.member_decorate(t, 0, decoration::OFFSET, &[0]);
    }
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);

    let gvar_q = m.variable(t_ptr_sb_q, storage_class::STORAGE_BUFFER);
    let gvar_k = m.variable(t_ptr_sb_k, storage_class::STORAGE_BUFFER);
    let gvar_v = m.variable(t_ptr_sb_v, storage_class::STORAGE_BUFFER);
    let gvar_out = m.variable(t_ptr_sb_out, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let gvar_gid = m.variable(t_ptr_input_v3, storage_class::INPUT);

    for (var, binding) in [(gvar_q, 0), (gvar_k, 1), (gvar_v, 2), (gvar_out, 3)] {
        m.decorate(var, decoration::DESCRIPTOR_SET, &[0]);
        m.decorate(var, decoration::BINDING, &[binding]);
    }
    m.decorate(
        gvar_gid,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_gid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    m.function(t_void, func_id, 0, t_fn_void);
    let lbl_entry = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let var_i = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_j = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let _var_dot = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_num = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_den = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_max = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    // mv39: NEON 의 acc0 (4-lane) + acc1 (4-lane) = 8 indep scalar acc 정확 emulate.
    // chunk = 8 (NEON neon_dot_f32 의 8-elem STEP). acc[0..3] = NEON acc0.lane[0..3],
    // acc[4..7] = NEON acc1.lane[0..3]. ARM `vfmaq_f32 + vaddq_f32 + vaddvq_f32`
    // 의 fp accumulation 순서/정밀도 정확 일치.
    let var_acc0 = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_acc1 = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_acc2 = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_acc3 = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_acc4 = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_acc5 = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_acc6 = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_acc7 = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);

    let gid_vec = m.load(t_v3u32, gvar_gid);
    let gid = m.composite_extract(t_u32, gid_vec, 0);

    let pc_head_dim_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let head_dim = m.load(t_u32, pc_head_dim_ptr);
    let pc_kv_len_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let kv_len = m.load(t_u32, pc_kv_len_ptr);
    let pc_scale_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_2]);
    let scale_bits = m.load(t_u32, pc_scale_ptr);
    let scale = m.bitcast(t_f32, scale_bits);

    let gid_ok = m.u_less_than(t_bool, gid, head_dim);
    let lbl_gid_ok = m.alloc_id();
    let lbl_gid_end = m.alloc_id();
    m.selection_merge(lbl_gid_end, 0);
    m.branch_conditional(gid_ok, lbl_gid_ok, lbl_gid_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_gid_ok.0]));

    let c_f32_neg_large = m.constant_f32(t_f32, -1.0e30);
    m.store(var_j, c_u32_0);
    m.store(var_max, c_f32_neg_large);

    let lbl_mx_h = m.alloc_id();
    let lbl_mx_c = m.alloc_id();
    let lbl_mx_b = m.alloc_id();
    let lbl_mx_cont = m.alloc_id();
    let lbl_mx_merge = m.alloc_id();
    m.branch(lbl_mx_h);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_mx_h.0]));
    m.loop_merge(lbl_mx_merge, lbl_mx_cont, 0);
    m.branch(lbl_mx_c);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_mx_c.0]));
    let jv = m.load(t_u32, var_j);
    let kv_cond = m.u_less_than(t_bool, jv, kv_len);
    m.branch_conditional(kv_cond, lbl_mx_b, lbl_mx_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_mx_b.0]));

    m.store(var_i, c_u32_0);
    // mv39: 8-acc init (Pass 1) — NEON 의 acc0/acc1 4-lane vec 정확 emulate.
    m.store(var_acc0, c_f32_0);
    m.store(var_acc1, c_f32_0);
    m.store(var_acc2, c_f32_0);
    m.store(var_acc3, c_f32_0);
    m.store(var_acc4, c_f32_0);
    m.store(var_acc5, c_f32_0);
    m.store(var_acc6, c_f32_0);
    m.store(var_acc7, c_f32_0);

    let lbl_h = m.alloc_id();
    let lbl_c = m.alloc_id();
    let lbl_b = m.alloc_id();
    let lbl_cont = m.alloc_id();
    let lbl_m = m.alloc_id();
    m.branch(lbl_h);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_h.0]));
    m.loop_merge(lbl_m, lbl_cont, 0);
    m.branch(lbl_c);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_c.0]));
    let iv = m.load(t_u32, var_i);
    let cond = m.u_less_than(t_bool, iv, head_dim);
    m.branch_conditional(cond, lbl_b, lbl_m);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_b.0]));
    // mv39: NEON neon_dot_f32 의 8-elem STEP + 2-way 4-lane SIMD acc 정확 emulate.
    // 8 fma per iter. acc[0..3] = NEON acc0.lane[0..3], acc[4..7] = NEON acc1.lane[0..3].
    // head_dim must be multiple of 8 (Qwen3.5/3.6 head_dim=256 OK; 64/128 도 OK).
    let k_base = m.imul(t_u32, jv, head_dim);
    let i_p1 = m.iadd(t_u32, iv, c_u32_1);
    let i_p2 = m.iadd(t_u32, iv, c_u32_2);
    let i_p3 = m.iadd(t_u32, iv, c_u32_3);
    let i_p4 = m.iadd(t_u32, iv, c_u32_4);
    let i_p5 = m.iadd(t_u32, iv, c_u32_5);
    let i_p6 = m.iadd(t_u32, iv, c_u32_6);
    let i_p7 = m.iadd(t_u32, iv, c_u32_7);
    let q_ptr_0 = m.access_chain(t_ptr_sb_f32, gvar_q, &[c_u32_0, iv]);
    let q_val_0 = m.load(t_f32, q_ptr_0);
    let q_ptr_1 = m.access_chain(t_ptr_sb_f32, gvar_q, &[c_u32_0, i_p1]);
    let q_val_1 = m.load(t_f32, q_ptr_1);
    let q_ptr_2 = m.access_chain(t_ptr_sb_f32, gvar_q, &[c_u32_0, i_p2]);
    let q_val_2 = m.load(t_f32, q_ptr_2);
    let q_ptr_3 = m.access_chain(t_ptr_sb_f32, gvar_q, &[c_u32_0, i_p3]);
    let q_val_3 = m.load(t_f32, q_ptr_3);
    let q_ptr_4 = m.access_chain(t_ptr_sb_f32, gvar_q, &[c_u32_0, i_p4]);
    let q_val_4 = m.load(t_f32, q_ptr_4);
    let q_ptr_5 = m.access_chain(t_ptr_sb_f32, gvar_q, &[c_u32_0, i_p5]);
    let q_val_5 = m.load(t_f32, q_ptr_5);
    let q_ptr_6 = m.access_chain(t_ptr_sb_f32, gvar_q, &[c_u32_0, i_p6]);
    let q_val_6 = m.load(t_f32, q_ptr_6);
    let q_ptr_7 = m.access_chain(t_ptr_sb_f32, gvar_q, &[c_u32_0, i_p7]);
    let q_val_7 = m.load(t_f32, q_ptr_7);
    let k_idx_0 = m.iadd(t_u32, k_base, iv);
    let k_idx_1 = m.iadd(t_u32, k_base, i_p1);
    let k_idx_2 = m.iadd(t_u32, k_base, i_p2);
    let k_idx_3 = m.iadd(t_u32, k_base, i_p3);
    let k_idx_4 = m.iadd(t_u32, k_base, i_p4);
    let k_idx_5 = m.iadd(t_u32, k_base, i_p5);
    let k_idx_6 = m.iadd(t_u32, k_base, i_p6);
    let k_idx_7 = m.iadd(t_u32, k_base, i_p7);
    let k_ptr_0 = m.access_chain(t_ptr_sb_f32, gvar_k, &[c_u32_0, k_idx_0]);
    let k_val_0 = m.load(t_f32, k_ptr_0);
    let k_ptr_1 = m.access_chain(t_ptr_sb_f32, gvar_k, &[c_u32_0, k_idx_1]);
    let k_val_1 = m.load(t_f32, k_ptr_1);
    let k_ptr_2 = m.access_chain(t_ptr_sb_f32, gvar_k, &[c_u32_0, k_idx_2]);
    let k_val_2 = m.load(t_f32, k_ptr_2);
    let k_ptr_3 = m.access_chain(t_ptr_sb_f32, gvar_k, &[c_u32_0, k_idx_3]);
    let k_val_3 = m.load(t_f32, k_ptr_3);
    let k_ptr_4 = m.access_chain(t_ptr_sb_f32, gvar_k, &[c_u32_0, k_idx_4]);
    let k_val_4 = m.load(t_f32, k_ptr_4);
    let k_ptr_5 = m.access_chain(t_ptr_sb_f32, gvar_k, &[c_u32_0, k_idx_5]);
    let k_val_5 = m.load(t_f32, k_ptr_5);
    let k_ptr_6 = m.access_chain(t_ptr_sb_f32, gvar_k, &[c_u32_0, k_idx_6]);
    let k_val_6 = m.load(t_f32, k_ptr_6);
    let k_ptr_7 = m.access_chain(t_ptr_sb_f32, gvar_k, &[c_u32_0, k_idx_7]);
    let k_val_7 = m.load(t_f32, k_ptr_7);
    // mv39: 8 indep FMA — NEON 의 acc0 (4-lane) + acc1 (4-lane) 의 같은 lane 누적.
    let acc0 = m.load(t_f32, var_acc0);
    let acc1 = m.load(t_f32, var_acc1);
    let acc2 = m.load(t_f32, var_acc2);
    let acc3 = m.load(t_f32, var_acc3);
    let acc4 = m.load(t_f32, var_acc4);
    let acc5 = m.load(t_f32, var_acc5);
    let acc6 = m.load(t_f32, var_acc6);
    let acc7 = m.load(t_f32, var_acc7);
    let acc0_n = m.ext_inst(t_f32, glsl, 50, &[q_val_0, k_val_0, acc0]);
    let acc1_n = m.ext_inst(t_f32, glsl, 50, &[q_val_1, k_val_1, acc1]);
    let acc2_n = m.ext_inst(t_f32, glsl, 50, &[q_val_2, k_val_2, acc2]);
    let acc3_n = m.ext_inst(t_f32, glsl, 50, &[q_val_3, k_val_3, acc3]);
    let acc4_n = m.ext_inst(t_f32, glsl, 50, &[q_val_4, k_val_4, acc4]);
    let acc5_n = m.ext_inst(t_f32, glsl, 50, &[q_val_5, k_val_5, acc5]);
    let acc6_n = m.ext_inst(t_f32, glsl, 50, &[q_val_6, k_val_6, acc6]);
    let acc7_n = m.ext_inst(t_f32, glsl, 50, &[q_val_7, k_val_7, acc7]);
    m.store(var_acc0, acc0_n);
    m.store(var_acc1, acc1_n);
    m.store(var_acc2, acc2_n);
    m.store(var_acc3, acc3_n);
    m.store(var_acc4, acc4_n);
    m.store(var_acc5, acc5_n);
    m.store(var_acc6, acc6_n);
    m.store(var_acc7, acc7_n);
    m.branch(lbl_cont);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_cont.0]));
    let iv_next = m.iadd(t_u32, iv, c_u32_8);
    m.store(var_i, iv_next);
    m.branch(lbl_h);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_m.0]));
    // mv39: NEON `vaddq_f32(acc0, acc1) → vaddvq_f32` 정확 emulate.
    // s.lane[k] = acc[k] + acc[k+4] (vaddq_f32 lane-wise add)
    // result = ((s.l0+s.l1) + (s.l2+s.l3)) (vaddvq_f32 pairwise tree)
    let final_a0 = m.load(t_f32, var_acc0);
    let final_a1 = m.load(t_f32, var_acc1);
    let final_a2 = m.load(t_f32, var_acc2);
    let final_a3 = m.load(t_f32, var_acc3);
    let final_a4 = m.load(t_f32, var_acc4);
    let final_a5 = m.load(t_f32, var_acc5);
    let final_a6 = m.load(t_f32, var_acc6);
    let final_a7 = m.load(t_f32, var_acc7);
    // NEON vaddq_f32: s.l[k] = acc0.l[k] + acc1.l[k]
    let s_l0 = m.fadd(t_f32, final_a0, final_a4);
    let s_l1 = m.fadd(t_f32, final_a1, final_a5);
    let s_l2 = m.fadd(t_f32, final_a2, final_a6);
    let s_l3 = m.fadd(t_f32, final_a3, final_a7);
    // NEON vaddvq_f32: ((s.l0+s.l1) + (s.l2+s.l3))
    let final_s01 = m.fadd(t_f32, s_l0, s_l1);
    let final_s23 = m.fadd(t_f32, s_l2, s_l3);
    let final_dot = m.fadd(t_f32, final_s01, final_s23);
    let scaled = m.fmul(t_f32, final_dot, scale);
    let cur_max = m.load(t_f32, var_max);
    let max_gt = m.f_ord_greater_than(t_bool, scaled, cur_max);
    let next_max = m.select(t_f32, max_gt, scaled, cur_max);
    m.store(var_max, next_max);
    m.branch(lbl_mx_cont);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_mx_cont.0]));
    let jn = m.iadd(t_u32, jv, c_u32_1);
    m.store(var_j, jn);
    m.branch(lbl_mx_h);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_mx_merge.0]));

    m.store(var_j, c_u32_0);
    m.store(var_num, c_f32_0);
    m.store(var_den, c_f32_0);

    let lbl_kv_h = m.alloc_id();
    let lbl_kv_c = m.alloc_id();
    let lbl_kv_b = m.alloc_id();
    let lbl_kv_cont = m.alloc_id();
    let lbl_kv_m = m.alloc_id();
    m.branch(lbl_kv_h);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_kv_h.0]));
    m.loop_merge(lbl_kv_m, lbl_kv_cont, 0);
    m.branch(lbl_kv_c);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_kv_c.0]));
    let jv2 = m.load(t_u32, var_j);
    let kv_cond2 = m.u_less_than(t_bool, jv2, kv_len);
    m.branch_conditional(kv_cond2, lbl_kv_b, lbl_kv_m);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_kv_b.0]));

    m.store(var_i, c_u32_0);
    // mv39: 8-acc init (Pass 2) — NEON acc0/acc1 4-lane vec emulate.
    m.store(var_acc0, c_f32_0);
    m.store(var_acc1, c_f32_0);
    m.store(var_acc2, c_f32_0);
    m.store(var_acc3, c_f32_0);
    m.store(var_acc4, c_f32_0);
    m.store(var_acc5, c_f32_0);
    m.store(var_acc6, c_f32_0);
    m.store(var_acc7, c_f32_0);

    let lbl_h2 = m.alloc_id();
    let lbl_c2 = m.alloc_id();
    let lbl_b2 = m.alloc_id();
    let lbl_cont2 = m.alloc_id();
    let lbl_m2 = m.alloc_id();
    m.branch(lbl_h2);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_h2.0]));
    m.loop_merge(lbl_m2, lbl_cont2, 0);
    m.branch(lbl_c2);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_c2.0]));
    let iv2 = m.load(t_u32, var_i);
    let cond2 = m.u_less_than(t_bool, iv2, head_dim);
    m.branch_conditional(cond2, lbl_b2, lbl_m2);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_b2.0]));
    // mv39: NEON 8-elem STEP + 8 indep acc Pass 2 — same pattern as Pass 1.
    let k_base2 = m.imul(t_u32, jv2, head_dim);
    let i2_p1 = m.iadd(t_u32, iv2, c_u32_1);
    let i2_p2 = m.iadd(t_u32, iv2, c_u32_2);
    let i2_p3 = m.iadd(t_u32, iv2, c_u32_3);
    let i2_p4 = m.iadd(t_u32, iv2, c_u32_4);
    let i2_p5 = m.iadd(t_u32, iv2, c_u32_5);
    let i2_p6 = m.iadd(t_u32, iv2, c_u32_6);
    let i2_p7 = m.iadd(t_u32, iv2, c_u32_7);
    let q_ptr2_0 = m.access_chain(t_ptr_sb_f32, gvar_q, &[c_u32_0, iv2]);
    let q_val2_0 = m.load(t_f32, q_ptr2_0);
    let q_ptr2_1 = m.access_chain(t_ptr_sb_f32, gvar_q, &[c_u32_0, i2_p1]);
    let q_val2_1 = m.load(t_f32, q_ptr2_1);
    let q_ptr2_2 = m.access_chain(t_ptr_sb_f32, gvar_q, &[c_u32_0, i2_p2]);
    let q_val2_2 = m.load(t_f32, q_ptr2_2);
    let q_ptr2_3 = m.access_chain(t_ptr_sb_f32, gvar_q, &[c_u32_0, i2_p3]);
    let q_val2_3 = m.load(t_f32, q_ptr2_3);
    let q_ptr2_4 = m.access_chain(t_ptr_sb_f32, gvar_q, &[c_u32_0, i2_p4]);
    let q_val2_4 = m.load(t_f32, q_ptr2_4);
    let q_ptr2_5 = m.access_chain(t_ptr_sb_f32, gvar_q, &[c_u32_0, i2_p5]);
    let q_val2_5 = m.load(t_f32, q_ptr2_5);
    let q_ptr2_6 = m.access_chain(t_ptr_sb_f32, gvar_q, &[c_u32_0, i2_p6]);
    let q_val2_6 = m.load(t_f32, q_ptr2_6);
    let q_ptr2_7 = m.access_chain(t_ptr_sb_f32, gvar_q, &[c_u32_0, i2_p7]);
    let q_val2_7 = m.load(t_f32, q_ptr2_7);
    let k_idx2_0 = m.iadd(t_u32, k_base2, iv2);
    let k_idx2_1 = m.iadd(t_u32, k_base2, i2_p1);
    let k_idx2_2 = m.iadd(t_u32, k_base2, i2_p2);
    let k_idx2_3 = m.iadd(t_u32, k_base2, i2_p3);
    let k_idx2_4 = m.iadd(t_u32, k_base2, i2_p4);
    let k_idx2_5 = m.iadd(t_u32, k_base2, i2_p5);
    let k_idx2_6 = m.iadd(t_u32, k_base2, i2_p6);
    let k_idx2_7 = m.iadd(t_u32, k_base2, i2_p7);
    let k_ptr2_0 = m.access_chain(t_ptr_sb_f32, gvar_k, &[c_u32_0, k_idx2_0]);
    let k_val2_0 = m.load(t_f32, k_ptr2_0);
    let k_ptr2_1 = m.access_chain(t_ptr_sb_f32, gvar_k, &[c_u32_0, k_idx2_1]);
    let k_val2_1 = m.load(t_f32, k_ptr2_1);
    let k_ptr2_2 = m.access_chain(t_ptr_sb_f32, gvar_k, &[c_u32_0, k_idx2_2]);
    let k_val2_2 = m.load(t_f32, k_ptr2_2);
    let k_ptr2_3 = m.access_chain(t_ptr_sb_f32, gvar_k, &[c_u32_0, k_idx2_3]);
    let k_val2_3 = m.load(t_f32, k_ptr2_3);
    let k_ptr2_4 = m.access_chain(t_ptr_sb_f32, gvar_k, &[c_u32_0, k_idx2_4]);
    let k_val2_4 = m.load(t_f32, k_ptr2_4);
    let k_ptr2_5 = m.access_chain(t_ptr_sb_f32, gvar_k, &[c_u32_0, k_idx2_5]);
    let k_val2_5 = m.load(t_f32, k_ptr2_5);
    let k_ptr2_6 = m.access_chain(t_ptr_sb_f32, gvar_k, &[c_u32_0, k_idx2_6]);
    let k_val2_6 = m.load(t_f32, k_ptr2_6);
    let k_ptr2_7 = m.access_chain(t_ptr_sb_f32, gvar_k, &[c_u32_0, k_idx2_7]);
    let k_val2_7 = m.load(t_f32, k_ptr2_7);
    // mv39: 8 indep FMA — NEON acc0/acc1 lane 별 누적.
    let acc20 = m.load(t_f32, var_acc0);
    let acc21 = m.load(t_f32, var_acc1);
    let acc22 = m.load(t_f32, var_acc2);
    let acc23 = m.load(t_f32, var_acc3);
    let acc24 = m.load(t_f32, var_acc4);
    let acc25 = m.load(t_f32, var_acc5);
    let acc26 = m.load(t_f32, var_acc6);
    let acc27 = m.load(t_f32, var_acc7);
    let acc20_n = m.ext_inst(t_f32, glsl, 50, &[q_val2_0, k_val2_0, acc20]);
    let acc21_n = m.ext_inst(t_f32, glsl, 50, &[q_val2_1, k_val2_1, acc21]);
    let acc22_n = m.ext_inst(t_f32, glsl, 50, &[q_val2_2, k_val2_2, acc22]);
    let acc23_n = m.ext_inst(t_f32, glsl, 50, &[q_val2_3, k_val2_3, acc23]);
    let acc24_n = m.ext_inst(t_f32, glsl, 50, &[q_val2_4, k_val2_4, acc24]);
    let acc25_n = m.ext_inst(t_f32, glsl, 50, &[q_val2_5, k_val2_5, acc25]);
    let acc26_n = m.ext_inst(t_f32, glsl, 50, &[q_val2_6, k_val2_6, acc26]);
    let acc27_n = m.ext_inst(t_f32, glsl, 50, &[q_val2_7, k_val2_7, acc27]);
    m.store(var_acc0, acc20_n);
    m.store(var_acc1, acc21_n);
    m.store(var_acc2, acc22_n);
    m.store(var_acc3, acc23_n);
    m.store(var_acc4, acc24_n);
    m.store(var_acc5, acc25_n);
    m.store(var_acc6, acc26_n);
    m.store(var_acc7, acc27_n);
    m.branch(lbl_cont2);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_cont2.0]));
    let iv_next2 = m.iadd(t_u32, iv2, c_u32_8);
    m.store(var_i, iv_next2);
    m.branch(lbl_h2);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_m2.0]));
    // mv39: NEON `vaddq_f32(acc0, acc1) → vaddvq_f32` Pass 2 — same as Pass 1.
    let final_a20 = m.load(t_f32, var_acc0);
    let final_a21 = m.load(t_f32, var_acc1);
    let final_a22 = m.load(t_f32, var_acc2);
    let final_a23 = m.load(t_f32, var_acc3);
    let final_a24 = m.load(t_f32, var_acc4);
    let final_a25 = m.load(t_f32, var_acc5);
    let final_a26 = m.load(t_f32, var_acc6);
    let final_a27 = m.load(t_f32, var_acc7);
    let s2_l0 = m.fadd(t_f32, final_a20, final_a24);
    let s2_l1 = m.fadd(t_f32, final_a21, final_a25);
    let s2_l2 = m.fadd(t_f32, final_a22, final_a26);
    let s2_l3 = m.fadd(t_f32, final_a23, final_a27);
    let final_s2_01 = m.fadd(t_f32, s2_l0, s2_l1);
    let final_s2_23 = m.fadd(t_f32, s2_l2, s2_l3);
    let final_dot2 = m.fadd(t_f32, final_s2_01, final_s2_23);
    let scaled2 = m.fmul(t_f32, final_dot2, scale);
    let max_score = m.load(t_f32, var_max);
    let shifted = m.fsub(t_f32, scaled2, max_score);
    // mv37: GLSL Exp 대신 Exp2 (`exp(x) = exp2(x * log2(e))`). mv38 B 시도
    // (polynomial direct emit + op code swap bug fix) 결과 max_abs 동일 (Mali
    // Exp2 가 이미 매우 정확). polynomial 자체는 ULP 후퇴 (733→1198) 라 Exp2
    // keep. SPIR-V op code swap fix 는 별도 keep — 다른 vulkan kernel 영향.
    let c_log2e = m.constant_f32(t_f32, std::f32::consts::LOG2_E);
    let exp2_arg = m.fmul(t_f32, shifted, c_log2e);
    let weight = m.ext_inst(t_f32, glsl, 29, &[exp2_arg]);
    let den = m.load(t_f32, var_den);
    let den_next = m.fadd(t_f32, den, weight);
    m.store(var_den, den_next);
    let v_base = m.imul(t_u32, jv2, head_dim);
    let v_idx = m.iadd(t_u32, v_base, gid);
    let v_ptr = m.access_chain(t_ptr_sb_f32, gvar_v, &[c_u32_0, v_idx]);
    let v_val = m.load(t_f32, v_ptr);
    let num = m.load(t_f32, var_num);
    // mv38 C: V accumulation 도 OpExtInst Fma 로 single-rounding (NEON vfmaq).
    // 현 separate fmul + fadd → driver 가 자동 fma fuse 시 single-rounding,
    // 분리 시 two-rounding. 명시적 Fma 으로 NEON 과 비트 일치.
    let num_next = m.ext_inst(t_f32, glsl, 50, &[weight, v_val, num]);
    m.store(var_num, num_next);
    m.branch(lbl_kv_cont);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_kv_cont.0]));
    let jn2 = m.iadd(t_u32, jv2, c_u32_1);
    m.store(var_j, jn2);
    m.branch(lbl_kv_h);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_kv_m.0]));
    let final_num = m.load(t_f32, var_num);
    let final_den = m.load(t_f32, var_den);
    let out_val = m.fdiv(t_f32, final_num, final_den);
    let out_ptr = m.access_chain(t_ptr_sb_f32, gvar_out, &[c_u32_0, gid]);
    m.store(out_ptr, out_val);
    m.branch(lbl_gid_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_gid_end.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// KV append: writes K_in/V_in into kv_buffer at (layer_idx, cursor).
/// Layout: kv_buffer[layer][cursor][0|1=K|V][kv_head][head_dim] (f32).
/// Bindings: 0=K_in, 1=V_in, 2=kv_buffer. Push: layer_idx, max_ctx, kv_heads, head_dim, cursor (5 u32).
/// Dispatch: ceil_div(kv_heads*head_dim, local_size_x).
pub fn emit_kv_append(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_u32 = m.type_int(32, 0);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_arr_f32 = m.type_runtime_array(t_f32);

    let t_struct_k = m.type_struct(&[t_arr_f32]);
    let t_struct_v = m.type_struct(&[t_arr_f32]);
    let t_struct_kv = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32, t_u32, t_u32]);

    let t_ptr_sb_k = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_k);
    let t_ptr_sb_v = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_v);
    let t_ptr_sb_kv = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_kv);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_4 = m.constant_u32(t_u32, 4);

    m.decorate(t_struct_k, decoration::BLOCK, &[]);
    m.decorate(t_struct_v, decoration::BLOCK, &[]);
    m.decorate(t_struct_kv, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_k, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_v, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_kv, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);
    m.member_decorate(t_struct_pc, 3, decoration::OFFSET, &[12]);
    m.member_decorate(t_struct_pc, 4, decoration::OFFSET, &[16]);

    let gvar_k = m.variable(t_ptr_sb_k, storage_class::STORAGE_BUFFER);
    let gvar_v = m.variable(t_ptr_sb_v, storage_class::STORAGE_BUFFER);
    let gvar_kv = m.variable(t_ptr_sb_kv, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let gvar_gid = m.variable(t_ptr_input_v3, storage_class::INPUT);

    m.decorate(gvar_k, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_k, decoration::BINDING, &[0]);
    m.decorate(gvar_v, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_v, decoration::BINDING, &[1]);
    m.decorate(gvar_kv, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_kv, decoration::BINDING, &[2]);
    m.decorate(
        gvar_gid,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_gid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    m.function(t_void, func_id, 0, t_fn_void);
    let lbl_entry = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let gid_vec = m.load(t_v3u32, gvar_gid);
    let gid = m.composite_extract(t_u32, gid_vec, 0);

    let pc_layer_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let layer_idx = m.load(t_u32, pc_layer_ptr);
    let pc_maxctx_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let max_ctx = m.load(t_u32, pc_maxctx_ptr);
    let pc_kvh_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_2]);
    let kv_heads = m.load(t_u32, pc_kvh_ptr);
    let pc_hd_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_3]);
    let head_dim = m.load(t_u32, pc_hd_ptr);
    let pc_cur_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_4]);
    let cursor = m.load(t_u32, pc_cur_ptr);

    let stripe = m.imul(t_u32, kv_heads, head_dim);

    let t_bool = m.type_bool();
    let in_bounds = m.u_less_than(t_bool, gid, stripe);
    let lbl_body = m.alloc_id();
    let lbl_end = m.alloc_id();
    m.selection_merge(lbl_end, 0);
    m.branch_conditional(in_bounds, lbl_body, lbl_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_body.0]));

    let two_stripe = m.imul(t_u32, c_u32_2, stripe);
    let per_layer = m.imul(t_u32, max_ctx, two_stripe);
    let layer_base = m.imul(t_u32, layer_idx, per_layer);
    let token_base = m.imul(t_u32, cursor, two_stripe);
    let layer_token = m.iadd(t_u32, layer_base, token_base);
    let k_off = m.iadd(t_u32, layer_token, gid);
    let v_off = m.iadd(t_u32, k_off, stripe);

    let k_src_ptr = m.access_chain(t_ptr_sb_f32, gvar_k, &[c_u32_0, gid]);
    let k_val = m.load(t_f32, k_src_ptr);
    let v_src_ptr = m.access_chain(t_ptr_sb_f32, gvar_v, &[c_u32_0, gid]);
    let v_val = m.load(t_f32, v_src_ptr);

    let k_dst_ptr = m.access_chain(t_ptr_sb_f32, gvar_kv, &[c_u32_0, k_off]);
    m.store(k_dst_ptr, k_val);
    let v_dst_ptr = m.access_chain(t_ptr_sb_f32, gvar_kv, &[c_u32_0, v_off]);
    m.store(v_dst_ptr, v_val);

    m.branch(lbl_end);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_end.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// Gather + Q6_K dequant of embedding rows.
///
/// Bindings:
///   binding 0: token_ids (u32 array, length = num_tokens)
///   binding 1: embed_table_q6k (u32 runtime_array — raw bytes of vocab × hidden Q6_K table)
///   binding 2: embed_rows (f32 array, length = num_tokens × hidden)
///
/// Push constants (3 × u32): num_tokens, hidden, vocab.
///
/// Each thread writes one f32 output element. Q6_K block is 210 bytes (not u32-aligned),
/// so per-byte reads go through u32 word access + shift/mask.
pub fn emit_embed_lookup_q6k(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_i32 = m.type_int(32, 1);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);

    let t_arr_u32_ids = m.type_runtime_array(t_u32);
    let t_arr_f32_out = m.type_runtime_array(t_f32);

    let t_struct_ids = m.type_struct(&[t_arr_u32_ids]);
    let t_struct_table = m.type_struct(&[t_arr_u32_ids]);
    let t_struct_out = m.type_struct(&[t_arr_f32_out]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);

    let t_ptr_sb_struct_ids = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_ids);
    let t_ptr_sb_struct_table = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_table);
    let t_ptr_sb_struct_out = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_out);
    let t_ptr_pc_struct = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_4 = m.constant_u32(t_u32, 4);
    let c_u32_5 = m.constant_u32(t_u32, 5);
    let c_u32_7 = m.constant_u32(t_u32, 7);
    let c_u32_8 = m.constant_u32(t_u32, 8);
    let c_u32_10 = m.constant_u32(t_u32, 10);
    let c_u32_13 = m.constant_u32(t_u32, 13);
    let c_u32_15 = m.constant_u32(t_u32, 15);
    let c_u32_23 = m.constant_u32(t_u32, 23);
    let c_u32_24 = m.constant_u32(t_u32, 24);
    let c_u32_31 = m.constant_u32(t_u32, 31);
    let c_u32_32 = m.constant_u32(t_u32, 32);
    let c_u32_64 = m.constant_u32(t_u32, 64);
    let c_u32_112 = m.constant_u32(t_u32, 112);
    let c_u32_128 = m.constant_u32(t_u32, 128);
    let c_u32_192 = m.constant_u32(t_u32, 192);
    let c_u32_208 = m.constant_u32(t_u32, 208);
    let c_u32_209 = m.constant_u32(t_u32, 209);
    let c_u32_210 = m.constant_u32(t_u32, 210);
    let c_u32_256 = m.constant_u32(t_u32, 256);
    let c_u32_0f = m.constant_u32(t_u32, 0x0F);
    let c_u32_1f = m.constant_u32(t_u32, 0x1F);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3FF);
    let c_u32_ff = m.constant_u32(t_u32, 0xFF);
    let c_u32_7f = m.constant_u32(t_u32, 0x7F);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);

    m.decorate(t_struct_ids, decoration::BLOCK, &[]);
    m.decorate(t_struct_table, decoration::BLOCK, &[]);
    m.decorate(t_struct_out, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_ids, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_table, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_out, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_u32_ids, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_f32_out, decoration::ARRAY_STRIDE, &[4]);

    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);

    let gvar_ids = m.variable(t_ptr_sb_struct_ids, storage_class::STORAGE_BUFFER);
    let gvar_table = m.variable(t_ptr_sb_struct_table, storage_class::STORAGE_BUFFER);
    let gvar_out = m.variable(t_ptr_sb_struct_out, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc_struct, storage_class::PUSH_CONSTANT);
    let gvar_gid = m.variable(t_ptr_input_v3, storage_class::INPUT);

    m.decorate(gvar_ids, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_ids, decoration::BINDING, &[0]);
    m.decorate(gvar_table, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_table, decoration::BINDING, &[1]);
    m.decorate(gvar_out, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_out, decoration::BINDING, &[2]);
    m.decorate(
        gvar_gid,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_gid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    m.function(t_void, func_id, 0, t_fn_void);
    let lbl_entry = m.alloc_id();
    let lbl_body = m.alloc_id();
    let lbl_end = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let gid_vec = m.load(t_v3u32, gvar_gid);
    let gid = m.composite_extract(t_u32, gid_vec, 0);

    let pc_ptr_nt = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let num_tokens = m.load(t_u32, pc_ptr_nt);
    let pc_ptr_h = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let hidden = m.load(t_u32, pc_ptr_h);

    let total = m.imul(t_u32, num_tokens, hidden);
    let in_bounds = m.u_less_than(t_bool, gid, total);
    m.selection_merge(lbl_end, 0);
    m.branch_conditional(in_bounds, lbl_body, lbl_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_body.0]));

    let token_idx = m.udiv(t_u32, gid, hidden);
    let in_row_idx = m.umod(t_u32, gid, hidden);

    let tok_ptr = m.access_chain(t_ptr_sb_u32, gvar_ids, &[c_u32_0, token_idx]);
    let token_id = m.load(t_u32, tok_ptr);

    let pc_ptr_vocab = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_2]);
    let vocab = m.load(t_u32, pc_ptr_vocab);
    let in_vocab = m.u_less_than(t_bool, token_id, vocab);

    let lbl_valid = m.alloc_id();
    let lbl_oob = m.alloc_id();
    let lbl_body_end = m.alloc_id();
    m.selection_merge(lbl_body_end, 0);
    m.branch_conditional(in_vocab, lbl_valid, lbl_oob);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_valid.0]));

    let blocks_per_row = m.udiv(t_u32, hidden, c_u32_256);
    let block_idx_in_row = m.udiv(t_u32, in_row_idx, c_u32_256);
    let elem_in_block = m.umod(t_u32, in_row_idx, c_u32_256);

    let row_block_count_x_id = m.imul(t_u32, token_id, blocks_per_row);
    let global_block_idx = m.iadd(t_u32, row_block_count_x_id, block_idx_in_row);
    let block_byte_off = m.imul(t_u32, global_block_idx, c_u32_210);

    // Element decomposition (matches CPU dequantize_q6_k):
    //   n      = elem >> 7   (0..2) — group of 128
    //   local  = elem & 0x7F (0..128)
    //   stripe = local >> 5  (0..4)
    //   l      = local & 0x1F (0..32)
    //   is     = l >> 4      (0..2)
    let n = m.shift_right_logical(t_u32, elem_in_block, c_u32_7);
    let local = m.bitwise_and(t_u32, elem_in_block, c_u32_7f);
    let stripe = m.shift_right_logical(t_u32, local, c_u32_5);
    let l = m.bitwise_and(t_u32, local, c_u32_1f);
    let is_idx = m.shift_right_logical(t_u32, l, c_u32_4);

    let ql_base = m.imul(t_u32, n, c_u32_64);
    let qh_base = m.imul(t_u32, n, c_u32_32);
    let sc_base = m.imul(t_u32, n, c_u32_8);

    let ql_high = m.shift_right_logical(t_u32, stripe, c_u32_1);
    let stripe_lo_bit = m.bitwise_and(t_u32, stripe, c_u32_1);
    let ql_l_extra = m.shift_left_logical(t_u32, stripe_lo_bit, c_u32_5);
    let qh_shift_amt = m.shift_left_logical(t_u32, stripe, c_u32_1);
    let stripe_x2 = m.shift_left_logical(t_u32, stripe, c_u32_1);
    let sc_idx_off = m.iadd(t_u32, is_idx, stripe_x2);

    let ql_idx_a = m.iadd(t_u32, ql_base, l);
    let ql_idx = m.iadd(t_u32, ql_idx_a, ql_l_extra);
    let qh_idx = m.iadd(t_u32, qh_base, l);
    let sc_idx = m.iadd(t_u32, sc_base, sc_idx_off);

    let ql_byte_off = m.iadd(t_u32, block_byte_off, ql_idx);
    let qh_off_in_blk = m.iadd(t_u32, c_u32_128, qh_idx);
    let qh_byte_off = m.iadd(t_u32, block_byte_off, qh_off_in_blk);
    let sc_off_in_blk = m.iadd(t_u32, c_u32_192, sc_idx);
    let sc_byte_off = m.iadd(t_u32, block_byte_off, sc_off_in_blk);
    let d_lo_byte_off = m.iadd(t_u32, block_byte_off, c_u32_208);
    let d_hi_byte_off = m.iadd(t_u32, block_byte_off, c_u32_209);

    // Helper: read one byte at `byte_off` via u32 word access + shift/mask.
    let read_byte = |m: &mut SpirvModule, byte_off: Id| -> Id {
        let word_idx = m.shift_right_logical(t_u32, byte_off, c_u32_2);
        let byte_in_word = m.bitwise_and(t_u32, byte_off, c_u32_3);
        let shift_bits = m.shift_left_logical(t_u32, byte_in_word, c_u32_3);
        let word_ptr = m.access_chain(t_ptr_sb_u32, gvar_table, &[c_u32_0, word_idx]);
        let word = m.load(t_u32, word_ptr);
        let shifted = m.shift_right_logical(t_u32, word, shift_bits);
        m.bitwise_and(t_u32, shifted, c_u32_ff)
    };

    let ql_byte = read_byte(&mut m, ql_byte_off);
    let qh_byte = read_byte(&mut m, qh_byte_off);
    let sc_byte = read_byte(&mut m, sc_byte_off);
    let d_lo = read_byte(&mut m, d_lo_byte_off);
    let d_hi = read_byte(&mut m, d_hi_byte_off);

    let ql_shift_amt = m.shift_left_logical(t_u32, ql_high, c_u32_2);
    let ql_shifted = m.shift_right_logical(t_u32, ql_byte, ql_shift_amt);
    let ql_nib = m.bitwise_and(t_u32, ql_shifted, c_u32_0f);

    let qh_shifted = m.shift_right_logical(t_u32, qh_byte, qh_shift_amt);
    let qh_2bit = m.bitwise_and(t_u32, qh_shifted, c_u32_3);

    let qh_2bit_sh4 = m.shift_left_logical(t_u32, qh_2bit, c_u32_4);
    let q6 = m.bitwise_or(t_u32, ql_nib, qh_2bit_sh4);
    let q6_i32 = m.bitcast(t_i32, q6);
    let c_32_i32 = m.bitcast(t_i32, c_u32_32);
    let q_centered = m.isub(t_i32, q6_i32, c_32_i32);
    let q_f = m.convert_s_to_f(t_f32, q_centered);

    // Sign-extend i8 scale: shl 24, sar 24
    let sc_shl24 = m.shift_left_logical(t_u32, sc_byte, c_u32_24);
    let sc_shl24_i32 = m.bitcast(t_i32, sc_shl24);
    let sc_i32 = m.shift_right_arithmetic(t_i32, sc_shl24_i32, c_u32_24);
    let sc_f = m.convert_s_to_f(t_f32, sc_i32);

    let d_hi_sh8 = m.shift_left_logical(t_u32, d_hi, c_u32_8);
    let d_f16_raw = m.bitwise_or(t_u32, d_lo, d_hi_sh8);

    let c_f32_2pow_neg24 = m.constant_f32(t_f32, 5.9604644775390625e-8);
    let d_f32 = {
        let exp_raw = m.shift_right_logical(t_u32, d_f16_raw, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, d_f16_raw, c_u32_3ff);
        let sign = m.shift_right_logical(t_u32, d_f16_raw, c_u32_15);
        let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
        let s_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
        let e_adj = m.iadd(t_u32, exp, c_u32_112);
        let e_part = m.shift_left_logical(t_u32, e_adj, c_u32_23);
        let m_part = m.shift_left_logical(t_u32, mant, c_u32_13);
        let bits_mid = m.bitwise_or(t_u32, s_part, e_part);
        let bits = m.bitwise_or(t_u32, bits_mid, m_part);
        let normal = m.bitcast(t_f32, bits);
        let mant_f = m.convert_u_to_f(t_f32, mant);
        let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
        let denorm_neg = m.fnegate(t_f32, denorm_abs);
        let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
        let denormal = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
        let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
        m.select(t_f32, exp_nonzero, normal, denormal)
    };

    let d_sc = m.fmul(t_f32, d_f32, sc_f);
    let val = m.fmul(t_f32, d_sc, q_f);

    let out_ptr = m.access_chain(t_ptr_sb_f32, gvar_out, &[c_u32_0, gid]);
    m.store(out_ptr, val);

    m.branch(lbl_body_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_oob.0]));
    let out_ptr_oob = m.access_chain(t_ptr_sb_f32, gvar_out, &[c_u32_0, gid]);
    m.store(out_ptr_oob, c_f32_0);
    m.branch(lbl_body_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_body_end.0]));
    m.branch(lbl_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_end.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// Fused Q6_K matmul (hidden × vocab) + argmax reduction in a single dispatch.
///
/// Bindings:
///   binding 0: hidden_vec (f32 array, length = hidden)
///   binding 1: output_table_q6k (u32 runtime_array — raw bytes of vocab × hidden Q6_K table)
///   binding 2: argmax_out (u32, single element — argmax token id)
///
/// Push constants (2 × u32): vocab, hidden.
///
/// Dispatch shape: (1, 1, 1) — single workgroup of `local_size_x` threads.
/// Each thread iterates over a stripe of vocab rows; pairs reduce via shared memory.
pub fn emit_logit_argmax_q6k(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_i32 = m.type_int(32, 1);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);

    let t_arr_f32_in = m.type_runtime_array(t_f32);
    let t_arr_u32_table = m.type_runtime_array(t_u32);
    let t_arr_u32_out = m.type_runtime_array(t_u32);

    let t_struct_in = m.type_struct(&[t_arr_f32_in]);
    let t_struct_table = m.type_struct(&[t_arr_u32_table]);
    let t_struct_out = m.type_struct(&[t_arr_u32_out]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32]);

    let c_local_size = m.constant_u32(t_u32, local_size_x);
    let t_shared_arr_f32 = m.type_array(t_f32, c_local_size);
    let t_shared_arr_u32 = m.type_array(t_u32, c_local_size);
    let t_ptr_wg_arr_f32 = m.type_pointer(storage_class::WORKGROUP, t_shared_arr_f32);
    let t_ptr_wg_arr_u32 = m.type_pointer(storage_class::WORKGROUP, t_shared_arr_u32);
    let t_ptr_wg_f32 = m.type_pointer(storage_class::WORKGROUP, t_f32);
    let t_ptr_wg_u32 = m.type_pointer(storage_class::WORKGROUP, t_u32);

    let t_ptr_sb_in = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_in);
    let t_ptr_sb_table = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_table);
    let t_ptr_sb_out = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_out);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_4 = m.constant_u32(t_u32, 4);
    let c_u32_5 = m.constant_u32(t_u32, 5);
    let c_u32_7 = m.constant_u32(t_u32, 7);
    let c_u32_8 = m.constant_u32(t_u32, 8);
    let c_u32_10 = m.constant_u32(t_u32, 10);
    let c_u32_13 = m.constant_u32(t_u32, 13);
    let c_u32_15 = m.constant_u32(t_u32, 15);
    let c_u32_23 = m.constant_u32(t_u32, 23);
    let c_u32_24 = m.constant_u32(t_u32, 24);
    let c_u32_31 = m.constant_u32(t_u32, 31);
    let c_u32_32 = m.constant_u32(t_u32, 32);
    let c_u32_64 = m.constant_u32(t_u32, 64);
    let c_u32_112 = m.constant_u32(t_u32, 112);
    let c_u32_128 = m.constant_u32(t_u32, 128);
    let c_u32_192 = m.constant_u32(t_u32, 192);
    let c_u32_208 = m.constant_u32(t_u32, 208);
    let c_u32_209 = m.constant_u32(t_u32, 209);
    let c_u32_210 = m.constant_u32(t_u32, 210);
    let c_u32_256 = m.constant_u32(t_u32, 256);
    let c_u32_0f = m.constant_u32(t_u32, 0x0F);
    let c_u32_1f = m.constant_u32(t_u32, 0x1F);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3FF);
    let c_u32_ff = m.constant_u32(t_u32, 0xFF);
    let c_u32_7f = m.constant_u32(t_u32, 0x7F);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);
    let c_f32_neg_inf = m.constant_f32(t_f32, f32::NEG_INFINITY);
    let c_f32_2pow_neg24 = m.constant_f32(t_f32, 5.9604644775390625e-8);

    m.decorate(t_struct_in, decoration::BLOCK, &[]);
    m.decorate(t_struct_table, decoration::BLOCK, &[]);
    m.decorate(t_struct_out, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_in, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_table, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_out, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_f32_in, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_u32_table, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_u32_out, decoration::ARRAY_STRIDE, &[4]);

    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);

    let gvar_in = m.variable(t_ptr_sb_in, storage_class::STORAGE_BUFFER);
    let gvar_table = m.variable(t_ptr_sb_table, storage_class::STORAGE_BUFFER);
    let gvar_out = m.variable(t_ptr_sb_out, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let gvar_lid = m.variable(t_ptr_input_v3, storage_class::INPUT);
    let gvar_shared_vals = m.variable(t_ptr_wg_arr_f32, storage_class::WORKGROUP);
    let gvar_shared_idxs = m.variable(t_ptr_wg_arr_u32, storage_class::WORKGROUP);

    m.decorate(gvar_in, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_in, decoration::BINDING, &[0]);
    m.decorate(gvar_table, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_table, decoration::BINDING, &[1]);
    m.decorate(gvar_out, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_out, decoration::BINDING, &[2]);
    m.decorate(
        gvar_lid,
        decoration::BUILTIN,
        &[builtin::LOCAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_lid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    m.function(t_void, func_id, 0, t_fn_void);
    let lbl_entry = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    // Function variables (must be declared at top of first block)
    let var_v = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_h = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_best_val = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_best_idx = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_step = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);

    let lid_vec = m.load(t_v3u32, gvar_lid);
    let lid = m.composite_extract(t_u32, lid_vec, 0);

    let pc_vocab_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let vocab = m.load(t_u32, pc_vocab_ptr);
    let pc_hidden_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let hidden = m.load(t_u32, pc_hidden_ptr);
    let blocks_per_row = m.udiv(t_u32, hidden, c_u32_256);

    m.store(var_best_val, c_f32_neg_inf);
    m.store(var_best_idx, c_u32_0);
    m.store(var_v, lid);

    // ---- Outer loop: stride over vocab rows ----
    let lbl_v_h = m.alloc_id();
    let lbl_v_c = m.alloc_id();
    let lbl_v_b = m.alloc_id();
    let lbl_v_cont = m.alloc_id();
    let lbl_v_m = m.alloc_id();

    m.branch(lbl_v_h);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_v_h.0]));
    m.loop_merge(lbl_v_m, lbl_v_cont, 0);
    m.branch(lbl_v_c);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_v_c.0]));
    let cur_v = m.load(t_u32, var_v);
    let v_in_bounds = m.u_less_than(t_bool, cur_v, vocab);
    m.branch_conditional(v_in_bounds, lbl_v_b, lbl_v_m);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_v_b.0]));
    m.store(var_sum, c_f32_0);
    m.store(var_h, c_u32_0);

    // ---- Inner loop: hidden dot product with Q6_K dequant per element ----
    let lbl_h_h = m.alloc_id();
    let lbl_h_c = m.alloc_id();
    let lbl_h_b = m.alloc_id();
    let lbl_h_cont = m.alloc_id();
    let lbl_h_m = m.alloc_id();

    m.branch(lbl_h_h);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_h_h.0]));
    m.loop_merge(lbl_h_m, lbl_h_cont, 0);
    m.branch(lbl_h_c);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_h_c.0]));
    let cur_h = m.load(t_u32, var_h);
    let h_in_bounds = m.u_less_than(t_bool, cur_h, hidden);
    m.branch_conditional(h_in_bounds, lbl_h_b, lbl_h_m);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_h_b.0]));

    // Q6_K dequant for (v=cur_v, elem_in_row=cur_h)
    let block_idx_in_row = m.udiv(t_u32, cur_h, c_u32_256);
    let elem_in_block = m.umod(t_u32, cur_h, c_u32_256);
    let row_block_count_x_v = m.imul(t_u32, cur_v, blocks_per_row);
    let global_block_idx = m.iadd(t_u32, row_block_count_x_v, block_idx_in_row);
    let block_byte_off = m.imul(t_u32, global_block_idx, c_u32_210);

    let n = m.shift_right_logical(t_u32, elem_in_block, c_u32_7);
    let local = m.bitwise_and(t_u32, elem_in_block, c_u32_7f);
    let stripe = m.shift_right_logical(t_u32, local, c_u32_5);
    let l = m.bitwise_and(t_u32, local, c_u32_1f);
    let is_idx = m.shift_right_logical(t_u32, l, c_u32_4);

    let ql_base = m.imul(t_u32, n, c_u32_64);
    let qh_base = m.imul(t_u32, n, c_u32_32);
    let sc_base = m.imul(t_u32, n, c_u32_8);

    let ql_high = m.shift_right_logical(t_u32, stripe, c_u32_1);
    let stripe_lo_bit = m.bitwise_and(t_u32, stripe, c_u32_1);
    let ql_l_extra = m.shift_left_logical(t_u32, stripe_lo_bit, c_u32_5);
    let qh_shift_amt = m.shift_left_logical(t_u32, stripe, c_u32_1);
    let stripe_x2 = m.shift_left_logical(t_u32, stripe, c_u32_1);
    let sc_idx_off = m.iadd(t_u32, is_idx, stripe_x2);

    let ql_idx_a = m.iadd(t_u32, ql_base, l);
    let ql_idx = m.iadd(t_u32, ql_idx_a, ql_l_extra);
    let qh_idx = m.iadd(t_u32, qh_base, l);
    let sc_idx = m.iadd(t_u32, sc_base, sc_idx_off);

    let ql_byte_off = m.iadd(t_u32, block_byte_off, ql_idx);
    let qh_off_in_blk = m.iadd(t_u32, c_u32_128, qh_idx);
    let qh_byte_off = m.iadd(t_u32, block_byte_off, qh_off_in_blk);
    let sc_off_in_blk = m.iadd(t_u32, c_u32_192, sc_idx);
    let sc_byte_off = m.iadd(t_u32, block_byte_off, sc_off_in_blk);
    let d_lo_byte_off = m.iadd(t_u32, block_byte_off, c_u32_208);
    let d_hi_byte_off = m.iadd(t_u32, block_byte_off, c_u32_209);

    let read_byte = |m: &mut SpirvModule, byte_off: Id| -> Id {
        let word_idx = m.shift_right_logical(t_u32, byte_off, c_u32_2);
        let byte_in_word = m.bitwise_and(t_u32, byte_off, c_u32_3);
        let shift_bits = m.shift_left_logical(t_u32, byte_in_word, c_u32_3);
        let word_ptr = m.access_chain(t_ptr_sb_u32, gvar_table, &[c_u32_0, word_idx]);
        let word = m.load(t_u32, word_ptr);
        let shifted = m.shift_right_logical(t_u32, word, shift_bits);
        m.bitwise_and(t_u32, shifted, c_u32_ff)
    };

    let ql_byte = read_byte(&mut m, ql_byte_off);
    let qh_byte = read_byte(&mut m, qh_byte_off);
    let sc_byte = read_byte(&mut m, sc_byte_off);
    let d_lo = read_byte(&mut m, d_lo_byte_off);
    let d_hi = read_byte(&mut m, d_hi_byte_off);

    let ql_shift_amt = m.shift_left_logical(t_u32, ql_high, c_u32_2);
    let ql_shifted = m.shift_right_logical(t_u32, ql_byte, ql_shift_amt);
    let ql_nib = m.bitwise_and(t_u32, ql_shifted, c_u32_0f);

    let qh_shifted = m.shift_right_logical(t_u32, qh_byte, qh_shift_amt);
    let qh_2bit = m.bitwise_and(t_u32, qh_shifted, c_u32_3);

    let qh_2bit_sh4 = m.shift_left_logical(t_u32, qh_2bit, c_u32_4);
    let q6 = m.bitwise_or(t_u32, ql_nib, qh_2bit_sh4);
    let q6_i32 = m.bitcast(t_i32, q6);
    let c_32_i32 = m.bitcast(t_i32, c_u32_32);
    let q_centered = m.isub(t_i32, q6_i32, c_32_i32);
    let q_f = m.convert_s_to_f(t_f32, q_centered);

    let sc_shl24 = m.shift_left_logical(t_u32, sc_byte, c_u32_24);
    let sc_shl24_i32 = m.bitcast(t_i32, sc_shl24);
    let sc_i32 = m.shift_right_arithmetic(t_i32, sc_shl24_i32, c_u32_24);
    let sc_f = m.convert_s_to_f(t_f32, sc_i32);

    let d_hi_sh8 = m.shift_left_logical(t_u32, d_hi, c_u32_8);
    let d_f16_raw = m.bitwise_or(t_u32, d_lo, d_hi_sh8);
    let d_f32 = {
        let exp_raw = m.shift_right_logical(t_u32, d_f16_raw, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, d_f16_raw, c_u32_3ff);
        let sign = m.shift_right_logical(t_u32, d_f16_raw, c_u32_15);
        let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
        let s_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
        let e_adj = m.iadd(t_u32, exp, c_u32_112);
        let e_part = m.shift_left_logical(t_u32, e_adj, c_u32_23);
        let m_part = m.shift_left_logical(t_u32, mant, c_u32_13);
        let bits_mid = m.bitwise_or(t_u32, s_part, e_part);
        let bits = m.bitwise_or(t_u32, bits_mid, m_part);
        let normal = m.bitcast(t_f32, bits);
        let mant_f = m.convert_u_to_f(t_f32, mant);
        let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
        let denorm_neg = m.fnegate(t_f32, denorm_abs);
        let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
        let denormal = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
        let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
        m.select(t_f32, exp_nonzero, normal, denormal)
    };

    let d_sc = m.fmul(t_f32, d_f32, sc_f);
    let weight_val = m.fmul(t_f32, d_sc, q_f);

    let hv_ptr = m.access_chain(t_ptr_sb_f32, gvar_in, &[c_u32_0, cur_h]);
    let hv = m.load(t_f32, hv_ptr);
    let prod = m.fmul(t_f32, weight_val, hv);
    let cur_sum = m.load(t_f32, var_sum);
    let new_sum = m.fadd(t_f32, cur_sum, prod);
    m.store(var_sum, new_sum);

    m.branch(lbl_h_cont);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_h_cont.0]));
    let next_h = m.iadd(t_u32, cur_h, c_u32_1);
    m.store(var_h, next_h);
    m.branch(lbl_h_h);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_h_m.0]));

    // Update local best (val, idx) if sum > best_val
    let final_sum = m.load(t_f32, var_sum);
    let cur_best = m.load(t_f32, var_best_val);
    let beats = m.f_ord_greater_than(t_bool, final_sum, cur_best);
    let lbl_upd = m.alloc_id();
    let lbl_upd_m = m.alloc_id();
    m.selection_merge(lbl_upd_m, 0);
    m.branch_conditional(beats, lbl_upd, lbl_upd_m);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_upd.0]));
    m.store(var_best_val, final_sum);
    m.store(var_best_idx, cur_v);
    m.branch(lbl_upd_m);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_upd_m.0]));

    m.branch(lbl_v_cont);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_v_cont.0]));
    let next_v = m.iadd(t_u32, cur_v, c_local_size);
    m.store(var_v, next_v);
    m.branch(lbl_v_h);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_v_m.0]));

    // ---- Stash this thread's local best into shared memory ----
    let final_best_val = m.load(t_f32, var_best_val);
    let final_best_idx = m.load(t_u32, var_best_idx);
    let sv_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared_vals, &[lid]);
    let si_ptr = m.access_chain(t_ptr_wg_u32, gvar_shared_idxs, &[lid]);
    m.store(sv_ptr, final_best_val);
    m.store(si_ptr, final_best_idx);
    {
        let c_scope_wg = m.constant_u32(t_u32, scope::WORKGROUP);
        let c_sem = m.constant_u32(
            t_u32,
            memory_semantics::WORKGROUP_MEMORY | memory_semantics::ACQUIRE_RELEASE,
        );
        m.control_barrier(c_scope_wg, c_scope_wg, c_sem);
    }

    // ---- Pairwise reduction (paired val + idx) ----
    let c_half = m.constant_u32(t_u32, local_size_x / 2);
    m.store(var_step, c_half);

    let lbl_r_h = m.alloc_id();
    let lbl_r_c = m.alloc_id();
    let lbl_r_b = m.alloc_id();
    let lbl_r_cont = m.alloc_id();
    let lbl_r_m = m.alloc_id();

    m.branch(lbl_r_h);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_h.0]));
    m.loop_merge(lbl_r_m, lbl_r_cont, 0);
    m.branch(lbl_r_c);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_c.0]));
    let step_v = m.load(t_u32, var_step);
    let step_pos = m.u_less_than(t_bool, c_u32_0, step_v);
    m.branch_conditional(step_pos, lbl_r_b, lbl_r_m);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_b.0]));
    let lid_lt_step = m.u_less_than(t_bool, lid, step_v);
    let lbl_r_a = m.alloc_id();
    let lbl_r_am = m.alloc_id();
    m.selection_merge(lbl_r_am, 0);
    m.branch_conditional(lid_lt_step, lbl_r_a, lbl_r_am);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_a.0]));
    let other_lid = m.iadd(t_u32, lid, step_v);
    let self_v_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared_vals, &[lid]);
    let self_i_ptr = m.access_chain(t_ptr_wg_u32, gvar_shared_idxs, &[lid]);
    let other_v_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared_vals, &[other_lid]);
    let other_i_ptr = m.access_chain(t_ptr_wg_u32, gvar_shared_idxs, &[other_lid]);
    let self_v = m.load(t_f32, self_v_ptr);
    let self_i = m.load(t_u32, self_i_ptr);
    let other_v = m.load(t_f32, other_v_ptr);
    let other_i = m.load(t_u32, other_i_ptr);
    let other_beats = m.f_ord_greater_than(t_bool, other_v, self_v);
    let new_v = m.select(t_f32, other_beats, other_v, self_v);
    let new_i = m.select(t_u32, other_beats, other_i, self_i);
    m.store(self_v_ptr, new_v);
    m.store(self_i_ptr, new_i);
    m.branch(lbl_r_am);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_am.0]));
    {
        let c_scope_wg = m.constant_u32(t_u32, scope::WORKGROUP);
        let c_sem = m.constant_u32(
            t_u32,
            memory_semantics::WORKGROUP_MEMORY | memory_semantics::ACQUIRE_RELEASE,
        );
        m.control_barrier(c_scope_wg, c_scope_wg, c_sem);
    }
    m.branch(lbl_r_cont);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_cont.0]));
    let step_half = m.shift_right_logical(t_u32, step_v, c_u32_1);
    m.store(var_step, step_half);
    m.branch(lbl_r_h);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_m.0]));

    // ---- Thread 0 writes argmax_out[0] ----
    let is_lid_zero = m.u_less_than(t_bool, lid, c_u32_1);
    let lbl_w = m.alloc_id();
    let lbl_w_m = m.alloc_id();
    m.selection_merge(lbl_w_m, 0);
    m.branch_conditional(is_lid_zero, lbl_w, lbl_w_m);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_w.0]));
    let win_idx_ptr = m.access_chain(t_ptr_wg_u32, gvar_shared_idxs, &[c_u32_0]);
    let win_idx = m.load(t_u32, win_idx_ptr);
    let out_ptr = m.access_chain(t_ptr_sb_u32, gvar_out, &[c_u32_0, c_u32_0]);
    m.store(out_ptr, win_idx);
    m.branch(lbl_w_m);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_w_m.0]));

    m.ret();
    m.function_end();

    m.encode()
}

/// Fused Q8_0 matmul (hidden × vocab) + argmax reduction.
///
/// Binding 1 is the raw row-major GGML Q8_0 output table:
/// each 32-value block is `d:f16` followed by 32 signed i8 values.
pub fn emit_logit_argmax_q8_0(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_i32 = m.type_int(32, 1);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);

    let t_arr_f32_in = m.type_runtime_array(t_f32);
    let t_arr_u32_table = m.type_runtime_array(t_u32);
    let t_arr_u32_out = m.type_runtime_array(t_u32);

    let t_struct_in = m.type_struct(&[t_arr_f32_in]);
    let t_struct_table = m.type_struct(&[t_arr_u32_table]);
    let t_struct_out = m.type_struct(&[t_arr_u32_out]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32]);

    let c_local_size = m.constant_u32(t_u32, local_size_x);
    let t_shared_arr_f32 = m.type_array(t_f32, c_local_size);
    let t_shared_arr_u32 = m.type_array(t_u32, c_local_size);
    let t_ptr_wg_arr_f32 = m.type_pointer(storage_class::WORKGROUP, t_shared_arr_f32);
    let t_ptr_wg_arr_u32 = m.type_pointer(storage_class::WORKGROUP, t_shared_arr_u32);
    let t_ptr_wg_f32 = m.type_pointer(storage_class::WORKGROUP, t_f32);
    let t_ptr_wg_u32 = m.type_pointer(storage_class::WORKGROUP, t_u32);

    let t_ptr_sb_in = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_in);
    let t_ptr_sb_table = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_table);
    let t_ptr_sb_out = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_out);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_8 = m.constant_u32(t_u32, 8);
    let c_u32_10 = m.constant_u32(t_u32, 10);
    let c_u32_13 = m.constant_u32(t_u32, 13);
    let c_u32_15 = m.constant_u32(t_u32, 15);
    let c_u32_23 = m.constant_u32(t_u32, 23);
    let c_u32_31 = m.constant_u32(t_u32, 31);
    let c_u32_32 = m.constant_u32(t_u32, 32);
    let c_u32_34 = m.constant_u32(t_u32, 34);
    let c_u32_112 = m.constant_u32(t_u32, 112);
    let c_u32_1f = m.constant_u32(t_u32, 0x1F);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3FF);
    let c_u32_ff = m.constant_u32(t_u32, 0xFF);
    let c_i32_24 = m.constant_u32(t_i32, 24);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);
    let c_f32_neg_inf = m.constant_f32(t_f32, f32::NEG_INFINITY);
    let c_f32_2pow_neg24 = m.constant_f32(t_f32, 5.9604644775390625e-8);

    m.decorate(t_struct_in, decoration::BLOCK, &[]);
    m.decorate(t_struct_table, decoration::BLOCK, &[]);
    m.decorate(t_struct_out, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_in, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_table, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_out, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_f32_in, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_u32_table, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_u32_out, decoration::ARRAY_STRIDE, &[4]);

    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);

    let gvar_in = m.variable(t_ptr_sb_in, storage_class::STORAGE_BUFFER);
    let gvar_table = m.variable(t_ptr_sb_table, storage_class::STORAGE_BUFFER);
    let gvar_out = m.variable(t_ptr_sb_out, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let gvar_lid = m.variable(t_ptr_input_v3, storage_class::INPUT);
    let gvar_shared_vals = m.variable(t_ptr_wg_arr_f32, storage_class::WORKGROUP);
    let gvar_shared_idxs = m.variable(t_ptr_wg_arr_u32, storage_class::WORKGROUP);

    m.decorate(gvar_in, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_in, decoration::BINDING, &[0]);
    m.decorate(gvar_table, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_table, decoration::BINDING, &[1]);
    m.decorate(gvar_out, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_out, decoration::BINDING, &[2]);
    m.decorate(
        gvar_lid,
        decoration::BUILTIN,
        &[builtin::LOCAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_lid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    m.function(t_void, func_id, 0, t_fn_void);
    let lbl_entry = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let var_v = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_h = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_best_val = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_best_idx = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_step = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);

    let lid_vec = m.load(t_v3u32, gvar_lid);
    let lid = m.composite_extract(t_u32, lid_vec, 0);

    let pc_vocab_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let vocab = m.load(t_u32, pc_vocab_ptr);
    let pc_hidden_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let hidden = m.load(t_u32, pc_hidden_ptr);
    let blocks_per_row = m.udiv(t_u32, hidden, c_u32_32);

    m.store(var_best_val, c_f32_neg_inf);
    m.store(var_best_idx, c_u32_0);
    m.store(var_v, lid);

    let read_byte = |m: &mut SpirvModule, byte_off: Id| -> Id {
        let word_idx = m.shift_right_logical(t_u32, byte_off, c_u32_2);
        let byte_in_word = m.bitwise_and(t_u32, byte_off, c_u32_3);
        let shift_bits = m.shift_left_logical(t_u32, byte_in_word, c_u32_3);
        let word_ptr = m.access_chain(t_ptr_sb_u32, gvar_table, &[c_u32_0, word_idx]);
        let word = m.load(t_u32, word_ptr);
        let shifted = m.shift_right_logical(t_u32, word, shift_bits);
        m.bitwise_and(t_u32, shifted, c_u32_ff)
    };

    let lbl_v_h = m.alloc_id();
    let lbl_v_c = m.alloc_id();
    let lbl_v_b = m.alloc_id();
    let lbl_v_cont = m.alloc_id();
    let lbl_v_m = m.alloc_id();

    m.branch(lbl_v_h);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_v_h.0]));
    m.loop_merge(lbl_v_m, lbl_v_cont, 0);
    m.branch(lbl_v_c);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_v_c.0]));
    let cur_v = m.load(t_u32, var_v);
    let v_in_bounds = m.u_less_than(t_bool, cur_v, vocab);
    m.branch_conditional(v_in_bounds, lbl_v_b, lbl_v_m);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_v_b.0]));
    m.store(var_sum, c_f32_0);
    m.store(var_h, c_u32_0);

    let lbl_h_h = m.alloc_id();
    let lbl_h_c = m.alloc_id();
    let lbl_h_b = m.alloc_id();
    let lbl_h_cont = m.alloc_id();
    let lbl_h_m = m.alloc_id();

    m.branch(lbl_h_h);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_h_h.0]));
    m.loop_merge(lbl_h_m, lbl_h_cont, 0);
    m.branch(lbl_h_c);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_h_c.0]));
    let cur_h = m.load(t_u32, var_h);
    let h_in_bounds = m.u_less_than(t_bool, cur_h, hidden);
    m.branch_conditional(h_in_bounds, lbl_h_b, lbl_h_m);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_h_b.0]));
    let block_idx_in_row = m.udiv(t_u32, cur_h, c_u32_32);
    let elem_in_block = m.umod(t_u32, cur_h, c_u32_32);
    let row_block_count_x_v = m.imul(t_u32, cur_v, blocks_per_row);
    let global_block_idx = m.iadd(t_u32, row_block_count_x_v, block_idx_in_row);
    let block_byte_off = m.imul(t_u32, global_block_idx, c_u32_34);
    let q_byte_rel = m.iadd(t_u32, c_u32_2, elem_in_block);
    let q_byte_off = m.iadd(t_u32, block_byte_off, q_byte_rel);
    let d_hi_byte_off = m.iadd(t_u32, block_byte_off, c_u32_1);

    let q_byte = read_byte(&mut m, q_byte_off);
    let d_lo = read_byte(&mut m, block_byte_off);
    let d_hi = read_byte(&mut m, d_hi_byte_off);

    let q_i32 = m.bitcast(t_i32, q_byte);
    let q_shl = m.shift_left_logical(t_i32, q_i32, c_i32_24);
    let q_signed = m.shift_right_arithmetic(t_i32, q_shl, c_i32_24);
    let q_f = m.convert_s_to_f(t_f32, q_signed);

    let d_hi_sh8 = m.shift_left_logical(t_u32, d_hi, c_u32_8);
    let d_f16_raw = m.bitwise_or(t_u32, d_lo, d_hi_sh8);
    let d_f32 = {
        let exp_raw = m.shift_right_logical(t_u32, d_f16_raw, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, d_f16_raw, c_u32_3ff);
        let sign = m.shift_right_logical(t_u32, d_f16_raw, c_u32_15);
        let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
        let s_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
        let e_adj = m.iadd(t_u32, exp, c_u32_112);
        let e_part = m.shift_left_logical(t_u32, e_adj, c_u32_23);
        let m_part = m.shift_left_logical(t_u32, mant, c_u32_13);
        let bits_mid = m.bitwise_or(t_u32, s_part, e_part);
        let bits = m.bitwise_or(t_u32, bits_mid, m_part);
        let normal = m.bitcast(t_f32, bits);
        let mant_f = m.convert_u_to_f(t_f32, mant);
        let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
        let denorm_neg = m.fnegate(t_f32, denorm_abs);
        let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
        let denormal = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
        let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
        m.select(t_f32, exp_nonzero, normal, denormal)
    };

    let weight_val = m.fmul(t_f32, d_f32, q_f);
    let hv_ptr = m.access_chain(t_ptr_sb_f32, gvar_in, &[c_u32_0, cur_h]);
    let hv = m.load(t_f32, hv_ptr);
    let prod = m.fmul(t_f32, weight_val, hv);
    let cur_sum = m.load(t_f32, var_sum);
    let new_sum = m.fadd(t_f32, cur_sum, prod);
    m.store(var_sum, new_sum);

    m.branch(lbl_h_cont);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_h_cont.0]));
    let next_h = m.iadd(t_u32, cur_h, c_u32_1);
    m.store(var_h, next_h);
    m.branch(lbl_h_h);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_h_m.0]));
    let final_sum = m.load(t_f32, var_sum);
    let cur_best = m.load(t_f32, var_best_val);
    let beats = m.f_ord_greater_than(t_bool, final_sum, cur_best);
    let lbl_upd = m.alloc_id();
    let lbl_upd_m = m.alloc_id();
    m.selection_merge(lbl_upd_m, 0);
    m.branch_conditional(beats, lbl_upd, lbl_upd_m);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_upd.0]));
    m.store(var_best_val, final_sum);
    m.store(var_best_idx, cur_v);
    m.branch(lbl_upd_m);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_upd_m.0]));

    m.branch(lbl_v_cont);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_v_cont.0]));
    let next_v = m.iadd(t_u32, cur_v, c_local_size);
    m.store(var_v, next_v);
    m.branch(lbl_v_h);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_v_m.0]));
    let final_best_val = m.load(t_f32, var_best_val);
    let final_best_idx = m.load(t_u32, var_best_idx);
    let sv_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared_vals, &[lid]);
    let si_ptr = m.access_chain(t_ptr_wg_u32, gvar_shared_idxs, &[lid]);
    m.store(sv_ptr, final_best_val);
    m.store(si_ptr, final_best_idx);
    {
        let c_scope_wg = m.constant_u32(t_u32, scope::WORKGROUP);
        let c_sem = m.constant_u32(
            t_u32,
            memory_semantics::WORKGROUP_MEMORY | memory_semantics::ACQUIRE_RELEASE,
        );
        m.control_barrier(c_scope_wg, c_scope_wg, c_sem);
    }

    let c_half = m.constant_u32(t_u32, local_size_x / 2);
    m.store(var_step, c_half);

    let lbl_r_h = m.alloc_id();
    let lbl_r_c = m.alloc_id();
    let lbl_r_b = m.alloc_id();
    let lbl_r_cont = m.alloc_id();
    let lbl_r_m = m.alloc_id();

    m.branch(lbl_r_h);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_h.0]));
    m.loop_merge(lbl_r_m, lbl_r_cont, 0);
    m.branch(lbl_r_c);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_c.0]));
    let step_v = m.load(t_u32, var_step);
    let step_pos = m.u_less_than(t_bool, c_u32_0, step_v);
    m.branch_conditional(step_pos, lbl_r_b, lbl_r_m);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_b.0]));
    let lid_lt_step = m.u_less_than(t_bool, lid, step_v);
    let lbl_r_a = m.alloc_id();
    let lbl_r_am = m.alloc_id();
    m.selection_merge(lbl_r_am, 0);
    m.branch_conditional(lid_lt_step, lbl_r_a, lbl_r_am);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_a.0]));
    let other_lid = m.iadd(t_u32, lid, step_v);
    let self_v_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared_vals, &[lid]);
    let self_i_ptr = m.access_chain(t_ptr_wg_u32, gvar_shared_idxs, &[lid]);
    let other_v_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared_vals, &[other_lid]);
    let other_i_ptr = m.access_chain(t_ptr_wg_u32, gvar_shared_idxs, &[other_lid]);
    let self_v = m.load(t_f32, self_v_ptr);
    let self_i = m.load(t_u32, self_i_ptr);
    let other_v = m.load(t_f32, other_v_ptr);
    let other_i = m.load(t_u32, other_i_ptr);
    let other_beats = m.f_ord_greater_than(t_bool, other_v, self_v);
    let new_v = m.select(t_f32, other_beats, other_v, self_v);
    let new_i = m.select(t_u32, other_beats, other_i, self_i);
    m.store(self_v_ptr, new_v);
    m.store(self_i_ptr, new_i);
    m.branch(lbl_r_am);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_am.0]));
    {
        let c_scope_wg = m.constant_u32(t_u32, scope::WORKGROUP);
        let c_sem = m.constant_u32(
            t_u32,
            memory_semantics::WORKGROUP_MEMORY | memory_semantics::ACQUIRE_RELEASE,
        );
        m.control_barrier(c_scope_wg, c_scope_wg, c_sem);
    }
    m.branch(lbl_r_cont);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_cont.0]));
    let step_half = m.shift_right_logical(t_u32, step_v, c_u32_1);
    m.store(var_step, step_half);
    m.branch(lbl_r_h);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_m.0]));
    let is_lid_zero = m.u_less_than(t_bool, lid, c_u32_1);
    let lbl_w = m.alloc_id();
    let lbl_w_m = m.alloc_id();
    m.selection_merge(lbl_w_m, 0);
    m.branch_conditional(is_lid_zero, lbl_w, lbl_w_m);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_w.0]));
    let win_idx_ptr = m.access_chain(t_ptr_wg_u32, gvar_shared_idxs, &[c_u32_0]);
    let win_idx = m.load(t_u32, win_idx_ptr);
    let out_ptr = m.access_chain(t_ptr_sb_u32, gvar_out, &[c_u32_0, c_u32_0]);
    m.store(out_ptr, win_idx);
    m.branch(lbl_w_m);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_w_m.0]));

    m.ret();
    m.function_end();

    m.encode()
}

/// Chunked raw-row-major Q8_0 output matmul + per-workgroup argmax.
///
/// Binding 0: normalized hidden f32 array
/// Binding 1: raw GGML Q8_0 output table
/// Binding 2: partial best values f32 array; descriptor may be offset to one group slot
/// Binding 3: partial best token ids u32 array; descriptor may be offset to one group slot
/// Push constants: row_base, group_rows, hidden
pub fn emit_logit_argmax_q8_0_chunked(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_i32 = m.type_int(32, 1);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);

    let t_arr_f32 = m.type_runtime_array(t_f32);
    let t_arr_u32 = m.type_runtime_array(t_u32);
    let t_struct_hidden = m.type_struct(&[t_arr_f32]);
    let t_struct_table = m.type_struct(&[t_arr_u32]);
    let t_struct_partial_vals = m.type_struct(&[t_arr_f32]);
    let t_struct_partial_idxs = m.type_struct(&[t_arr_u32]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);

    let c_local_size = m.constant_u32(t_u32, local_size_x);
    let t_shared_arr_f32 = m.type_array(t_f32, c_local_size);
    let t_shared_arr_u32 = m.type_array(t_u32, c_local_size);
    let t_ptr_wg_arr_f32 = m.type_pointer(storage_class::WORKGROUP, t_shared_arr_f32);
    let t_ptr_wg_arr_u32 = m.type_pointer(storage_class::WORKGROUP, t_shared_arr_u32);
    let t_ptr_wg_f32 = m.type_pointer(storage_class::WORKGROUP, t_f32);
    let t_ptr_wg_u32 = m.type_pointer(storage_class::WORKGROUP, t_u32);

    let t_ptr_sb_hidden = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_hidden);
    let t_ptr_sb_table = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_table);
    let t_ptr_sb_partial_vals =
        m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_partial_vals);
    let t_ptr_sb_partial_idxs =
        m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_partial_idxs);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_8 = m.constant_u32(t_u32, 8);
    let c_u32_10 = m.constant_u32(t_u32, 10);
    let c_u32_13 = m.constant_u32(t_u32, 13);
    let c_u32_15 = m.constant_u32(t_u32, 15);
    let c_u32_23 = m.constant_u32(t_u32, 23);
    let c_u32_31 = m.constant_u32(t_u32, 31);
    let c_u32_32 = m.constant_u32(t_u32, 32);
    let c_u32_34 = m.constant_u32(t_u32, 34);
    let c_u32_112 = m.constant_u32(t_u32, 112);
    let c_u32_1f = m.constant_u32(t_u32, 0x1F);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3FF);
    let c_u32_ff = m.constant_u32(t_u32, 0xFF);
    let c_i32_24 = m.constant_u32(t_i32, 24);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);
    let c_f32_neg_inf = m.constant_f32(t_f32, f32::NEG_INFINITY);
    let c_f32_2pow_neg24 = m.constant_f32(t_f32, 5.9604644775390625e-8);

    m.decorate(t_struct_hidden, decoration::BLOCK, &[]);
    m.decorate(t_struct_table, decoration::BLOCK, &[]);
    m.decorate(t_struct_partial_vals, decoration::BLOCK, &[]);
    m.decorate(t_struct_partial_idxs, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_hidden, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_table, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_partial_vals, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_partial_idxs, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_u32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);

    let gvar_hidden = m.variable(t_ptr_sb_hidden, storage_class::STORAGE_BUFFER);
    let gvar_table = m.variable(t_ptr_sb_table, storage_class::STORAGE_BUFFER);
    let gvar_partial_vals = m.variable(t_ptr_sb_partial_vals, storage_class::STORAGE_BUFFER);
    let gvar_partial_idxs = m.variable(t_ptr_sb_partial_idxs, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let gvar_lid = m.variable(t_ptr_input_v3, storage_class::INPUT);
    let gvar_wgid = m.variable(t_ptr_input_v3, storage_class::INPUT);
    let gvar_shared_vals = m.variable(t_ptr_wg_arr_f32, storage_class::WORKGROUP);
    let gvar_shared_idxs = m.variable(t_ptr_wg_arr_u32, storage_class::WORKGROUP);

    m.decorate(gvar_hidden, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_hidden, decoration::BINDING, &[0]);
    m.decorate(gvar_table, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_table, decoration::BINDING, &[1]);
    m.decorate(gvar_partial_vals, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_partial_vals, decoration::BINDING, &[2]);
    m.decorate(gvar_partial_idxs, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_partial_idxs, decoration::BINDING, &[3]);
    m.decorate(
        gvar_lid,
        decoration::BUILTIN,
        &[builtin::LOCAL_INVOCATION_ID],
    );
    m.decorate(gvar_wgid, decoration::BUILTIN, &[builtin::WORKGROUP_ID]);

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_lid, gvar_wgid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    m.function(t_void, func_id, 0, t_fn_void);
    let lbl_entry = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let var_v = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_h = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_best_val = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_best_idx = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_step = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);

    let lid_vec = m.load(t_v3u32, gvar_lid);
    let lid = m.composite_extract(t_u32, lid_vec, 0);
    let pc_row_base_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let row_base = m.load(t_u32, pc_row_base_ptr);
    let pc_group_rows_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let group_rows = m.load(t_u32, pc_group_rows_ptr);
    let pc_hidden_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_2]);
    let hidden = m.load(t_u32, pc_hidden_ptr);
    let blocks_per_row = m.udiv(t_u32, hidden, c_u32_32);

    m.store(var_best_val, c_f32_neg_inf);
    m.store(var_best_idx, c_u32_0);
    m.store(var_v, lid);

    let read_byte = |m: &mut SpirvModule, byte_off: Id| -> Id {
        let word_idx = m.shift_right_logical(t_u32, byte_off, c_u32_2);
        let byte_in_word = m.bitwise_and(t_u32, byte_off, c_u32_3);
        let shift_bits = m.shift_left_logical(t_u32, byte_in_word, c_u32_3);
        let word_ptr = m.access_chain(t_ptr_sb_u32, gvar_table, &[c_u32_0, word_idx]);
        let word = m.load(t_u32, word_ptr);
        let shifted = m.shift_right_logical(t_u32, word, shift_bits);
        m.bitwise_and(t_u32, shifted, c_u32_ff)
    };

    let lbl_v_h = m.alloc_id();
    let lbl_v_c = m.alloc_id();
    let lbl_v_b = m.alloc_id();
    let lbl_v_cont = m.alloc_id();
    let lbl_v_m = m.alloc_id();

    m.branch(lbl_v_h);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_v_h.0]));
    m.loop_merge(lbl_v_m, lbl_v_cont, 0);
    m.branch(lbl_v_c);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_v_c.0]));
    let cur_v = m.load(t_u32, var_v);
    let v_in_bounds = m.u_less_than(t_bool, cur_v, group_rows);
    m.branch_conditional(v_in_bounds, lbl_v_b, lbl_v_m);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_v_b.0]));
    m.store(var_sum, c_f32_0);
    m.store(var_h, c_u32_0);

    let lbl_h_h = m.alloc_id();
    let lbl_h_c = m.alloc_id();
    let lbl_h_b = m.alloc_id();
    let lbl_h_cont = m.alloc_id();
    let lbl_h_m = m.alloc_id();

    m.branch(lbl_h_h);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_h_h.0]));
    m.loop_merge(lbl_h_m, lbl_h_cont, 0);
    m.branch(lbl_h_c);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_h_c.0]));
    let cur_h = m.load(t_u32, var_h);
    let block_in_bounds = m.u_less_than(t_bool, cur_h, blocks_per_row);
    m.branch_conditional(block_in_bounds, lbl_h_b, lbl_h_m);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_h_b.0]));
    let row_block_count_x_v = m.imul(t_u32, cur_v, blocks_per_row);
    let global_block_idx = m.iadd(t_u32, row_block_count_x_v, cur_h);
    let block_byte_off = m.imul(t_u32, global_block_idx, c_u32_34);
    let d_hi_byte_off = m.iadd(t_u32, block_byte_off, c_u32_1);

    let d_lo = read_byte(&mut m, block_byte_off);
    let d_hi = read_byte(&mut m, d_hi_byte_off);
    let d_hi_sh8 = m.shift_left_logical(t_u32, d_hi, c_u32_8);
    let d_f16_raw = m.bitwise_or(t_u32, d_lo, d_hi_sh8);
    let exp_raw = m.shift_right_logical(t_u32, d_f16_raw, c_u32_10);
    let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
    let mant = m.bitwise_and(t_u32, d_f16_raw, c_u32_3ff);
    let sign = m.shift_right_logical(t_u32, d_f16_raw, c_u32_15);
    let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
    let s_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
    let e_adj = m.iadd(t_u32, exp, c_u32_112);
    let e_part = m.shift_left_logical(t_u32, e_adj, c_u32_23);
    let m_part = m.shift_left_logical(t_u32, mant, c_u32_13);
    let bits_mid = m.bitwise_or(t_u32, s_part, e_part);
    let bits = m.bitwise_or(t_u32, bits_mid, m_part);
    let normal = m.bitcast(t_f32, bits);
    let mant_f = m.convert_u_to_f(t_f32, mant);
    let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
    let denorm_neg = m.fnegate(t_f32, denorm_abs);
    let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
    let denormal = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
    let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
    let d_f32 = m.select(t_f32, exp_nonzero, normal, denormal);

    let hidden_block_base = m.imul(t_u32, cur_h, c_u32_32);
    let mut block_sum = c_f32_0;
    for elem in 0..32u32 {
        let c_q_rel = m.constant_u32(t_u32, 2 + elem);
        let q_byte_off = m.iadd(t_u32, block_byte_off, c_q_rel);
        let q_byte = read_byte(&mut m, q_byte_off);
        let q_i32 = m.bitcast(t_i32, q_byte);
        let q_shl = m.shift_left_logical(t_i32, q_i32, c_i32_24);
        let q_signed = m.shift_right_arithmetic(t_i32, q_shl, c_i32_24);
        let q_f = m.convert_s_to_f(t_f32, q_signed);

        let hidden_idx = if elem == 0 {
            hidden_block_base
        } else {
            let c_elem = m.constant_u32(t_u32, elem);
            m.iadd(t_u32, hidden_block_base, c_elem)
        };
        let hv_ptr = m.access_chain(t_ptr_sb_f32, gvar_hidden, &[c_u32_0, hidden_idx]);
        let hv = m.load(t_f32, hv_ptr);
        let weight_val = m.fmul(t_f32, d_f32, q_f);
        let prod = m.fmul(t_f32, weight_val, hv);
        block_sum = m.fadd(t_f32, block_sum, prod);
    }
    let cur_sum = m.load(t_f32, var_sum);
    let new_sum = m.fadd(t_f32, cur_sum, block_sum);
    m.store(var_sum, new_sum);

    m.branch(lbl_h_cont);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_h_cont.0]));
    let next_h = m.iadd(t_u32, cur_h, c_u32_1);
    m.store(var_h, next_h);
    m.branch(lbl_h_h);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_h_m.0]));
    let final_sum = m.load(t_f32, var_sum);
    let cur_best = m.load(t_f32, var_best_val);
    let beats = m.f_ord_greater_than(t_bool, final_sum, cur_best);
    let lbl_upd = m.alloc_id();
    let lbl_upd_m = m.alloc_id();
    m.selection_merge(lbl_upd_m, 0);
    m.branch_conditional(beats, lbl_upd, lbl_upd_m);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_upd.0]));
    let global_v = m.iadd(t_u32, row_base, cur_v);
    m.store(var_best_val, final_sum);
    m.store(var_best_idx, global_v);
    m.branch(lbl_upd_m);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_upd_m.0]));

    m.branch(lbl_v_cont);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_v_cont.0]));
    let next_v = m.iadd(t_u32, cur_v, c_local_size);
    m.store(var_v, next_v);
    m.branch(lbl_v_h);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_v_m.0]));
    let final_best_val = m.load(t_f32, var_best_val);
    let final_best_idx = m.load(t_u32, var_best_idx);
    let sv_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared_vals, &[lid]);
    let si_ptr = m.access_chain(t_ptr_wg_u32, gvar_shared_idxs, &[lid]);
    m.store(sv_ptr, final_best_val);
    m.store(si_ptr, final_best_idx);
    {
        let c_scope_wg = m.constant_u32(t_u32, scope::WORKGROUP);
        let c_sem = m.constant_u32(
            t_u32,
            memory_semantics::WORKGROUP_MEMORY | memory_semantics::ACQUIRE_RELEASE,
        );
        m.control_barrier(c_scope_wg, c_scope_wg, c_sem);
    }

    let c_half = m.constant_u32(t_u32, local_size_x / 2);
    m.store(var_step, c_half);

    let lbl_r_h = m.alloc_id();
    let lbl_r_c = m.alloc_id();
    let lbl_r_b = m.alloc_id();
    let lbl_r_cont = m.alloc_id();
    let lbl_r_m = m.alloc_id();

    m.branch(lbl_r_h);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_h.0]));
    m.loop_merge(lbl_r_m, lbl_r_cont, 0);
    m.branch(lbl_r_c);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_c.0]));
    let step_v = m.load(t_u32, var_step);
    let step_pos = m.u_less_than(t_bool, c_u32_0, step_v);
    m.branch_conditional(step_pos, lbl_r_b, lbl_r_m);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_b.0]));
    let lid_lt_step = m.u_less_than(t_bool, lid, step_v);
    let lbl_r_a = m.alloc_id();
    let lbl_r_am = m.alloc_id();
    m.selection_merge(lbl_r_am, 0);
    m.branch_conditional(lid_lt_step, lbl_r_a, lbl_r_am);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_a.0]));
    let other_lid = m.iadd(t_u32, lid, step_v);
    let self_v_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared_vals, &[lid]);
    let self_i_ptr = m.access_chain(t_ptr_wg_u32, gvar_shared_idxs, &[lid]);
    let other_v_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared_vals, &[other_lid]);
    let other_i_ptr = m.access_chain(t_ptr_wg_u32, gvar_shared_idxs, &[other_lid]);
    let self_v = m.load(t_f32, self_v_ptr);
    let self_i = m.load(t_u32, self_i_ptr);
    let other_v = m.load(t_f32, other_v_ptr);
    let other_i = m.load(t_u32, other_i_ptr);
    let other_beats = m.f_ord_greater_than(t_bool, other_v, self_v);
    let new_v = m.select(t_f32, other_beats, other_v, self_v);
    let new_i = m.select(t_u32, other_beats, other_i, self_i);
    m.store(self_v_ptr, new_v);
    m.store(self_i_ptr, new_i);
    m.branch(lbl_r_am);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_am.0]));
    {
        let c_scope_wg = m.constant_u32(t_u32, scope::WORKGROUP);
        let c_sem = m.constant_u32(
            t_u32,
            memory_semantics::WORKGROUP_MEMORY | memory_semantics::ACQUIRE_RELEASE,
        );
        m.control_barrier(c_scope_wg, c_scope_wg, c_sem);
    }
    m.branch(lbl_r_cont);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_cont.0]));
    let step_half = m.shift_right_logical(t_u32, step_v, c_u32_1);
    m.store(var_step, step_half);
    m.branch(lbl_r_h);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_m.0]));
    let is_lid_zero = m.u_less_than(t_bool, lid, c_u32_1);
    let lbl_w = m.alloc_id();
    let lbl_w_m = m.alloc_id();
    m.selection_merge(lbl_w_m, 0);
    m.branch_conditional(is_lid_zero, lbl_w, lbl_w_m);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_w.0]));
    let win_val_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared_vals, &[c_u32_0]);
    let win_idx_ptr = m.access_chain(t_ptr_wg_u32, gvar_shared_idxs, &[c_u32_0]);
    let win_val = m.load(t_f32, win_val_ptr);
    let win_idx = m.load(t_u32, win_idx_ptr);
    let out_val_ptr = m.access_chain(t_ptr_sb_f32, gvar_partial_vals, &[c_u32_0, c_u32_0]);
    let out_idx_ptr = m.access_chain(t_ptr_sb_u32, gvar_partial_idxs, &[c_u32_0, c_u32_0]);
    m.store(out_val_ptr, win_val);
    m.store(out_idx_ptr, win_idx);
    m.branch(lbl_w_m);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_w_m.0]));

    m.ret();
    m.function_end();

    m.encode()
}

/// Reduce partial argmax candidates to the final token id.
///
/// Binding 0: partial values f32 array
/// Binding 1: partial token ids u32 array
/// Binding 2: output u32 array with at least one element
/// Push constants: count u32
pub fn emit_argmax_pairs_f32(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);

    let t_arr_f32 = m.type_runtime_array(t_f32);
    let t_arr_u32 = m.type_runtime_array(t_u32);
    let t_struct_vals = m.type_struct(&[t_arr_f32]);
    let t_struct_idxs = m.type_struct(&[t_arr_u32]);
    let t_struct_out = m.type_struct(&[t_arr_u32]);
    let t_struct_pc = m.type_struct(&[t_u32]);

    let c_local_size = m.constant_u32(t_u32, local_size_x);
    let t_shared_arr_f32 = m.type_array(t_f32, c_local_size);
    let t_shared_arr_u32 = m.type_array(t_u32, c_local_size);
    let t_ptr_wg_arr_f32 = m.type_pointer(storage_class::WORKGROUP, t_shared_arr_f32);
    let t_ptr_wg_arr_u32 = m.type_pointer(storage_class::WORKGROUP, t_shared_arr_u32);
    let t_ptr_wg_f32 = m.type_pointer(storage_class::WORKGROUP, t_f32);
    let t_ptr_wg_u32 = m.type_pointer(storage_class::WORKGROUP, t_u32);

    let t_ptr_sb_vals = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_vals);
    let t_ptr_sb_idxs = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_idxs);
    let t_ptr_sb_out = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_out);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_f32_neg_inf = m.constant_f32(t_f32, f32::NEG_INFINITY);

    m.decorate(t_struct_vals, decoration::BLOCK, &[]);
    m.decorate(t_struct_idxs, decoration::BLOCK, &[]);
    m.decorate(t_struct_out, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_vals, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_idxs, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_out, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_u32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);

    let gvar_vals = m.variable(t_ptr_sb_vals, storage_class::STORAGE_BUFFER);
    let gvar_idxs = m.variable(t_ptr_sb_idxs, storage_class::STORAGE_BUFFER);
    let gvar_out = m.variable(t_ptr_sb_out, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let gvar_lid = m.variable(t_ptr_input_v3, storage_class::INPUT);
    let gvar_shared_vals = m.variable(t_ptr_wg_arr_f32, storage_class::WORKGROUP);
    let gvar_shared_idxs = m.variable(t_ptr_wg_arr_u32, storage_class::WORKGROUP);

    m.decorate(gvar_vals, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_vals, decoration::BINDING, &[0]);
    m.decorate(gvar_idxs, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_idxs, decoration::BINDING, &[1]);
    m.decorate(gvar_out, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_out, decoration::BINDING, &[2]);
    m.decorate(
        gvar_lid,
        decoration::BUILTIN,
        &[builtin::LOCAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_lid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    m.function(t_void, func_id, 0, t_fn_void);
    let lbl_entry = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let var_i = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_best_val = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_best_idx = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_step = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);

    let lid_vec = m.load(t_v3u32, gvar_lid);
    let lid = m.composite_extract(t_u32, lid_vec, 0);
    let pc_count_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let count = m.load(t_u32, pc_count_ptr);

    m.store(var_best_val, c_f32_neg_inf);
    m.store(var_best_idx, c_u32_0);
    m.store(var_i, lid);

    let lbl_scan_h = m.alloc_id();
    let lbl_scan_c = m.alloc_id();
    let lbl_scan_b = m.alloc_id();
    let lbl_scan_cont = m.alloc_id();
    let lbl_scan_m = m.alloc_id();

    m.branch(lbl_scan_h);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_scan_h.0]));
    m.loop_merge(lbl_scan_m, lbl_scan_cont, 0);
    m.branch(lbl_scan_c);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_scan_c.0]));
    let cur_i = m.load(t_u32, var_i);
    let in_bounds = m.u_less_than(t_bool, cur_i, count);
    m.branch_conditional(in_bounds, lbl_scan_b, lbl_scan_m);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_scan_b.0]));
    let val_ptr = m.access_chain(t_ptr_sb_f32, gvar_vals, &[c_u32_0, cur_i]);
    let val = m.load(t_f32, val_ptr);
    let idx_ptr = m.access_chain(t_ptr_sb_u32, gvar_idxs, &[c_u32_0, cur_i]);
    let token_idx = m.load(t_u32, idx_ptr);
    let cur_best = m.load(t_f32, var_best_val);
    let beats = m.f_ord_greater_than(t_bool, val, cur_best);
    let lbl_upd = m.alloc_id();
    let lbl_upd_m = m.alloc_id();
    m.selection_merge(lbl_upd_m, 0);
    m.branch_conditional(beats, lbl_upd, lbl_upd_m);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_upd.0]));
    m.store(var_best_val, val);
    m.store(var_best_idx, token_idx);
    m.branch(lbl_upd_m);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_upd_m.0]));
    m.branch(lbl_scan_cont);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_scan_cont.0]));
    let next_i = m.iadd(t_u32, cur_i, c_local_size);
    m.store(var_i, next_i);
    m.branch(lbl_scan_h);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_scan_m.0]));
    let final_best_val = m.load(t_f32, var_best_val);
    let final_best_idx = m.load(t_u32, var_best_idx);
    let sv_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared_vals, &[lid]);
    let si_ptr = m.access_chain(t_ptr_wg_u32, gvar_shared_idxs, &[lid]);
    m.store(sv_ptr, final_best_val);
    m.store(si_ptr, final_best_idx);
    {
        let c_scope_wg = m.constant_u32(t_u32, scope::WORKGROUP);
        let c_sem = m.constant_u32(
            t_u32,
            memory_semantics::WORKGROUP_MEMORY | memory_semantics::ACQUIRE_RELEASE,
        );
        m.control_barrier(c_scope_wg, c_scope_wg, c_sem);
    }

    let c_half = m.constant_u32(t_u32, local_size_x / 2);
    m.store(var_step, c_half);

    let lbl_r_h = m.alloc_id();
    let lbl_r_c = m.alloc_id();
    let lbl_r_b = m.alloc_id();
    let lbl_r_cont = m.alloc_id();
    let lbl_r_m = m.alloc_id();

    m.branch(lbl_r_h);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_h.0]));
    m.loop_merge(lbl_r_m, lbl_r_cont, 0);
    m.branch(lbl_r_c);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_c.0]));
    let step_v = m.load(t_u32, var_step);
    let step_pos = m.u_less_than(t_bool, c_u32_0, step_v);
    m.branch_conditional(step_pos, lbl_r_b, lbl_r_m);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_b.0]));
    let lid_lt_step = m.u_less_than(t_bool, lid, step_v);
    let lbl_r_a = m.alloc_id();
    let lbl_r_am = m.alloc_id();
    m.selection_merge(lbl_r_am, 0);
    m.branch_conditional(lid_lt_step, lbl_r_a, lbl_r_am);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_a.0]));
    let other_lid = m.iadd(t_u32, lid, step_v);
    let self_v_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared_vals, &[lid]);
    let self_i_ptr = m.access_chain(t_ptr_wg_u32, gvar_shared_idxs, &[lid]);
    let other_v_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared_vals, &[other_lid]);
    let other_i_ptr = m.access_chain(t_ptr_wg_u32, gvar_shared_idxs, &[other_lid]);
    let self_v = m.load(t_f32, self_v_ptr);
    let self_i = m.load(t_u32, self_i_ptr);
    let other_v = m.load(t_f32, other_v_ptr);
    let other_i = m.load(t_u32, other_i_ptr);
    let other_beats = m.f_ord_greater_than(t_bool, other_v, self_v);
    let new_v = m.select(t_f32, other_beats, other_v, self_v);
    let new_i = m.select(t_u32, other_beats, other_i, self_i);
    m.store(self_v_ptr, new_v);
    m.store(self_i_ptr, new_i);
    m.branch(lbl_r_am);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_am.0]));
    {
        let c_scope_wg = m.constant_u32(t_u32, scope::WORKGROUP);
        let c_sem = m.constant_u32(
            t_u32,
            memory_semantics::WORKGROUP_MEMORY | memory_semantics::ACQUIRE_RELEASE,
        );
        m.control_barrier(c_scope_wg, c_scope_wg, c_sem);
    }
    m.branch(lbl_r_cont);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_cont.0]));
    let step_half = m.shift_right_logical(t_u32, step_v, c_u32_1);
    m.store(var_step, step_half);
    m.branch(lbl_r_h);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_m.0]));
    let is_lid_zero = m.u_less_than(t_bool, lid, c_u32_1);
    let lbl_w = m.alloc_id();
    let lbl_w_m = m.alloc_id();
    m.selection_merge(lbl_w_m, 0);
    m.branch_conditional(is_lid_zero, lbl_w, lbl_w_m);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_w.0]));
    let win_idx_ptr = m.access_chain(t_ptr_wg_u32, gvar_shared_idxs, &[c_u32_0]);
    let win_idx = m.load(t_u32, win_idx_ptr);
    let out_ptr = m.access_chain(t_ptr_sb_u32, gvar_out, &[c_u32_0, c_u32_0]);
    m.store(out_ptr, win_idx);
    m.branch(lbl_w_m);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_w_m.0]));

    m.ret();
    m.function_end();

    m.encode()
}

/// NEOX RoPE (Rotary Position Embedding) in-place rotation shader.
///
/// Performs the NEOX-style rotation for a single buffer containing
/// `seq_len × num_heads × head_dim` f32 elements in row-major order:
///
/// ```text
/// for t in 0..seq_len:
///   for h in 0..num_heads:
///     for p in 0..head_dim/2:
///       freq = base_freq ^ (-(2*p) / head_dim)
///       angle = (pos_offset + t) * freq
///       i_x = (t * num_heads + h) * head_dim + p
///       i_y = i_x + head_dim/2
///       (x, y) = (buf[i_x], buf[i_y])
///       buf[i_x] = x*cos(angle) - y*sin(angle)
///       buf[i_y] = x*sin(angle) + y*cos(angle)
/// ```
///
/// Bindings:
///   binding 0: target (f32 runtime_array) — in-place rotation target
///
/// Push constants (5 × u32/f32, 20 bytes total):
///   [0] head_dim   u32
///   [1] num_heads  u32
///   [2] pos_offset u32
///   [3] seq_len    u32  (used for bounds check only)
///   [4] base_freq  f32  (e.g. 10000.0, stored as raw bits)
///
/// Dispatch: `ceil(seq_len * num_heads * head_dim/2 / local_size_x)` groups × 1 × 1.
/// Each thread handles one (t, h, p) triple via 1D decomposition.
///
/// GLSL.std.450 ext-inst numbers used: Log2=30, Exp2=29, Sin=13, Cos=14.
pub fn emit_rope_apply(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    let glsl = m.ext_inst_import("GLSL.std.450");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_u32 = m.type_int(32, 0);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_arr_f32 = m.type_runtime_array(t_f32);

    // Single binding: target buffer (f32 array, read+write in-place).
    let t_struct_target = m.type_struct(&[t_arr_f32]);
    // Push constants:
    // [head_dim, num_heads, pos_offset, seq_len, base_freq_bits, rot_dim, neox]
    // All slots occupy u32-sized words; base_freq is bitcast u32→f32 at runtime.
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32, t_u32, t_u32, t_u32, t_u32]);

    let t_ptr_sb_target = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_target);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_4 = m.constant_u32(t_u32, 4);
    let c_u32_5 = m.constant_u32(t_u32, 5);
    let c_u32_6 = m.constant_u32(t_u32, 6);
    let c_f32_2 = m.constant_f32(t_f32, 2.0);
    let c_f32_neg1 = m.constant_f32(t_f32, -1.0);

    m.decorate(t_struct_target, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_target, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);
    m.member_decorate(t_struct_pc, 3, decoration::OFFSET, &[12]);
    m.member_decorate(t_struct_pc, 4, decoration::OFFSET, &[16]);
    m.member_decorate(t_struct_pc, 5, decoration::OFFSET, &[20]);
    m.member_decorate(t_struct_pc, 6, decoration::OFFSET, &[24]);

    let gvar_target = m.variable(t_ptr_sb_target, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let gvar_gid = m.variable(t_ptr_input_v3, storage_class::INPUT);

    m.decorate(gvar_target, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_target, decoration::BINDING, &[0]);
    m.decorate(
        gvar_gid,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_gid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    m.function(t_void, func_id, 0, t_fn_void);
    let lbl_entry = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    // Load push constants.
    let pc_hd_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let head_dim = m.load(t_u32, pc_hd_ptr);
    let pc_nh_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let num_heads = m.load(t_u32, pc_nh_ptr);
    let pc_pos_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_2]);
    let pos_offset = m.load(t_u32, pc_pos_ptr);
    let pc_sl_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_3]);
    let seq_len = m.load(t_u32, pc_sl_ptr);
    // base_freq is stored as raw f32 bits in slot [4]; bitcast u32 → f32.
    let pc_bf_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_4]);
    let base_freq_bits = m.load(t_u32, pc_bf_ptr);
    let base_freq = m.bitcast(t_f32, base_freq_bits);
    let pc_rot_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_5]);
    let rot_dim_raw = m.load(t_u32, pc_rot_ptr);
    let pc_neox_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_6]);
    let neox_raw = m.load(t_u32, pc_neox_ptr);

    // gid = global_invocation_id.x
    let gid_vec = m.load(t_v3u32, gvar_gid);
    let gid = m.composite_extract(t_u32, gid_vec, 0);

    let t_bool = m.type_bool();
    let rot_nonzero = m.i_not_equal(t_bool, rot_dim_raw, c_u32_0);
    let rot_dim = m.select(t_u32, rot_nonzero, rot_dim_raw, head_dim);
    let neox_enabled = m.i_not_equal(t_bool, neox_raw, c_u32_0);

    // total = seq_len * num_heads * (rot_dim / 2)
    let half_hd = m.shift_right_logical(t_u32, head_dim, c_u32_1); // head_dim >> 1 = head_dim/2
    let pair_count = m.shift_right_logical(t_u32, rot_dim, c_u32_1);
    let heads_x_pairs = m.imul(t_u32, num_heads, pair_count);
    let total = m.imul(t_u32, seq_len, heads_x_pairs);

    let in_bounds = m.u_less_than(t_bool, gid, total);
    let lbl_body = m.alloc_id();
    let lbl_end = m.alloc_id();
    m.selection_merge(lbl_end, 0);
    m.branch_conditional(in_bounds, lbl_body, lbl_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_body.0]));

    // Decompose gid into (t, h, p).
    // gid = t * (num_heads * pair_count) + h * pair_count + p
    let t_idx = m.udiv(t_u32, gid, heads_x_pairs); // t
    let rem_th = m.umod(t_u32, gid, heads_x_pairs); // h * pair_count + p
    let h_idx = m.udiv(t_u32, rem_th, pair_count); // h
    let p_idx = m.umod(t_u32, rem_th, pair_count); // p

    // Compute freq = base_freq ^ (-(2*p) / denom).
    // Adjacent-pair partial RoPE uses denom=rot_dim; NEOX uses denom=head_dim.
    let p_f = m.convert_u_to_f(t_f32, p_idx);
    let denom_dim = m.select(t_u32, neox_enabled, head_dim, rot_dim);
    let denom_dim_f = m.convert_u_to_f(t_f32, denom_dim);
    let two_p = m.fmul(t_f32, c_f32_2, p_f); // 2*p
    let neg_two_p = m.fmul(t_f32, c_f32_neg1, two_p); // -(2*p)
    let exponent = m.fdiv(t_f32, neg_two_p, denom_dim_f); // -(2*p)/denom
    let log2_base = m.ext_inst(t_f32, glsl, 30, &[base_freq]); // GLSL Log2 = 30
    let pow_arg = m.fmul(t_f32, log2_base, exponent); // log2(base) * exponent
    let freq = m.ext_inst(t_f32, glsl, 29, &[pow_arg]); // GLSL Exp2 = 29 → base^exponent

    // angle = (pos_offset + t) * freq
    let pos_u = m.iadd(t_u32, pos_offset, t_idx); // pos_offset + t
    let pos_f = m.convert_u_to_f(t_f32, pos_u); // float
    let angle = m.fmul(t_f32, pos_f, freq);

    let cos_a = m.ext_inst(t_f32, glsl, 14, &[angle]); // GLSL Cos = 14
    let sin_a = m.ext_inst(t_f32, glsl, 13, &[angle]); // GLSL Sin = 13

    // Buffer indices:
    //   adjacent: i_x = base + p*2, i_y = i_x + 1
    //   NEOX:     i_x = base + p,   i_y = base + head_dim/2 + p
    let t_x_nh = m.imul(t_u32, t_idx, num_heads);
    let th = m.iadd(t_u32, t_x_nh, h_idx);
    let th_x_hd = m.imul(t_u32, th, head_dim);
    let adjacent_pair = m.imul(t_u32, p_idx, c_u32_2);
    let adjacent_x = m.iadd(t_u32, th_x_hd, adjacent_pair);
    let adjacent_y = m.iadd(t_u32, adjacent_x, c_u32_1);
    let neox_x = m.iadd(t_u32, th_x_hd, p_idx);
    let neox_base_y = m.iadd(t_u32, th_x_hd, half_hd);
    let neox_y = m.iadd(t_u32, neox_base_y, p_idx);
    let i_x = m.select(t_u32, neox_enabled, neox_x, adjacent_x);
    let i_y = m.select(t_u32, neox_enabled, neox_y, adjacent_y);

    // Load x (at p) and y (at p + head_dim/2).
    let ptr_x = m.access_chain(t_ptr_sb_f32, gvar_target, &[c_u32_0, i_x]);
    let x_val = m.load(t_f32, ptr_x);
    let ptr_y = m.access_chain(t_ptr_sb_f32, gvar_target, &[c_u32_0, i_y]);
    let y_val = m.load(t_f32, ptr_y);

    // Rotation: x_new = x*cos - y*sin, y_new = x*sin + y*cos
    let x_cos = m.fmul(t_f32, x_val, cos_a);
    let y_sin = m.fmul(t_f32, y_val, sin_a);
    let x_new = m.fsub(t_f32, x_cos, y_sin);

    let x_sin = m.fmul(t_f32, x_val, sin_a);
    let y_cos = m.fmul(t_f32, y_val, cos_a);
    let y_new = m.fadd(t_f32, x_sin, y_cos);

    // Write back in-place.
    m.store(ptr_x, x_new);
    m.store(ptr_y, y_new);

    m.branch(lbl_end);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_end.0]));
    m.ret();
    m.function_end();

    m.encode()
}
