use crate::codegen::ast::{
    Argument, ArgumentFlag, DeclAnnotation, Program, RelationFlags, Rule, Term, Type,
};
use crate::codegen::diag::DiagnosticContext;
use crate::codegen::ir::{Metadata, Relation, RelationBuilder, SymbolTable};
use crate::error::{Result, SemanticError};
use std::borrow::Borrow;
use std::cell::RefCell;
use std::fmt::{Arguments, Write};
use std::iter::zip;
use std::mem;
use std::rc::Rc;

struct Printer {
    pub buf: Vec<u8>,
}

impl Write for Printer {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        use std::io::Write;
        self.buf
            .write(s.as_bytes())
            .map(|_| ())
            .map_err(|_| std::fmt::Error)
    }
}

pub struct LKMCodeGen<'a> {
    prog: Program<'a>,
    out: &'a RefCell<dyn Write>,
    relations: Vec<Rc<Relation>>,
    sym_table: SymbolTable,
    diag: Rc<DiagnosticContext<'a>>,
}

struct CodeGenContext<'a> {
    out: &'a RefCell<dyn Write>,
    top_rel: Rc<Relation>,
    sym_table: SymbolTable,
    diag: Rc<DiagnosticContext<'a>>,
}

struct Mangler {}

impl Mangler {
    pub fn wrap_relation_name(n: &String) -> String {
        n.clone()
    }
    pub fn wrap_type(n: &Type) -> String {
        match n {
            Type::U32 => "u32".to_string(),
            Type::U64 => "u64".to_string(),
            Type::OPAQUE(x) => x.clone(),
            Type::VOID => "void".to_string(),
            Type::BOOL => "bool".to_string(),
            Type::STRING => unreachable!(),
        }
    }
}

impl<'a> LKMCodeGen<'a> {
    pub fn new(out: &'a RefCell<dyn Write>, prog: Program<'a>) -> LKMCodeGen<'a> {
        let ret = LKMCodeGen {
            prog,
            out,
            relations: vec![],
            sym_table: SymbolTable::new(),
            diag: Rc::new(DiagnosticContext::default()),
        };
        ret
    }

    pub fn compile(&'a mut self) -> Result<()> {
        let diag = self.diag.clone();
        let r = self.compile_impl();
        match r {
            Ok(_) => Ok(()),
            Err(e) => Err(diag.err_with_location(e)),
        }
    }

    fn compile_impl(&'a mut self) -> Result<()> {
        self.populate_declarations()?;

        for rel in self
            .relations
            .iter()
            .filter(|x| x.flag.contains(RelationFlags::GLOBAL))
        {
            let rules = self
                .prog
                .rules
                .iter()
                .filter(|x| x.head.relation.id == rel.name)
                .collect::<Vec<&Rule>>();

            let ctx = CodeGenContext {
                out: self.out,
                top_rel: rel.clone(),
                sym_table: self.sym_table.clone(),
                diag: self.diag.clone(),
            };
            ctx.compile_relation(rel.as_ref(), rules.as_slice())?;
        }
        Ok(())
    }

    fn populate_declarations(&mut self) -> Result<()> {
        let mut relations = Vec::new();
        let sym = &self.sym_table;
        sym.enter();
        for d in &self.prog.declarations {
            let out_args: Vec<&Argument> = d
                .params
                .iter()
                .filter(|x| x.flag.contains(ArgumentFlag::OUT))
                .collect();
            let (rel_ty, flag) = if out_args.len() == 0 {
                Ok((Type::VOID, d.flags.clone()))
            } else if out_args.len() == 1 {
                Ok((
                    out_args.first().unwrap().ty.clone(),
                    d.flags.union(RelationFlags::HAS_SINGLE_OUT_ARG),
                ))
            } else {
                Err(SemanticError::unimplemented(
                    "Unimplemented support for multiple output args".to_string(),
                ))
            }?;

            let mut b = &mut RelationBuilder::new(d.name.id.as_str(), flag, rel_ty);
            for a in &d.params {
                b = b.arg_with_flags(a.name.id.as_str(), a.ty.clone(), a.flag);
            }
            if !d.annotations.is_empty() {
                let p = d.annotations.first().unwrap();
                match p {
                    DeclAnnotation::CFuncName(x) => {
                        b = b.metadata(Metadata { c_name: x.clone() });
                    }
                }
            }
            let rel = Rc::new(b.build());
            relations.push(rel.clone());
            sym.add_relation_decl(rel.clone()).unwrap();
        }
        let cut = Rc::new(
            RelationBuilder::new(
                "cut",
                RelationFlags::ACTION | RelationFlags::INTRINSIC,
                Type::VOID,
            )
            .build(),
        );
        relations.push(cut.clone());
        sym.add_relation_decl(cut.clone()).unwrap();

        let _ = mem::replace(&mut self.relations, relations);
        Ok(())
    }
}

impl<'a> CodeGenContext<'a> {
    // TODO: It should be a DAG instead of passing in a number of rules.
    pub fn compile_relation(&self, relation: &Relation, rules: &[&'a Rule]) -> Result<()> {
        let sym = &self.sym_table;
        sym.enter();

        self.write_fmt(format_args!(
            "{} {} (",
            Mangler::wrap_type(&relation.return_ty),
            Mangler::wrap_relation_name(&relation.name)
        ))?;
        let mut sep = false;
        for arg in &relation.args {
            if arg.flag.contains(ArgumentFlag::OUT) {
                continue;
            }

            sym.add_formal_arg(arg)?;
            if sep {
                self.write_str(", ")?;
            }
            self.write_fmt(format_args!("{} {}", Mangler::wrap_type(&arg.ty), arg.name))?;
            sep = true;
        }
        self.write_str(") {\n")?;

        if let Some(out_arg_idx) = relation.out_arg_idx {
            let arg = relation.args.get(out_arg_idx).unwrap();
            self.print_var_decl(&arg.ty, arg.name.as_str())?;
            sym.add_formal_arg(arg)?;
        }

        for r in rules {
            assert_eq!(relation.name, r.head.relation.id);
            sym.enter();
            let ret = self.compile_rule(relation, r);
            sym.leave();
            ret?;
        }

        self.write_str("}\n\n")?;
        Ok(())
    }

    fn compile_rule(&self, relation: &Relation, r: &'a Rule) -> Result<()> {
        assert_eq!(relation.name, r.head.relation.id);
        let sym_table = self.sym_table.borrow();

        // Bind the declaration of the specific rule
        if r.head.terms.len() != relation.args.len() {
            return Err(SemanticError::mismatched_argument(format!(
                "Expected {} arguments",
                relation.args.len()
            )));
        }

        zip(relation.args.iter(), r.head.terms.iter()).try_for_each(|(decl, param)| {
            let formal = sym_table.lookup(decl.name.as_str()).unwrap();
            match param {
                Term::IDENTIFIER(x) => {
                    sym_table.bind(x.id.as_str(), formal);
                    Ok(())
                }
                _ => Err(SemanticError::mismatched_argument(format!(
                    "Expected identifier not `{:?}`",
                    param
                ))),
            }
        })?;

        let mut it = r.predicates.iter();
        let mut cond = vec![];
        let mut nested_level = 0;
        while let Some(atom) = it.next() {
            self.diag.set_location(&atom.pos);
            let rel = sym_table
                .lookup(atom.relation.id.as_str())
                .ok_or(SemanticError::unresolved_symbol(atom.relation.id.clone()))?
                .as_relation()
                .ok_or(SemanticError::unexpected_type(
                    "Expect a relation".to_string(),
                ))?;

            let is_predicate = rel.flag.contains(RelationFlags::PREDICATE);
            if is_predicate {
                cond.push(self.compile_predicate(sym_table, rel.borrow(), &atom.terms)?);
                continue;
            }

            if !cond.is_empty() {
                self.write_fmt(format_args!("if ({}) {{\n", cond.join(" && ")))?;
                cond.clear();
                nested_level += 1;
                sym_table.enter();
            }

            self.compile_action(sym_table, rel.borrow(), &atom.terms)?;
        }
        for _ in 0..nested_level {
            sym_table.leave();
            self.write_str("}\n")?;
        }
        Ok(())
    }

    fn compile_action(
        &self,
        sym_table: &SymbolTable,
        action: &Relation,
        terms: &'a [Term],
    ) -> Result<()> {
        assert!(
            action.flag.contains(RelationFlags::ACTION)
                && !action.flag.contains(RelationFlags::HAS_MULTI_OUT_ARGS)
        );

        if action.flag.contains(RelationFlags::INTRINSIC) {
            return self.compile_intrinsic(sym_table, action, terms);
        }

        let func_name = match action.md.as_ref() {
            None => action.name.as_str(),
            Some(x) => x.c_name.as_str(),
        };
        if action.flag.contains(RelationFlags::HAS_SINGLE_OUT_ARG) {
            let name = self.decl_out_var(sym_table, action, terms)?;

            self.write_fmt(format_args!(
                "{} = {}({});\n",
                name,
                func_name,
                self.bind_args(sym_table, action, terms)?
            ))?;
        } else {
            assert_eq!(action.return_ty, Type::VOID);
            self.write_fmt(format_args!(
                "{}({});\n",
                func_name,
                self.bind_args(sym_table, action, terms)?
            ))?;
        }

        Ok(())
    }

    fn decl_out_var(
        &self,
        sym_table: &SymbolTable,
        action: &Relation,
        terms: &'a [Term],
    ) -> Result<String> {
        let out_arg_idx = action.out_arg_idx.unwrap();
        let out_arg = action.args.get(out_arg_idx).unwrap();
        let out_term = terms.get(out_arg_idx);
        if let Some(Term::IDENTIFIER(t)) = out_term {
            self.diag.set_location(&t.pos);
            if let Some(bound) = sym_table.lookup(t.id.as_str()) {
                Ok(bound.as_varexpr().unwrap().name)
            } else {
                let name = sym_table.add_unique_variable(t.id.as_str(), out_arg.ty.clone())?;
                self.print_var_decl(&out_arg.ty, name.as_str())?;
                Ok(name)
            }
        } else {
            Err(SemanticError::unexpected_type(format!(
                "expect an identifier on output argument {}",
                out_arg_idx
            )))
        }
    }

    fn print_var_decl(&self, ty: &Type, name: &str) -> Result<()> {
        self.write_fmt(format_args!("{} {};\n", Mangler::wrap_type(ty), name))?;
        Ok(())
    }

    fn compile_intrinsic(
        &self,
        sym_table: &SymbolTable,
        action: &Relation,
        _: &[Term],
    ) -> Result<()> {
        if action.name == "cut" {
            if self.top_rel.return_ty == Type::VOID {
                self.write_str("return;\n")?;
            } else {
                let out_arg = self
                    .top_rel
                    .out_arg_idx
                    .map(|x| self.top_rel.args.get(x).unwrap())
                    .unwrap();
                let bound_var = sym_table.lookup(out_arg.name.as_str());
                self.write_fmt(format_args!("return {};\n", bound_var.unwrap().get_name()))?;
            }
            return Ok(());
        }
        todo!()
    }

    fn compile_predicate(
        &self,
        sym_table: &SymbolTable,
        rel: &Relation,
        actuals: &'a [Term],
    ) -> Result<String> {
        assert!(rel.flag.contains(RelationFlags::PREDICATE));
        Ok(format!(
            "{}({})",
            rel.md.as_ref().unwrap().c_name,
            self.bind_args(sym_table, rel, actuals)?
        ))
    }

    fn bind_args(
        &self,
        sym_table: &SymbolTable,
        rel: &Relation,
        actual_args: &'a [Term],
    ) -> Result<String> {
        let mut actuals = vec![];
        if rel.args.len() != actual_args.len() {
            return Err(SemanticError::mismatched_argument(format!(
                "Invalid number of arguments {} for relation {}",
                rel.args.len(),
                rel.name
            )));
        }

        zip(rel.args.iter(), actual_args.iter()).try_for_each(|(formal, term)| {
            if !formal.flag.contains(ArgumentFlag::OUT) {
                match term {
                    Term::IDENTIFIER(x) => {
                        self.diag.set_location(&x.pos);
                        let actual = sym_table
                            .lookup(x.id.as_str())
                            .ok_or(SemanticError::unresolved_symbol(x.id.clone()))?
                            .as_varexpr()
                            .ok_or(SemanticError::unexpected_type(
                                "Expect type varexpr".to_string(),
                            ))?;
                        if !actual.ty.can_cast_to(&formal.ty) {
                            return Err(SemanticError::unexpected_type(format!(
                                "incompatible type cast from {} to {}",
                                actual.ty, formal.ty
                            )));
                        }
                        actuals.push(actual.name.clone());
                    }
                    Term::STRING(x) => actuals.push(x.clone()),
                    Term::INTEGER(x) => actuals.push(x.to_string()),
                }
            }
            Result::<()>::Ok(())
        })?;

        Ok(actuals.join(", "))
    }

    fn write_fmt(&self, args: Arguments<'_>) -> std::fmt::Result {
        self.out.borrow_mut().write_fmt(args)
    }

    fn write_str(&self, s: &str) -> std::fmt::Result {
        self.out.borrow_mut().write_str(s)
    }
}

#[cfg(test)]
mod tests {
    use crate::codegen::lkm::*;
    use crate::codegen::parser;
    use crate::error::Span;
    use std::mem;

    #[test]
    fn test_gen() {
        let prog_str = Span::new_extra(
            r#"
       cp_resume_rreg(adev, reg, val, acc_flags) :-
         soc15_reg(reg, "GC", 0, "mmCP_HQD_ACTIVE"),
         read_reg(adev, reg, val, acc_flags),
         update_gfx_state(adev, "HQD_ACTIVE", val),
         cut() ."#,
            "foo.dl",
        );
        let r = parser::program(prog_str).unwrap();
        assert!(r.0.is_empty());
        assert_eq!(r.1.rules.len(), 1);
        let writer = RefCell::new(Printer { buf: vec![] });
        let mut codegen = LKMCodeGen::new(&writer, r.1);
        let r = codegen.compile();
        let buf = mem::replace(&mut writer.borrow_mut().buf, vec![]);
        let dump = std::str::from_utf8(buf.as_slice()).unwrap();
        println!("Result: {}", dump);
        assert!(r.is_ok());
    }

    #[test]
    fn test_call() {
        let prog_str = Span::new_extra(
            r#"
            #[c_name="bitand"]
            extern action bitand(ret: [out] u32, val: u32, mask: u32) .
            output action cp_resume_rreg(adev: opaque("struct amdgpu_device *"), reg: u32, val: [out] u32, acc_flags: u32) .
            #[c_name="amdgpu_device_rreg"]
            extern action read_reg(adev: opaque("struct amdgpu_device *"), reg: u32, val: [out] u32, acc_flags: u32) .
            #[c_name="CP_RESUME_UPDATE_STATE"]
            extern action update_gfx_state(adev: opaque("struct amdgpu_device *"), bit: string, val: u32) .

            cp_resume_rreg(adev, reg, val, acc_flags) :-
              read_reg(adev, reg, val, acc_flags),
              bitand(en, val, 1),
              update_gfx_state(adev, "HQD_ACTIVE", en) ,
              cut() ."#,
            "foo.dl",
        );
        let r = parser::program(prog_str).unwrap();
        assert!(r.0.is_empty());
        let writer = RefCell::new(Printer { buf: vec![] });
        let mut codegen = LKMCodeGen::new(&writer, r.1);
        let r = codegen.compile();
        let buf = mem::replace(&mut writer.borrow_mut().buf, vec![]);
        let dump = std::str::from_utf8(buf.as_slice()).unwrap();
        if r.is_err() {
            println!("Error {:?}", r.as_ref().err().unwrap());
        }
        assert!(r.is_ok());
    }
}
