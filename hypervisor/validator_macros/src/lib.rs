mod helpers;
mod macros;

use proc_macro2::TokenStream;
use std::env;
use syn::DeriveInput;

fn debug_print_generated(ast: &DeriveInput, toks: &TokenStream) {
    let debug = env::var("MACRO_DEBUG");
    if let Ok(s) = debug {
        if s == "1" {
            println!("{}", toks);
        }

        if ast.ident == s {
            println!("{}", toks);
        }
    }
}

#[proc_macro_derive(Opcode, attributes(opcode))]
pub fn opcode(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast = syn::parse_macro_input!(input as DeriveInput);

    let toks = macros::display_inner(&ast).unwrap_or_else(|err| err.to_compile_error());
    debug_print_generated(&ast, &toks);
    toks.into()
}
