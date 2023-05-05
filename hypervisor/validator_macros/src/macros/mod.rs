use crate::helpers::non_enum_error;
use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields};

pub fn display_inner(ast: &DeriveInput) -> syn::Result<TokenStream> {
    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();
    let variants = match &ast.data {
        Data::Enum(v) => &v.variants,
        _ => return Err(non_enum_error()),
    };

    let mut display_arms = Vec::new();
    let mut convert_arms = Vec::new();
    for variant in variants {
        let ident = &variant.ident;
        let output = ident.to_string().to_lowercase();

        let params = match variant.fields {
            Fields::Unit => quote! {},
            Fields::Unnamed(..) => quote! { (..) },
            Fields::Named(..) => quote! { {..} },
        };

        display_arms.push(quote! { #name::#ident #params => f.pad(#output) });

        if let Some(x) = &variant.discriminant {
            let value = &x.1;
            convert_arms.push(quote! { #value => Ok(#name::#ident) });
        }
    }

    if display_arms.len() < variants.len() {
        display_arms.push(quote! { _ => panic!("fmt() called on disabled variant.") });
    }
    convert_arms.push(quote! { _ => Err(std::io::Error::from(std::io::ErrorKind::InvalidData)) });

    Ok(quote! {
        impl #impl_generics ::core::fmt::Display for #name #ty_generics #where_clause {
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::result::Result<(), ::core::fmt::Error> {
                match *self {
                    #(#display_arms),*
                }
            }
        }

        impl ::core::convert::TryFrom<u32> for #name {
            type Error = std::io::Error;
            fn try_from(value: u32) -> Result<Self, Self::Error> {
                match value {
                    #(#convert_arms),*
                }
            }
        }
    })
}
