mod token_parsed;
//mod str_parsed;
mod token_variant;
mod tensorboard_support;
mod mcp;

use quote::quote;

use syn::{DeriveInput, ItemStruct, parse_macro_input};
//use crate::str_parsed::{derive_code_str_parsed};
use crate::token_parsed::{derive_code_token_parsed};
use crate::token_variant::derive_code_token_variant;

#[proc_macro_attribute]
pub fn no_assessment_info_set(attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream{

    let item_copy = item.clone();

    let input: ItemStruct = parse_macro_input!(item_copy);

    let scheme = parse_macro_input!(attr as syn::Path);
    let ident = input.ident;
    let generics = input.generics;
    let where_clause = generics.where_clause;
    let params = generics.params;


    let implementation = match params.is_empty(){
        true => quote!{
            impl amfiteatr_core::agent::EvaluatedInformationSet<#scheme,  amfiteatr_core::scheme::NoneReward> for #ident{

                fn current_assessment(&self) -> amfiteatr_core::scheme::NoneReward{
                    amfiteatr_core::scheme::NoneReward{}
                }
                fn penalty_for_illegal(&self) -> amfiteatr_core::scheme::NoneReward {
                    amfiteatr_core::scheme::NoneReward{}
                }
            }
        },
        false => quote!{
            impl<#params> amfiteatr_core::agent::EvaluatedInformationSet<#scheme, amfiteatr_core::scheme::NoneReward> for #ident <#params>
            #where_clause{

                fn current_assessment(&self) -> amfiteatr_core::scheme::NoneReward {
                    amfiteatr_core::scheme::NoneReward{}
                }
                fn penalty_for_illegal(&self) -> amfiteatr_core::scheme::NoneReward {
                    amfiteatr_core::scheme::NoneReward{}
                }
            }
        }
    };
    let result = [item, implementation.into()];
    proc_macro::TokenStream::from_iter(result)





}

/*
#[proc_macro_derive(StrParsed, attributes(keywords))]
pub fn derive_str_parsed(item: proc_macro::TokenStream) -> proc_macro::TokenStream{
    let input = parse_macro_input!(item as DeriveInput);
    derive_code_str_parsed(input)
}
*/



#[proc_macro_derive(TokenParsed, attributes(token, token_type))]
pub fn derive_token_parsed(item: proc_macro::TokenStream) -> proc_macro::TokenStream{
    let input = parse_macro_input!(item as DeriveInput);
    derive_code_token_parsed(input)
}

#[proc_macro_derive(TokenVariant, attributes(primitive))]
pub fn derive_token_variant(item: proc_macro::TokenStream) -> proc_macro::TokenStream{
    let input = parse_macro_input!(item as DeriveInput);
    derive_code_token_variant(input)
}

/*
#[proc_macro_derive(TensorboardSupport, attributes(tboard_writer))]
pub fn derive_tensorboard_support(item: proc_macro::TokenStream) -> proc_macro::TokenStream{
    let input = parse_macro_input!(item as DeriveInput);
    derive_code_token_variant(input)
}

 */



#[proc_macro_attribute]
pub fn mcp_env_state(attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream{


    let item_copy = item.clone();
    let ast = parse_macro_input!(item_copy as DeriveInput);
    let scheme = parse_macro_input!(attr as syn::Ident);

    let implementation = {

        let ident = ast.ident.clone();

        let mcp_struct_name = syn::Ident::new(&format!("Mcp{}", ident), proc_macro2::Span::call_site());

        quote!{
            #[derive(Clone)]
            pub struct #mcp_struct_name {
                core: amfiteatr_core::env::McpCoreSequentialEnvironment<#scheme, #ident, ()>,
            }
        }
    };

    let result = [item, implementation.into()];
    proc_macro::TokenStream::from_iter(result)

}
