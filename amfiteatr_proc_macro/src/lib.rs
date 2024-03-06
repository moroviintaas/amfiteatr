use quote::quote;

use syn::{ItemStruct, parse_macro_input};

#[proc_macro_attribute]
pub fn no_assessment_info_set(attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream{

    let item_copy = item.clone();

    let input: ItemStruct = parse_macro_input!(item_copy);

    let domain = parse_macro_input!(attr as syn::Path);
    let ident = input.ident;
    let generics = input.generics;
    let where_clause = generics.where_clause;
    let params = generics.params;


    let implementation = match params.is_empty(){
        true => quote!{
            impl amfiteatr_core::agent::EvaluatedInformationSet<#domain> for #ident{
                type RewardType = amfiteatr_core::domain::NoneReward;

                fn current_subjective_score(&self) -> Self::RewardType {
                    amfiteatr_core::domain::NoneReward{}
                }
                fn penalty_for_illegal(&self) -> Self::RewardType {
                    amfiteatr_core::domain::NoneReward{}
                }
            }
        },
        false => quote!{
            impl<#params> amfiteatr_core::agent::EvaluatedInformationSet<#domain> for #ident <#params>
            #where_clause{
                type RewardType = amfiteatr_core::domain::NoneReward;

                fn current_subjective_score(&self) -> Self::RewardType {
                    amfiteatr_core::domain::NoneReward{}
                }
                fn penalty_for_illegal(&self) -> Self::RewardType {
                    amfiteatr_core::domain::NoneReward{}
                }
            }
        }
    };
    let result = [item, implementation.into()];
    proc_macro::TokenStream::from_iter(result)





}

