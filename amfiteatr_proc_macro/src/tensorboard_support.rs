/*
use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    DeriveInput, Fields, Meta
};
use syn::Data::Struct;


pub(crate) fn derive_code_tensorboard_support(input: DeriveInput) -> proc_macro::TokenStream{
    let data = input.data;
    let ident = input.ident;

    if let Struct(struct_data) = data{
        //let mut impl_streams = Vec::new();

        let mut writer_field_name = None;

        if let Fields::Named(named_fields) = struct_data.fields{
            for field in named_fields.named{
                for attribute in field.attrs{
                    if let Meta::Path(attrib) = attribute.meta{

                    }

                }
            }

        }

    }


}

 */

