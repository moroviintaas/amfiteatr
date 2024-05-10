use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Meta, parse_macro_input};
use syn::Data::Enum;

pub(crate) fn derive_code_token_variant(input: DeriveInput) -> proc_macro::TokenStream{
    let data = input.data;
    let ident = input.ident;

    if let Enum(enum_data) = data{

        let mut impl_streams = Vec::new();

        for v in enum_data.variants{
            let v_name = v.ident;
            for a in v.attrs{
                if let Meta::List(ml) = a.meta{
                    if ml.path.is_ident("primitive"){
                        let tokens = ml.tokens;
                        //let tokens = ml.tokens.into();
                        //let tp = parse_macro_input!(tokens as syn::Path);

                        let header_1 = quote! {
                            impl<'a> amfiteatr_core::reexport::nom::Parser<amfiteatr_core::util::TokensBorrowed<'a, #ident>, #ident, amfiteatr_core::reexport::nom::error::Error<amfiteatr_core::util::TokensBorrowed<'a, #ident>>> for
                        };
                        let for_what = tokens.clone();
                        let body = quote! {{
                            fn parse(&mut self, input: amfiteatr_core::util::TokensBorrowed<'a, #ident>)
                                -> amfiteatr_core::reexport::nom::IResult<amfiteatr_core::util::TokensBorrowed<'a, #ident>, #tokens, amfiteatr_core::reexport::nom::error::Error<amfiteatr_core::util::TokensBorrowed<'a, #ident>>>{

                                if input.is_empty(){
                                    return Err(amfiteatr_core::reexport::nom::Err::Failure(amfiteatr_core::reexport::nom::error::Error{input, code: amfiteatr_core::reexport::nom::error::ErrorKind::Eof}))
                                }
                                let token = input[0];
                                if let #ident :: #v_name(t) = token{
                                    let rest = input[1..];
                                    Ok((rest, t))
                                } else {
                                    Err(amfiteatr_core::reexport::nom::Err::Error(amfiteatr_core::reexport::nom::error::Error{input, code: amfiteatr_core::reexport::nom::error::ErrorKind::Tag}))
                                }
                            }
                        }};
                        let mut stream = vec![header_1, for_what, body];
                        let mut ts = TokenStream::new();
                        ts.extend(stream);
                        impl_streams.push(ts);
                    }
                }
            }
        }
        let mut ts: TokenStream = TokenStream::new();
        ts.extend(impl_streams);

        ts.into()

    } else {
        panic!("Token Variant derive implementation is not supported for unions and structs")
    }


}