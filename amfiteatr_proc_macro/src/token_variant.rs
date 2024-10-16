use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    DeriveInput, Fields, Meta
};
use syn::Data::Enum;

pub(crate) fn derive_code_token_variant(input: DeriveInput) -> proc_macro::TokenStream{
    let data = input.data;
    let ident = input.ident;

    if let Enum(enum_data) = data{

        let mut impl_streams = Vec::new();

        for v in enum_data.variants{
            let v_name = v.ident;

            let attrs = v.attrs;
            for a in attrs{
                if let Meta::Path(ml) = a.meta{
                    if ml.is_ident("primitive"){
                        let fields = v.fields.clone();

                        if let Fields::Unnamed(fs) = fields{
                            //panic!("Unnamed {ident:?}");
                            let field = fs.unnamed.first().expect("Primitive variant declared with no type");

                            let tp = &field.ty;
                            let header_1 = quote! {
                                //impl<'a> amfiteatr_core::util::TokenParsed<amfiteatr_core::util::TokensBorrowed<'a, #ident>> for #for_what
                                impl<'a> amfiteatr_core::util::PrimitiveMarker<#tp> for #ident
                            };

                            let body = quote! {{
                                fn primitive(&self) -> Option<#tp>{
                                    if let Self:: #v_name(pt) = self{
                                        Some(pt.clone())
                                    } else {
                                        None
                                    }
                                }
                                /*

                            fn parse_from_tokens(input: amfiteatr_core::util::TokensBorrowed<'a, #ident>)
                            -> amfiteatr_core::reexport::nom::IResult<amfiteatr_core::util::TokensBorrowed<'a, #ident>, v_ident>{

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

                             */


                        }};
                            let stream = vec![header_1, body];
                            let mut ts = TokenStream::new();
                            ts.extend(stream);
                            impl_streams.push(ts);

                        }

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