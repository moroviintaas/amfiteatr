
use proc_macro2::TokenStream;
use quote::{format_ident, quote, ToTokens};
use syn::{
    Data, DeriveInput, Fields, GenericParam, Meta
};

fn clone_generic_param_ident(param: &GenericParam) -> proc_macro2::TokenStream{
    match param{
        GenericParam::Lifetime(l) => l.lifetime.clone().into_token_stream(),
        GenericParam::Type(t) => t.ident.clone().into_token_stream(),
        GenericParam::Const(c) => c.ident.clone().into_token_stream()
    }
}
/*
fn make_variant_parser(token: &TokenStream) -> proc_macro2::TokenStream{

    let tag_streams: Vec<TokenStream> = .iter().map(|tag|{
        quote! { nom::sequence::terminated(nom::bytes::complete::tag(#tag), nom::character::complete::space1),}

    }).collect();

    let mut stream = TokenStream::new();
    stream.extend(tag_streams);
    quote!{nom::branch::alt((#stream))}
    //::<&str, Self, nom::IResult<&'input_lifetime str, Self>>
    
}


 */

fn build_parse_variant_stream(token_type: &TokenStream, token_type_variant: &TokenStream, parsed_variant_ident: &syn::Ident, fields: &Fields) -> proc_macro2::TokenStream{

    //let stream = make_variant_parser(token);
    match fields{
        Fields::Named(fields_named) => {
            let ref fields = fields_named.named;

            let mut member_idents = Vec::new();
            let mut codes = Vec::new();

            for field in fields.iter(){


                let ref ty  = field.ty;
                let field_name = &field.ident.clone().unwrap();

                codes.push(quote! {
                   let (rest, #field_name) =  <#ty as amfiteatr_core::util::TokenParsed<amfiteatr_core::util::TokensBorrowed<'input_lifetime, #token_type>>>::parse_from_tokens(rest)?;
                });
                member_idents.push(quote!{#field_name ,});



            }

            let code_stream: TokenStream = codes.into_iter().collect();
            let member_names: TokenStream = member_idents.into_iter().collect();

            quote!{

                if input[0] == #token_type :: #token_type_variant{
                    let rest = amfiteatr_core::util::TokensBorrowed(&input.0[1..]);
                    #code_stream
                    return Ok((rest, Self::#parsed_variant_ident{#member_names}))

                }
            }

        }
        Fields::Unnamed(fields_unnamed) => {
            let ref fields = fields_unnamed.unnamed;

            let mut streams = Vec::new();
            let mut member_names = Vec::new();
            let mut i= 1u8;
            for field in fields.iter(){
                //if let Some(ref member_type_ident) = field.ident{
                let ref ty = field.ty;
                    let this_member_tmp_name = format_ident!("result_member_{}",i);

                    streams.push(quote! {
                        let (rest, #this_member_tmp_name) = <#ty as amfiteatr_core::util::TokenParsed<amfiteatr_core::util::TokensBorrowed<'input_lifetime, #token_type>>>::parse_from_tokens(rest)?;
                    });
                    member_names.push(quote! {#this_member_tmp_name ,});
                //}
                i += 1;
            }

            let def_stream: TokenStream = streams.into_iter().collect();
            let member_names: TokenStream = member_names.into_iter().collect();
            //panic!("def_stream: \n {}", def_stream);
            //panic!("member_names: {}", member_names);

            quote! {
                /*
                let r: amfiteatr_core::reexport::nom::IResult<amfiteatr_core::util::TokensBorrowed<'input_lifetime, #token_type>, #parsed_variant_ident> = #parsed_variant_ident ::parse_from_tokens(input);
                if let Ok((rest, _var)) = r{
                    #def_stream
                    return Ok((rest, Self::#parsed_variant_ident(#member_names)))
                }

                 */
                if input[0] == #token_type :: #token_type_variant{
                    let rest = amfiteatr_core::util::TokensBorrowed(&input.0[1..]);
                    #def_stream
                    return Ok((rest, Self::#parsed_variant_ident(#member_names)))

                }
            }



        }
        Fields::Unit => {
            quote!{
                /*
                let r: nom::IResult<&str, &str> = #stream (input);
                let r: amfiteatr_core::reexport::nom::IResult<amfiteatr_core::util::TokensBorrowed<'input_lifetime, #token_type>, #parsed_variant_ident> = #parsed_variant_ident ::parse_from_tokens(input);
                if let Ok((rest, _var)) = r{
                    return Ok((rest, Self::#parsed_variant_ident))
                }

                 */
                if input[0] == #token_type :: #token_type_variant{
                    return Ok((amfiteatr_core::util::TokensBorrowed(&input.0[1..]), Self :: #parsed_variant_ident));
                }

            }
        }
    }


}



pub(crate) fn code_for_parse_input_data_from_slice(data: &Data,  token_type: &TokenStream) -> TokenStream{
    match data{

        Data::Struct(st) => match &st.fields{
            Fields::Named(_) => {todo!()}
            Fields::Unnamed(unnamed) => {
                let mut codes = Vec::new();
                let mut names = Vec::new();
                codes.push(quote!{
                    //let rest = &input[..];
                    let rest = amfiteatr_core::util::TokensBorrowed(&input.0[..]);
                });

                for (i, field) in unnamed.unnamed.iter().enumerate(){
                    let ref ty = field.ty;
                    let subname = format_ident!("_member_{}",i);
                    codes.push(quote! {
                        let (rest, #subname) = <#ty as amfiteatr_core::util::TokenParsed<amfiteatr_core::util::TokensBorrowed<'input_lifetime, #token_type>>>::parse_from_tokens(rest)?;

                    });
                    names.push(quote!{#subname, });
                }
                codes.push(quote! {
                    Ok((
                        rest,
                        Self(
                        #(#names)*
                        )
                    ))
                });

                codes.iter().cloned().collect()


            },
            Fields::Unit => {todo!()}
        }


        /*
        Data::Struct(st)  => {
            todo!()
        }

         */


        Data::Enum(enumeration) => {
            let mut variant_codes = Vec::new();
            variant_codes.push(quote!{
               if input.is_empty(){
                    return Err(amfiteatr_core::reexport::nom::Err::Failure(amfiteatr_core::reexport::nom::error::Error{input, code: amfiteatr_core::reexport::nom::error::ErrorKind::Eof}));
                }
            });
            for v in enumeration.variants.iter(){
                let parsed_variant_ident = &v.ident;
                let attributes = &v.attrs;
                let fields = &v.fields;
                //let found_attribute = None;
                for a in  attributes{
                    if let Meta::List(ml) = &a.meta{
                        if ml.path.is_ident("token"){
                            //let keywords = ml.tokens.clone();
                            let token_type_variant = &ml.tokens;
                            let variant_stream_parser = build_parse_variant_stream(token_type, token_type_variant, &parsed_variant_ident, &fields);

                            variant_codes.push(variant_stream_parser);
                            break;


                        }
                    }
                }
            }
            variant_codes.push(quote!{
                return Err(amfiteatr_core::reexport::nom::Err::Failure(amfiteatr_core::reexport::nom::error::Error{input, code: amfiteatr_core::reexport::nom::error::ErrorKind::Tag}));
            });
            variant_codes.iter().cloned().collect()
        }
        Data::Union(_) => {
            todo!()
        }
    }
}

pub(crate) fn derive_code_token_parsed(input: DeriveInput) -> proc_macro::TokenStream{
    //return proc_macro::TokenStream::new();

    //let generics = input.generics.clone();
    let generics_params = input.generics.params;
    let generic_params_ident_vec: Vec<proc_macro2::TokenStream> = generics_params.iter().map(|g|{
        let mut ident = clone_generic_param_ident(g);

        ident.extend([quote! {,}]);
        ident
    }).collect();

    //let mut types: Vec::new();
    //let should_parse_from_str

    let mut generic_params_idents = proc_macro2::TokenStream::new();
    generic_params_idents.extend(generic_params_ident_vec);

    let where_clause = input.generics.where_clause;
    let ident = input.ident;
    let data = input.data;

    let attrs = input.attrs;
    let mut token_type_option = None;
    for attribute in attrs{
        let meta = attribute.meta;
        if let Meta::List(attr) = meta{
            if attr.path.is_ident("token_type"){
                token_type_option = Some(attr.tokens)

            }
        }
    }

    if let Some(token_type) = token_type_option{
        let parsing_code = code_for_parse_input_data_from_slice(&data, &token_type);



        let implementation = quote! {
            impl <'input_lifetime, #generics_params> amfiteatr_core::util::TokenParsed< amfiteatr_core::util::TokensBorrowed<'input_lifetime, #token_type>> for #ident <#generic_params_idents>
            #where_clause{

                fn parse_from_tokens(input: amfiteatr_core::util::TokensBorrowed<'input_lifetime, #token_type>) -> amfiteatr_core::reexport::nom::IResult<amfiteatr_core::util::TokensBorrowed<'input_lifetime, #token_type>, Self>{
                    #parsing_code

                    //return Result::Err(amfiteatr_core::reexport::nom::Err::Failure(nom::error::Error::new(input, amfiteatr_core::reexport::nom::error::ErrorKind::Complete)))
                }
            }
        };



        let mut ts: TokenStream = TokenStream::new();
        ts.extend([implementation]);



        ts.into()

    } else {
        panic!( "#[token_type(..) is required")
    }










}

