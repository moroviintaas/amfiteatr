
use proc_macro2::TokenStream;
use quote::{format_ident, quote, ToTokens};
use syn::{Data, DeriveInput, Fields, GenericParam, LitStr, Meta,  Token};
use syn::parse::Parser;
use syn::punctuated::Punctuated;

fn clone_generic_param_ident(param: &GenericParam) -> proc_macro2::TokenStream{
    match param{
        GenericParam::Lifetime(l) => l.lifetime.clone().into_token_stream(),
        GenericParam::Type(t) => t.ident.clone().into_token_stream(),
        GenericParam::Const(c) => c.ident.clone().into_token_stream()
    }
}

fn make_variant_parser_str(tags: &Vec<LitStr>) -> proc_macro2::TokenStream{

    let tag_streams: Vec<TokenStream> = tags.iter().map(|tag|{
        quote! { nom::sequence::terminated(nom::bytes::complete::tag(#tag), nom::character::complete::space1),}

    }).collect();

    let mut stream = TokenStream::new();
    stream.extend(tag_streams);
    quote!{nom::branch::alt((#stream))}
    //::<&str, Self, nom::IResult<&'input_lifetime str, Self>>

}


fn build_parse_variant_stream_str(tags: &Vec<LitStr>, var_ident: &syn::Ident, fields: &Fields) -> proc_macro2::TokenStream{

    let stream = make_variant_parser_str(tags);
    match fields{
        Fields::Named(_) => {
            todo!()
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
                        let (rest, #this_member_tmp_name) = <#ty as amfiteatr_core::util::StrParsed>::parse_from_str(rest)?;
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
                let r: nom::IResult<&str, &str> = #stream (input);
                if let Ok((rest, _var)) = r{
                    #def_stream
                    return Ok((rest, Self::#var_ident(#member_names)))
                }
            }



        }
        Fields::Unit => {
            quote!{
                let r: nom::IResult<&str, &str> = #stream (input);
                if let Ok((rest, _var)) = r{
                    return Ok((rest, Self::#var_ident))
                }
            }
        }
    }


}



pub(crate) fn code_for_parse_input_data_from_slice_str(data: &Data) -> TokenStream{
    match data{
        Data::Struct(_) => {
            todo!()
        }
        Data::Enum(enumeration) => {
            let mut variant_codes = Vec::new();
            for v in enumeration.variants.iter(){
                let var_ident = &v.ident;
                let attributes = &v.attrs;
                let fields = &v.fields;
                //let found_attribute = None;
                for a in  attributes{
                    if let Meta::List(ml) = &a.meta{
                        if ml.path.is_ident("keywords"){
                            let keywords = ml.tokens.clone();
                            let parser = Punctuated::<LitStr, Token![,]>::parse_terminated;
                            let punc: Punctuated<LitStr, Token![,]> = parser.parse2(keywords)
                                .expect(format!("Keyword attributes were expected to be \"#[keywords(\"tag1\", \"tag2\", ...)]\", meanwhile provided: #[keywords({})]", &ml.tokens).as_str());
                            let tags: Vec<LitStr> = punc.iter().cloned().collect();
                            let variant_stream_parser = build_parse_variant_stream_str(&tags, &var_ident, &fields);
                            variant_codes.push(variant_stream_parser);
                            break;


                        }
                    }
                }
            }
            variant_codes.iter().cloned().collect()
        }
        Data::Union(_) => {
            todo!()
        }
    }
}

pub(crate) fn derive_code_str_parsed(input: DeriveInput) -> proc_macro::TokenStream{


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







    let parsing_code = code_for_parse_input_data_from_slice_str(&data);



    let implementation_str = quote! {
        impl <#generics_params> amfiteatr_core::util::StrParsed for #ident <#generic_params_idents>
        #where_clause{
            fn parse_from_str(input: &str) -> nom::IResult<&str, Self>{
                #parsing_code

                return Result::Err(nom::Err::Failure(nom::error::Error::new(input, nom::error::ErrorKind::Complete)))
            }
        }
    };



    let mut ts: TokenStream = TokenStream::new();
    ts.extend([implementation_str]);



    ts.into()


}

