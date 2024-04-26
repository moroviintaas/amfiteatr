
use proc_macro2::TokenStream;
use quote::{format_ident, quote, ToTokens};
use syn::{Data, DeriveInput, Fields, GenericParam, LitStr, Meta, parse_macro_input, Token};
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::token::{Comma, Slash};

fn clone_generic_param_ident(param: &GenericParam) -> proc_macro2::TokenStream{
    match param{
        GenericParam::Lifetime(l) => l.lifetime.clone().into_token_stream(),
        GenericParam::Type(t) => t.ident.clone().into_token_stream(),
        GenericParam::Const(c) => c.ident.clone().into_token_stream()
    }
}

fn make_variant_parser(tags: &Vec<LitStr>) -> proc_macro2::TokenStream{

    let tag_streams: Vec<TokenStream> = tags.iter().map(|tag|{
        quote! {nom::bytes::complete::tag(#tag),}

    }).collect();

    let mut stream = TokenStream::new();
    stream.extend(tag_streams);
    quote!{nom::branch::alt((#stream))}
    //::<&str, Self, nom::IResult<&'a str, Self>>
    
}
/*
fn parse_variant_fields(fields: &Fields) -> proc_macro2::TokenStream{
    match fields{
        Fields::Named(_) => {
            todo!()
        }
        Fields::Unnamed(f) => {
            todo!()
        }
        Fields::Unit => {
            match
        }
    }
}

 */

fn build_parse_variant_stream(tags: &Vec<LitStr>, var_ident: &syn::Ident, fields: &Fields) -> proc_macro2::TokenStream{

    let stream = make_variant_parser(tags);
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
                        let #this_member_tmp_name = <#ty as amfiteatr_core::util::NomParsed<&str>>::nom_parse(rest)?.1;
                    });
                    member_names.push(quote! {#this_member_tmp_name ,});
                    //panic!("This member tmp name: {}", this_member_tmp_name)
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

/*
if let Ok((rest, var)) = #stream (input){
                    return Ok((rest, Self::#var_ident))
                }
 */

pub(crate) fn nom_parsed(input: DeriveInput) -> proc_macro::TokenStream{


    let generics = input.generics.clone();
    let generics_params = input.generics.params;
    let generic_params_ident_vec: Vec<proc_macro2::TokenStream> = generics_params.iter().map(|g|{
        let mut ident = clone_generic_param_ident(g);

        ident.extend([quote! {,}]);
        ident
    }).collect();
    let mut generic_params_idents = proc_macro2::TokenStream::new();
    generic_params_idents.extend(generic_params_ident_vec);

    let where_clause = input.generics.where_clause;
    let ident = input.ident;
    let data = input.data;

    let mut variant_codes = Vec::new();
    match data{
        Data::Struct(_) => {
            //proc_macro::TokenStream::new()
        }
        Data::Enum(en) => {

            for v in en.variants{
                let var_ident = v.ident;
                let attributes = v.attrs;
                let fields = v.fields;
                //let found_attribute = None;
                for a in  attributes{
                    if let Meta::List(ml) = a.meta{
                        if ml.path.is_ident("keywords"){
                            let keywords = ml.tokens.clone();
                            //let stream = parse_macro_input!(keywords);
                            //let punc: Punctuated<LitStr, Token![,]> = Punctuated::parse_terminated(keywords).unwrap();
                            let parser = Punctuated::<LitStr, Token![,]>::parse_terminated;
                            let punc: Punctuated<LitStr, Token![,]> = parser.parse2(keywords)
                                .expect(format!("Keyword attributes were expected to be \"#[keywords(\"tag1\", \"tag2\", ...)]\", meanwhile provided: #[keywords({})]", &ml.tokens).as_str());
                            let tags: Vec<LitStr> = punc.iter().cloned().collect();
                            let variant_stream_parser = build_parse_variant_stream(&tags, &var_ident, &fields);
                            variant_codes.push(variant_stream_parser);
                            break;
                            //panic!("{}", variant_stream_parser)
                            //panic!("tags: {:?}", tags);


                        }
                    }
                }
            }
            //proc_macro::TokenStream::new()
        }
        Data::Union(_) => {
            //proc_macro::TokenStream::new()
        }
    }


    /*
    let implementation = quote! {
        impl <'a, #generics_params> amfiteatr_core::agent::NomParsed<&'a str> for #ident #generics
        #where_clause{
            todo!()
        }
    };

     */
    //panic!("generics: {}", generics_params.into_token_stream());
    let parse_variants = variant_codes.into_iter().fold(TokenStream::new(), |mut acc, n|{
        acc.extend([n]);
        acc
    });
    let implementation = quote! {
        impl <'a, #generics_params> amfiteatr_core::util::NomParsed<&'a str> for #ident <#generic_params_idents>
        #where_clause{
            fn nom_parse(input: &str) -> nom::IResult<&str, Self>{
                #parse_variants

                return Result::Err(nom::Err::Failure(nom::error::Error::new(input, nom::error::ErrorKind::Complete)))
            }
        }
    };


    implementation.into()



    //proc_macro::TokenStream::new()

}

