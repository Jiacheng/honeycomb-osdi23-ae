use crate::codegen::ast::Program;
use crate::error::Span;
use nom::bytes::complete::tag;
use nom::error::{ErrorKind, FromExternalError, ParseError};
use nom::Err;
use nom_locate::{position, LocatedSpan};
use std::str::from_utf8;

#[derive(Debug, Clone, PartialEq)]
pub enum SemanticFailure {
    InvalidValue,
    ConflictingDeclarationModifier,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CustomErrorKind<'a> {
    NomErrorKind(ErrorKind),
    Failure(Span<'a>, SemanticFailure),
}

impl<'a> CustomErrorKind<'a> {
    pub fn failure(pos: Span<'a>, f: SemanticFailure) -> CustomErrorKind<'a> {
        CustomErrorKind::Failure(pos, f)
    }
}

pub type IResult<'a, O> = Result<(Span<'a>, O), Err<CustomErrorKind<'a>>>;

impl<'a> ParseError<LocatedSpan<&'a str, &'a str>> for CustomErrorKind<'a> {
    fn from_error_kind(_: LocatedSpan<&'a str, &'a str>, kind: ErrorKind) -> Self {
        CustomErrorKind::NomErrorKind(kind)
    }

    fn append(_: LocatedSpan<&str, &str>, _: ErrorKind, other: Self) -> Self {
        other
    }
}

impl<'a> FromExternalError<Span<'a>, CustomErrorKind<'a>> for CustomErrorKind<'a> {
    fn from_external_error(_: Span<'a>, _: ErrorKind, e: CustomErrorKind<'a>) -> Self {
        e
    }
}

mod file_parser {
    use crate::codegen::ast::*;
    use crate::codegen::parser::*;
    use nom::branch::alt;
    use nom::bytes::complete::{escaped, is_not, take_while, take_while1};
    use nom::character::complete::{multispace0, multispace1, none_of};
    use nom::combinator::{map, map_res, opt, recognize};
    use nom::multi::{many0, many1, separated_list0, separated_list1};
    use nom::sequence::{delimited, pair, preceded, separated_pair, terminated, tuple};
    use nom::{AsChar, Parser};

    fn ws<'a, F: 'a, O>(inner: F) -> impl FnMut(Span<'a>) -> IResult<'a, O>
    where
        F: Fn(Span<'a>) -> IResult<'a, O>,
    {
        delimited(multispace0, inner, multispace0)
    }

    fn ws_or_comment(input: Span) -> IResult<()> {
        let comment_line = terminated(tag("//"), opt(is_not("\n\r")));
        let (input, _) = alt((multispace1, comment_line))(input)?;
        Ok((input, ()))
    }

    fn constant_string(input: Span) -> IResult<Span> {
        let esc = escaped(none_of("\\\""), '\\', tag("\""));
        let esc_or_empty = alt((esc, tag("")));
        let res = delimited(tag("\""), esc_or_empty, tag("\""))(input)?;
        Ok(res)
    }

    fn constant_integer(input: Span) -> IResult<u64> {
        let (input, pos) = position(input)?;
        let hex_digits = map_res(
            preceded(tag("0x"), take_while1(|c: char| c.is_hex_digit())),
            |x: Span| {
                u64::from_str_radix(from_utf8(x.as_bytes()).unwrap_or(""), 16)
                    .map_err(|_| CustomErrorKind::failure(pos, SemanticFailure::InvalidValue))
            },
        );
        let decimal_digits = map_res(take_while1(|c: char| c.is_digit(10)), |x: Span| {
            u64::from_str_radix(from_utf8(x.as_bytes()).unwrap_or(""), 10)
                .map_err(|_| CustomErrorKind::failure(pos, SemanticFailure::InvalidValue))
        });
        let (input, v) = alt((hex_digits, decimal_digits))(input)?;
        Ok((input, v))
    }

    fn term(input: Span) -> IResult<Term> {
        let (input, term) = alt((
            map(identifier, |x| Term::IDENTIFIER(x)),
            map(constant_string, |x| Term::STRING(x.to_string())),
            map(constant_integer, |x| Term::INTEGER(x)),
        ))(input)?;
        Ok((input, term))
    }

    fn atom(input: Span) -> IResult<Atom> {
        let (input, pos) = position(input)?;
        let (input, (id, terms)) = tuple((
            identifier,
            delimited(
                ws(tag("(")),
                separated_list0(ws(tag(",")), term),
                ws(tag(")")),
            ),
        ))(input)?;
        Ok((
            input,
            Atom {
                pos: pos.into(),
                relation: id,
                terms,
            },
        ))
    }

    fn rule(input: Span) -> IResult<Rule> {
        let (input, pos) = position(input)?;
        let (input, (head, _, predicates, _)) = tuple((
            atom,
            ws(tag(":-")),
            separated_list1(ws(tag(",")), atom),
            tag("."),
        ))(input)?;
        Ok((
            input,
            Rule {
                pos: pos.into(),
                head,
                predicates,
            },
        ))
    }

    fn parse_type(input: Span) -> IResult<Type> {
        let opaque = preceded(
            tag("opaque"),
            delimited(tag("("), constant_string, tag(")")),
        );
        let mut types = alt((
            opaque.map(|x| Type::OPAQUE(x.to_string())),
            tag("bool").map(|_| Type::BOOL),
            tag("u32").map(|_| Type::U32),
            tag("u64").map(|_| Type::U64),
            tag("string").map(|_| Type::STRING),
            tag("void").map(|_| Type::VOID),
        ));
        types(input)
    }

    fn identifier<'a>(input: Span<'a>) -> IResult<'a, Identifier> {
        let (input, pos) = position(input)?;
        let (input, name) = recognize(pair(
            take_while1(|c: char| c.is_alphabetic() || c == '_'),
            take_while(|c: char| c.is_alphanumeric() || c == '_'),
        ))(input)?;

        Ok((
            input,
            Identifier {
                pos,
                id: name.to_string(),
            },
        ))
    }

    fn opt_decl_modifiers(input: Span) -> IResult<RelationFlags> {
        let (input, pos) = position(input)?;
        let decl_modifier = map_res(
            alt((tag("output"), tag("extern"), tag("relation"), tag("action"))),
            |x: Span| match from_utf8(x.as_bytes()).unwrap_or("") {
                "output" => Ok(RelationFlags::GLOBAL),
                "extern" => Ok(RelationFlags::EXTERNAL),
                "relation" => Ok(RelationFlags::PREDICATE),
                "action" => Ok(RelationFlags::ACTION),
                _ => Err(CustomErrorKind::Failure(pos, SemanticFailure::InvalidValue)),
            },
        );

        let modifier_lists = separated_list0(multispace1, decl_modifier);
        let (input, decl_lists) = opt(modifier_lists)(input)?;
        if decl_lists.is_none() {
            return Ok((input, RelationFlags::empty()));
        }

        let decl_lists = decl_lists.unwrap();

        let decl_flags = decl_lists
            .iter()
            .fold(RelationFlags::empty(), |flag, item| flag.union(*item));

        Ok((input, decl_flags))
    }

    pub fn decl_annotation(input: Span) -> IResult<Vec<DeclAnnotation>> {
        let (input, pos) = position(input)?;
        let annotation = map_res(
            separated_pair(identifier, ws(tag("=")), constant_string),
            |(k, v)| match k.id.as_str() {
                "c_name" => Ok(DeclAnnotation::CFuncName(v.to_string())),
                _ => Err(CustomErrorKind::failure(pos, SemanticFailure::InvalidValue)),
            },
        );
        let full = delimited(
            tag("#["),
            separated_list1(ws(tag(",")), annotation),
            tag("]"),
        );
        let (input, res) = opt(full)(input)?;
        let r = match res {
            None => vec![],
            Some(x) => x,
        };
        Ok((input, r))

        // let mut annotation = separated_pair(identifier, tag("="), constant_string);
        // let (input, r) = annotation(input)?;
        // println!("hhh {:?}", r.0);
        // Ok((input, vec![]))
    }

    fn decl_arg(input: Span) -> IResult<Argument> {
        let (input, pos) = position(input)?;
        let arg_modifier = map_res(tag("out"), |x: Span| {
            match from_utf8(x.as_bytes()).unwrap_or("") {
                "out" => Ok(ArgumentFlag::OUT),
                _ => Err(CustomErrorKind::failure(pos, SemanticFailure::InvalidValue)),
            }
        });
        let arg_modifiers = delimited(
            ws(tag("[")),
            separated_list1(ws(tag(",")), arg_modifier),
            ws(tag("]")),
        );
        let mut entry = separated_pair(
            identifier,
            ws(tag(":")),
            pair(opt(arg_modifiers), parse_type),
        );
        let (input, r) = entry(input)?;
        let arg_type = r.1;
        let flag = match arg_type.0.as_ref() {
            None => ArgumentFlag::empty(),
            Some(x) => x.iter().fold(ArgumentFlag::empty(), |r, f| r.union(*f)),
        };
        Ok((
            input,
            Argument {
                pos: pos.into(),
                name: r.0,
                flag,
                ty: arg_type.1,
            },
        ))
    }

    fn parse_decl(input: Span) -> IResult<Declaration> {
        let (input, pos) = position(input)?;
        let (input, annotations) = ws(decl_annotation)(input)?;
        let (input, decl_flags) = ws(opt_decl_modifiers)(input)?;
        let (input, name) = ws(identifier)(input)?;
        let (input, params) = delimited(
            ws(tag("(")),
            separated_list0(ws(tag(",")), decl_arg),
            ws(tag(")")),
        )(input)?;
        let (input, _) = ws(tag("."))(input)?;

        // Defer the checks as the parsing will backtrack
        if (decl_flags.contains(RelationFlags::PREDICATE)
            && decl_flags.contains(RelationFlags::ACTION))
            || (!decl_flags.contains(RelationFlags::PREDICATE)
                && !decl_flags.contains(RelationFlags::ACTION))
        {
            return Err(nom::Err::Failure(CustomErrorKind::Failure(
                pos,
                SemanticFailure::ConflictingDeclarationModifier,
            )));
        } else if decl_flags.contains(RelationFlags::GLOBAL)
            && decl_flags.contains(RelationFlags::EXTERNAL)
        {
            return Err(nom::Err::Failure(CustomErrorKind::Failure(
                pos,
                SemanticFailure::ConflictingDeclarationModifier,
            )));
        }

        Ok((
            input,
            Declaration {
                pos: pos.into(),
                name,
                flags: decl_flags,
                params,
                annotations,
            },
        ))
    }

    pub fn program(input: Span) -> IResult<Program> {
        #[derive(Debug, Clone)]
        enum Statement<'a> {
            Rule(Rule<'a>),
            Declaration(Declaration<'a>),
        }

        let stmt = alt((
            map(parse_decl, |x| Statement::Declaration(x)),
            map(rule, |x| Statement::Rule(x)),
        ));
        let (input, res) = terminated(
            many1(preceded(many0(ws_or_comment), stmt)),
            many0(ws_or_comment),
        )(input)?;

        let mut decl = vec![];
        let mut rules = vec![];
        for x in res {
            match x {
                Statement::Rule(x) => rules.push(x),
                Statement::Declaration(x) => decl.push(x),
            }
        }
        Ok((
            input,
            Program {
                declarations: decl,
                rules,
            },
        ))
    }
}

pub fn program(input: Span) -> IResult<Program> {
    file_parser::program(input)
}

#[cfg(test)]
mod tests {
    // use crate::codegen::parser_v2::file_parser::*;
    use crate::codegen::parser::*;

    #[test]
    fn test_parse_rule_v2() {
        let input = Span::new_extra(
            r#"
output action cp_resume_rreg(adev: opaque("struct amdgpu_device *"), reg: u32, val: [out] u32, acc_flags: u32) .
#[c_name="SOC15_REG"]
extern relation soc15_reg(reg: u32, ip: string, inst_id: u32, reg_name: string) .
        "#,
            "foo.dl",
        );
        let output = program(input).unwrap();
        assert_eq!(output.0.len(), 0);
        assert_eq!(output.1.declarations.len(), 2);
    }
}
