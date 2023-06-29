from typing import List, Union
import re

class Tokenizer:

    _WORD = re.compile(r'^\w*\b')

    _LIBRARY = [' ', '\n', '-- A comment', '.', 'abort', 'else', 'new', 'return', 'elsif', 'not', 'reverse', 'abstract', 'end', 'null', 'accept', 'entry', 'select', 'access', 'exception', 'of', 'separate', 'aliased', 'exit', 'some', 'all', 'others', 'subtype', 'and', 'for', 'out', 'synchronized', 'array', 'function', 'overriding', 'at', 'tagged', 'generic', 'package', 'task', 'begin', 'goto', 'pragma', 'terminate', 'body', 'private', 'then', 'if', 'procedure', 'type', 'case', 'in', 'protected', 'constant', 'interface', 'until', 'is', 'raise', 'use', 'declare', 'range', 'delay', 'limited', 'record', 'when', 'delta', 'loop', 'rem', 'while', 'digits', 'renames', 'with', 'do', 'mod', 'requeue', 'xor', 'abs', 'or', '=>', '(', ')', "'", '>=', '<=', '/=', '>', '<', ':=', '=', '+', '-', '*', '/', '**', '&', ',', ';', ':', '[', ']']
    _LIBRARY_REGEX = {' ': re.compile(r'^ '), '\n': re.compile(r'^(\r)?\n'), '-- A comment': re.compile(r'^--.*'), '.': re.compile(r'^\.'), 'abort': re.compile(r'^\babort\b'), 'else': re.compile(r'^\belse\b'), 'new': re.compile(r'^\bnew\b'), 'return': re.compile(r'^\breturn\b'), 'elsif': re.compile(r'^\belsif\b'), 'not': re.compile(r'^\bnot\b'), 'reverse': re.compile(r'^\breverse\b'), 'abstract': re.compile(r'^\babstract\b'), 'end': re.compile(r'^\bend\b'), 'null': re.compile(r'^\bnull\b'), 'accept': re.compile(r'^\baccept\b'), 'entry': re.compile(r'^\bentry\b'), 'select': re.compile(r'^\bselect\b'), 'access': re.compile(r'^\baccess\b'), 'exception': re.compile(r'^\bexception\b'), 'of': re.compile(r'^\bof\b'), 'separate': re.compile(r'^\bseparate\b'), 'aliased': re.compile(r'^\baliased\b'), 'exit': re.compile(r'^\bexit\b'), 'some': re.compile(r'^\bsome\b'), 'all': re.compile(r'^\ball\b'), 'others': re.compile(r'^\bothers\b'), 'subtype': re.compile(r'^\bsubtype\b'), 'and': re.compile(r'^\band\b'), 'for': re.compile(r'^\bfor\b'), 'out': re.compile(r'^\bout\b'), 'synchronized': re.compile(r'^\bsynchronized\b'), 'array': re.compile(r'^\barray\b'), 'function': re.compile(r'^\bfunction\b'), 'overriding': re.compile(r'^\boverriding\b'), 'at': re.compile(r'^\bat\b'), 'tagged': re.compile(r'^\btagged\b'), 'generic': re.compile(r'^\bgeneric\b'), 'package': re.compile(r'^\bpackage\b'), 'task': re.compile(r'^\btask\b'), 'begin': re.compile(r'^\bbegin\b'), 'goto': re.compile(r'^\bgoto\b'), 'pragma': re.compile(r'^\bpragma\b'), 'terminate': re.compile(r'^\bterminate\b'), 'body': re.compile(r'^\bbody\b'), 'private': re.compile(r'^\bprivate\b'), 'then': re.compile(r'^\bthen\b'), 'if': re.compile(r'^\bif\b'), 'procedure': re.compile(r'^\bprocedure\b'), 'type': re.compile(r'^\btype\b'), 'case': re.compile(r'^\bcase\b'), 'in': re.compile(r'^\bin\b'), 'protected': re.compile(r'^\bprotected\b'), 'constant': re.compile(r'^\bconstant\b'), 'interface': re.compile(r'^\binterface\b'), 'until': re.compile(r'^\buntil\b'), 'is': re.compile(r'^\bis\b'), 'raise': re.compile(r'^\braise\b'), 'use': re.compile(r'^\buse\b'), 'declare': re.compile(r'^\bdeclare\b'), 'range': re.compile(r'^\brange\b'), 'delay': re.compile(r'^\bdelay\b'), 'limited': re.compile(r'^\blimited\b'), 'record': re.compile(r'^\brecord\b'), 'when': re.compile(r'^\bwhen\b'), 'delta': re.compile(r'^\bdelta\b'), 'loop': re.compile(r'^\bloop\b'), 'rem': re.compile(r'^\brem\b'), 'while': re.compile(r'^\bwhile\b'), 'digits': re.compile(r'^\bdigits\b'), 'renames': re.compile(r'^\brenames\b'), 'with': re.compile(r'^\bwith\b'), 'do': re.compile(r'^\bdo\b'), 'mod': re.compile(r'^\bmod\b'), 'requeue': re.compile(r'^\brequeue\b'), 'xor': re.compile(r'^\bxor\b'), 'abs': re.compile(r'^\babs\b'), 'or': re.compile(r'^\bor\b'), '=>': re.compile(r'^=>'), '(': re.compile(r'^\('), ')': re.compile(r'^\)'), "'": re.compile(r"^'"), '>=': re.compile(r'^>='), '<=': re.compile(r'^<='), '/=': re.compile(r'^/='), '>': re.compile(r'^>'), '<': re.compile(r'^<'), ':=': re.compile(r'^:='), '=': re.compile(r'^='), '+': re.compile(r'^\+'), '-': re.compile(r'^-'), '*': re.compile(r'^\*'), '/': re.compile(r'^/'), '**': re.compile(r'^\*\*'), '&': re.compile(r'^&'), ',': re.compile(r'^,'), ';': re.compile(r'^;'), ':': re.compile(r'^:'), '[': re.compile(r'^\['), ']': re.compile(r'^\]'),}

    _STRING_LIT = 'STRING_LIT'

    _LIBRARY = [_STRING_LIT] + _LIBRARY
    _LIBRARY_REGEX[_STRING_LIT] = re.compile(r'^"(""|[^"\n])*"')

    _TOKEN_TO_ID = {k: v + 1 for v, k in enumerate(_LIBRARY)}
    _ID_TO_TOKEN = {v: k for k, v in _TOKEN_TO_ID.items()}

    _PAD = 0
    _UKN1 = len(_LIBRARY) + 1
    _ID_TO_TOKEN[_PAD] = ''
    _ID_TO_TOKEN[_UKN1] = '#'


    _NEWLINE_TOKEN = _TOKEN_TO_ID['\n']
    @classmethod
    def newline_token(cls) -> int:
        return cls._NEWLINE_TOKEN

    _SPACE_TOKEN = _TOKEN_TO_ID[' ']
    @classmethod
    def space_token(cls) -> int:
        return cls._SPACE_TOKEN

    @classmethod
    def _gen_uknown(cls, unknown_count: int) -> List[int]:
        return [cls._UKN1] * unknown_count

    @classmethod
    def encode(cls, text: str) -> List[int]:
        token_ids = []
        unknown_count = 0
        while text:
            for token in cls._LIBRARY:
                if match := cls._LIBRARY_REGEX[token].match(text):
                    token_ids.extend(cls._gen_uknown(unknown_count))
                    unknown_count = 0
                    if token == cls._STRING_LIT:
                        match_length = len(match.group())
                        token_ids.extend([cls._TOKEN_TO_ID[token]] * match_length)
                        text = text[match_length:]
                    else:
                        text = cls._LIBRARY_REGEX[token].sub('', text, count=1)
                        token_ids.append(cls._TOKEN_TO_ID[token])
                    break
            else:
                word_length = 1
                if match := cls._WORD.match(text):
                    word_length = len(match.group())
                unknown_count += word_length
                text = text[word_length:]
        token_ids.extend(cls._gen_uknown(unknown_count))
        return token_ids

    @classmethod
    def _decode_string_literals(cls, token_ids: List[int]) -> List[Union[int, str]]:
        result = []
        str_char_count = 0
        for token_id in token_ids:
            if token_id == cls._TOKEN_TO_ID[cls._STRING_LIT]:
                str_char_count += 1
            else:
                if str_char_count > 0:
                    result.append('"' + cls._ID_TO_TOKEN[cls._UKN1] * (str_char_count - 2) + '"')
                    str_char_count = 0
                result.append(token_id)
        else:
            if str_char_count > 0:
                    result.append('"' + cls._ID_TO_TOKEN[cls._UKN1] * (str_char_count - 2) + '"')
                    str_char_count = 0
        return result

    @classmethod
    def decode(cls, token_ids: List[int]) -> str:
        partial_decode = cls._decode_string_literals(token_ids)
        text_parts = [cls._ID_TO_TOKEN[x] if isinstance(x, int) else x for x in partial_decode]
        return ''.join(text_parts)

    @classmethod
    def resize(cls, token_ids: List[int], max_length: int) -> List[int]:
        # If the token_ids are longer than max_length, truncate the start
        # If the token_ids are shorter than max_length, pad the start with _PAD
        if len(token_ids) > max_length:
            return token_ids[-max_length:]
        else:
            return [cls._PAD] * (max_length - len(token_ids)) + token_ids

    @classmethod
    def n_vocab(cls) -> int:
        return len(cls._ID_TO_TOKEN) + 1