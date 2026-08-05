/**
 * A tiny, dependency-free syntax tokenizer.
 *
 * DESIGN.md §6 asks `<MathDrawer>` to show "syntax-highlighted code", and the
 * code lane wants the same treatment. Pulling in Shiki/Prism for that is a lot
 * of bundle for two panels, so this scans just enough JavaScript and Python to
 * colour keywords, strings, numbers, comments and calls.
 *
 * It is intentionally *not* a parser: it never throws, and unknown input falls
 * through as `plain`. Colour is applied by the consumer from DESIGN.md tokens,
 * which keeps this module pure and unit-testable.
 *
 * Highlighting is decorative. The code is always readable without colour, so
 * this carries no accessibility weight (DESIGN.md §9: never encode meaning in
 * colour alone).
 */

export type TokenKind =
  | "keyword"
  | "string"
  | "number"
  | "comment"
  | "call"
  | "punctuation"
  | "plain";

export interface Token {
  kind: TokenKind;
  text: string;
}

export type HighlightLanguage = "javascript" | "python";

const JS_KEYWORDS = new Set([
  "async", "await", "break", "case", "catch", "class", "const", "continue",
  "default", "delete", "do", "else", "export", "extends", "false", "finally",
  "for", "from", "function", "if", "import", "in", "instanceof", "let", "new",
  "null", "of", "return", "static", "super", "switch", "this", "throw", "true",
  "try", "typeof", "undefined", "var", "void", "while", "yield",
]);

const PY_KEYWORDS = new Set([
  "and", "as", "assert", "async", "await", "break", "class", "continue", "def",
  "del", "elif", "else", "except", "False", "finally", "for", "from", "global",
  "if", "import", "in", "is", "lambda", "None", "nonlocal", "not", "or", "pass",
  "raise", "return", "True", "try", "while", "with", "yield",
]);

const PUNCTUATION = new Set([
  "{", "}", "(", ")", "[", "]", ";", ",", ".", ":", "=", "+", "-", "*", "/",
  "%", "<", ">", "!", "&", "|", "?", "^", "~",
]);

const IDENT_START = /[A-Za-z_$]/;
const IDENT_BODY = /[A-Za-z0-9_$]/;
const DIGIT = /[0-9]/;

/** Tokenize `code`. Never throws; always round-trips (tokens rejoin to input). */
export function highlight(
  code: string,
  language: HighlightLanguage = "javascript",
): Token[] {
  const keywords = language === "python" ? PY_KEYWORDS : JS_KEYWORDS;
  const lineComment = language === "python" ? "#" : "//";

  const tokens: Token[] = [];
  let plain = "";

  const flushPlain = () => {
    if (plain) {
      tokens.push({ kind: "plain", text: plain });
      plain = "";
    }
  };

  const push = (kind: TokenKind, text: string) => {
    flushPlain();
    tokens.push({ kind, text });
  };

  let i = 0;
  while (i < code.length) {
    const ch = code[i]!;
    const rest = code.slice(i);

    // --- comments ---------------------------------------------------------
    if (rest.startsWith(lineComment)) {
      const end = code.indexOf("\n", i);
      const stop = end === -1 ? code.length : end;
      push("comment", code.slice(i, stop));
      i = stop;
      continue;
    }

    if (language === "javascript" && rest.startsWith("/*")) {
      const end = code.indexOf("*/", i + 2);
      const stop = end === -1 ? code.length : end + 2;
      push("comment", code.slice(i, stop));
      i = stop;
      continue;
    }

    // --- strings ----------------------------------------------------------
    if (ch === '"' || ch === "'" || ch === "`") {
      const quote = ch;
      let j = i + 1;
      while (j < code.length) {
        if (code[j] === "\\") {
          j += 2;
          continue;
        }
        if (code[j] === quote) {
          j += 1;
          break;
        }
        // Unterminated single-quote strings stop at the newline so one typo
        // doesn't paint the rest of the snippet as a string.
        if (code[j] === "\n" && quote !== "`") break;
        j += 1;
      }
      push("string", code.slice(i, j));
      i = j;
      continue;
    }

    // --- numbers ----------------------------------------------------------
    if (DIGIT.test(ch)) {
      let j = i;
      while (j < code.length && /[0-9a-fA-FxX._eE+-]/.test(code[j]!)) {
        // Stop `+`/`-` unless it's an exponent sign.
        if (
          (code[j] === "+" || code[j] === "-") &&
          !/[eE]/.test(code[j - 1] ?? "")
        ) {
          break;
        }
        j += 1;
      }
      push("number", code.slice(i, j));
      i = j;
      continue;
    }

    // --- identifiers, keywords, calls -------------------------------------
    if (IDENT_START.test(ch)) {
      let j = i;
      while (j < code.length && IDENT_BODY.test(code[j]!)) j += 1;
      const word = code.slice(i, j);

      if (keywords.has(word)) {
        push("keyword", word);
      } else {
        // A following "(" (allowing spaces) makes it a call.
        let k = j;
        while (k < code.length && (code[k] === " " || code[k] === "\t")) k += 1;
        if (code[k] === "(") push("call", word);
        else plain += word;
      }
      i = j;
      continue;
    }

    // --- punctuation ------------------------------------------------------
    if (PUNCTUATION.has(ch)) {
      push("punctuation", ch);
      i += 1;
      continue;
    }

    plain += ch;
    i += 1;
  }

  flushPlain();
  return tokens;
}

/**
 * CSS colour for a token kind, drawn from DESIGN.md §2 tokens only.
 * Keys map to `var(--…)` so themes stay in one place.
 */
export const TOKEN_COLOR: Record<TokenKind, string> = {
  keyword: "var(--class-d)", // pink
  string: "var(--correct)", // green
  number: "var(--class-b)", // orange
  comment: "var(--text-muted)",
  call: "var(--primary)",
  punctuation: "var(--text-muted)",
  plain: "var(--text)",
};
