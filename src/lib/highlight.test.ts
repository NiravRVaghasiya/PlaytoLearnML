import { describe, expect, it } from "vitest";
import { TOKEN_COLOR, highlight, type TokenKind } from "./highlight";

const join = (code: string, language?: "javascript" | "python") =>
  highlight(code, language)
    .map((t) => t.text)
    .join("");

const kindsOf = (code: string, kind: TokenKind, language?: "javascript" | "python") =>
  highlight(code, language)
    .filter((t) => t.kind === kind)
    .map((t) => t.text);

describe("highlight", () => {
  it("round-trips the input exactly", () => {
    const samples = [
      "const lr = 0.03;",
      "// a comment\nreturn x;",
      "def step(x): return x * 2  # python",
      "",
      "   ",
      "a?.b ?? c",
      "`template ${value}`",
    ];
    for (const sample of samples) {
      expect(join(sample)).toBe(sample);
      expect(join(sample, "python")).toBe(sample);
    }
  });

  it("finds JavaScript keywords", () => {
    expect(kindsOf("const x = await fetchIt();", "keyword")).toEqual([
      "const",
      "await",
    ]);
  });

  it("finds Python keywords and not JS-only ones", () => {
    expect(kindsOf("def f(): return None", "keyword", "python")).toEqual([
      "def",
      "return",
      "None",
    ]);
    // `def` is not a JavaScript keyword.
    expect(kindsOf("def f()", "keyword", "javascript")).toEqual([]);
  });

  it("marks function calls", () => {
    expect(kindsOf("tf.tensor2d(xs)", "call")).toEqual(["tensor2d"]);
    expect(kindsOf("spaced   (1)", "call")).toEqual(["spaced"]);
  });

  it("handles all three JS string delimiters", () => {
    expect(kindsOf(`'a' "b" \`c\``, "string")).toEqual(["'a'", '"b"', "`c`"]);
  });

  it("does not let an unterminated string swallow the rest of the file", () => {
    const tokens = highlight("const a = 'oops\nconst b = 2;");
    const strings = tokens.filter((t) => t.kind === "string").map((t) => t.text);
    expect(strings).toEqual(["'oops"]);
    // The second line still tokenizes normally.
    expect(tokens.some((t) => t.kind === "keyword" && t.text === "const")).toBe(
      true,
    );
  });

  it("respects escaped quotes", () => {
    expect(kindsOf(`'it\\'s fine'`, "string")).toEqual([`'it\\'s fine'`]);
  });

  it("reads line and block comments", () => {
    expect(kindsOf("// note\ncode", "comment")).toEqual(["// note"]);
    expect(kindsOf("/* block\nspan */ code", "comment")).toEqual([
      "/* block\nspan */",
    ]);
    expect(kindsOf("# hash", "comment", "python")).toEqual(["# hash"]);
    // `#` is not a comment in JavaScript.
    expect(kindsOf("# hash", "comment", "javascript")).toEqual([]);
  });

  it("reads numbers including decimals, exponents and hex", () => {
    expect(kindsOf("0.03 1e-4 0xff 42", "number")).toEqual([
      "0.03",
      "1e-4",
      "0xff",
      "42",
    ]);
  });

  it("does not fold a following minus into a plain number", () => {
    // "3-1" must not tokenize as the single number "3-1".
    expect(kindsOf("3-1", "number")).toEqual(["3", "1"]);
    expect(kindsOf("3-1", "punctuation")).toEqual(["-"]);
  });

  it("never throws on hostile input", () => {
    const nasty = ["/*", "'", "`${", "0x", "\\", "\u0000", "0e+"];
    for (const input of nasty) {
      expect(() => highlight(input)).not.toThrow();
      expect(join(input)).toBe(input);
    }
  });

  it("only uses DESIGN.md colour tokens", () => {
    for (const value of Object.values(TOKEN_COLOR)) {
      expect(value).toMatch(/^var\(--[a-z-]+\)$/);
    }
  });
});
