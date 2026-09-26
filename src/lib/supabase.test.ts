import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getSupabaseClient, resetSupabaseClient } from "./supabase";

// Hoisted with the mock, so the factory below can reference it. The real
// createClient would start an auth client with timers and storage listeners.
const { createClient } = vi.hoisted(() => ({
  createClient: vi.fn((url: string, key: string) => ({ url, key })),
}));
vi.mock("@supabase/supabase-js", () => ({ createClient }));

const URL_KEY = "NEXT_PUBLIC_SUPABASE_URL";
const ANON_KEY = "NEXT_PUBLIC_SUPABASE_ANON_KEY";

describe("getSupabaseClient", () => {
  beforeEach(() => {
    resetSupabaseClient();
    createClient.mockClear();
    vi.stubEnv(URL_KEY, "");
    vi.stubEnv(ANON_KEY, "");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    resetSupabaseClient();
  });

  it("returns null when the project isn't configured — the default for every deploy", () => {
    expect(getSupabaseClient()).toBeNull();
    expect(createClient).not.toHaveBeenCalled();
  });

  it("needs both the URL and the anon key", () => {
    vi.stubEnv(URL_KEY, "https://example.supabase.co");
    expect(getSupabaseClient()).toBeNull();
  });

  it("builds one client from the public env vars and memoises it", () => {
    vi.stubEnv(URL_KEY, "https://example.supabase.co");
    vi.stubEnv(ANON_KEY, "anon-key");
    const first = getSupabaseClient();
    const second = getSupabaseClient();
    expect(first).not.toBeNull();
    expect(second).toBe(first);
    expect(createClient).toHaveBeenCalledTimes(1);
    expect(createClient).toHaveBeenCalledWith("https://example.supabase.co", "anon-key");
  });
});

describe("the service-role key", () => {
  it("is never read by code that can reach the browser", () => {
    // SUPABASE_SERVICE_ROLE_KEY bypasses row-level security. The only safe
    // place for it is a Route Handler / Server Action, and GameML has neither,
    // so no source file under src/ may mention it at all.
    const offenders: string[] = [];
    const walk = (dir: string) => {
      for (const name of readdirSync(dir)) {
        const path = join(dir, name);
        if (statSync(path).isDirectory()) walk(path);
        else if (/\.(ts|tsx|js|mjs)$/.test(name) && !/\.test\.tsx?$/.test(name)) {
          // Comments are stripped first: supabase.ts's own security note
          // spells out the forbidden read, and prose is not a read.
          const source = readFileSync(path, "utf8")
            .replace(/\/\*[\s\S]*?\*\//g, "")
            .replace(/^\s*\/\/.*$/gm, "");
          if (/process\.env\.SUPABASE_SERVICE_ROLE_KEY|process\.env\[["']SUPABASE_SERVICE_ROLE_KEY/.test(source)) {
            offenders.push(relative(process.cwd(), path));
          }
        }
      }
    };
    walk(join(process.cwd(), "src"));
    expect(offenders).toEqual([]);
  });
});
