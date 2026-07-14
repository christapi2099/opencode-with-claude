import assert from "node:assert/strict"
import test from "node:test"
import { hasLikelyNativeVision, visionDescribeTool } from "../../src/vision.ts"

test("hasLikelyNativeVision recognizes known vision-capable models", () => {
  assert.equal(hasLikelyNativeVision("openai", "gpt-4o"), true)
  assert.equal(hasLikelyNativeVision("anthropic", "claude-opus-4-8"), true)
  assert.equal(hasLikelyNativeVision("google", "gemini-2.5-flash"), true)
  assert.equal(hasLikelyNativeVision("zhipuai", "glm-4.6v"), true)
  assert.equal(hasLikelyNativeVision("moonshot", "kimi-k2.7"), true)
  assert.equal(hasLikelyNativeVision("xiaomi", "mimo-vl-7b"), true)
})

test("hasLikelyNativeVision rejects known text-only models", () => {
  assert.equal(hasLikelyNativeVision("zhipuai", "glm-4.7"), false)
  assert.equal(hasLikelyNativeVision("deepseek", "deepseek-chat"), false)
  assert.equal(hasLikelyNativeVision("moonshot", "kimi-k2.7-code"), false)
})

test("visionDescribeTool is registered with an execute function and a path arg", () => {
  assert.equal(typeof visionDescribeTool.execute, "function")
  assert.ok(visionDescribeTool.args.path, "expected a 'path' arg in the schema")
})
