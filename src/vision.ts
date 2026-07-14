import { execFile } from "node:child_process"
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { promisify } from "node:util"
import { tool } from "@opencode-ai/plugin"

const execFileAsync = promisify(execFile)

// Eagle Eyes (https://github.com/christapia50/eagle-eyes — local repo:
// ~/Repos/eagle-eyes) is a standalone local-first OCR/vision backend with a
// three-tier fallback (local GLM-OCR -> Ollama Cloud -> OpenRouter). This
// plugin just shells out to it; the fallback logic lives entirely in that
// repo so it stays in one place for every consumer (OpenCode, Hermes, CLI).
const EAGLE_EYES_DIR = process.env.EAGLE_EYES_DIR ?? `${process.env.HOME}/Repos/eagle-eyes`
const EAGLE_EYES_BIN = process.env.EAGLE_EYES_BIN ?? `${EAGLE_EYES_DIR}/.venv/bin/eagle-eyes`

async function describeImagePath(path: string): Promise<string> {
  // execFile with an argument array — never a shell string — so a path
  // containing spaces/`;`/`$()` etc. is inert, not interpreted.
  const { stdout } = await execFileAsync(EAGLE_EYES_BIN, ["describe", path], {
    timeout: 45_000,
    maxBuffer: 4 * 1024 * 1024,
  })
  return stdout.trim()
}

export const visionDescribeTool = tool({
  description:
    'Describe or transcribe the contents of an image file on disk (screenshot, document scan, diagram, photo). Use this when you need to "see" an image but have no native vision support. Routes through a local OCR model first, falling back to cloud vision models if the local backend is unreachable, slow, or returns low-quality output.',
  args: {
    path: tool.schema.string().describe("Absolute path to the image file"),
  },
  async execute(args) {
    try {
      const text = await describeImagePath(args.path)
      return text || "(no description returned)"
    } catch (err) {
      return {
        output: `Failed to describe image via Eagle Eyes: ${err instanceof Error ? err.message : String(err)}`,
        metadata: { error: true },
      }
    }
  },
})

// Heuristic model-name matching for "does this model have native vision".
// experimental.chat.messages.transform doesn't hand us resolved model
// capabilities (Model.capabilities.input.image) — only the messages
// themselves, and UserMessage.model only carries {providerID, modelID}
// strings, not the resolved capability object. A name-pattern heuristic is
// a reasonable v1; if opencode's plugin API later exposes capabilities
// directly in this hook, prefer that instead.
const VISION_MODEL_PATTERNS: RegExp[] = [
  /gpt-4o/i,
  /gpt-5(\.\d+)?[-.]?(vision|v)\b/i,
  /claude-/i,
  /gemini/i,
  /glm-[\d.]+v(\b|-)/i,
  /glm-5\.1v/i,
  // Kimi's *coding*-plan variant (e.g. "kimi-k2.7-code") explicitly has no
  // vision support even though the base K2.5+ line does — exclude it.
  /kimi-k2\.(5|6|7|8|9)(?!-code)/i,
  /mimo-vl/i,
  /pixtral/i,
  /llava/i,
  /qwen.*vl/i,
  /moondream/i,
]

export function hasLikelyNativeVision(providerID: string, modelID: string): boolean {
  const id = `${providerID}/${modelID}`.toLowerCase()
  return VISION_MODEL_PATTERNS.some((re) => re.test(id))
}

type MinimalPart = {
  type: string
  mime?: string
  url?: string
  filename?: string
  [key: string]: unknown
}

async function describeAttachedImage(part: MinimalPart): Promise<string> {
  let tmpDir: string | null = null
  try {
    let path: string
    const url = part.url ?? ""
    if (url.startsWith("data:")) {
      const match = url.match(/^data:([^;]+);base64,([\s\S]*)$/)
      if (!match) throw new Error("unsupported data URL format for attached image")
      const buf = Buffer.from(match[2], "base64")
      tmpDir = await mkdtemp(join(tmpdir(), "eagle-eyes-"))
      const ext = (part.mime ?? "image/png").split("/")[1] || "png"
      path = join(tmpDir, `image.${ext}`)
      await writeFile(path, buf)
    } else if (url.startsWith("file://")) {
      path = new URL(url).pathname
    } else {
      path = url
    }
    return await describeImagePath(path)
  } finally {
    if (tmpDir) await rm(tmpDir, { recursive: true, force: true }).catch(() => {})
  }
}

/**
 * Substitutes pasted/attached image parts with a text description for
 * sessions whose active model has no native vision support. This is the
 * complement to visionDescribeTool: the tool covers "look at this file
 * path," this covers images that arrive as inline message attachments
 * (which have no file path the model could hand to a tool).
 */
export function createVisionMessagesTransform(
  log: (level: "info" | "warn" | "error", message: string) => Promise<unknown>,
) {
  return async (
    _input: Record<string, never>,
    output: { messages: { info: any; parts: MinimalPart[] }[] },
  ) => {
    for (const entry of output.messages) {
      if (entry.info?.role !== "user") continue
      const model = entry.info.model as { providerID?: string; modelID?: string } | undefined
      if (!model?.providerID || !model?.modelID) continue
      if (hasLikelyNativeVision(model.providerID, model.modelID)) continue

      const hasImage = entry.parts.some((p) => p.type === "file" && p.mime?.startsWith("image/"))
      if (!hasImage) continue

      const newParts: MinimalPart[] = []
      for (const part of entry.parts) {
        if (part.type === "file" && part.mime?.startsWith("image/")) {
          try {
            const description = await describeAttachedImage(part)
            newParts.push({
              ...part,
              type: "text",
              text: `[Image attachment "${part.filename ?? "image"}" — described by Eagle Eyes since ${model.providerID}/${model.modelID} has no native vision]:\n\n${description}`,
            })
          } catch (err) {
            await log("warn", `eagle-eyes: failed to describe attached image: ${err}`)
            newParts.push(part)
          }
        } else {
          newParts.push(part)
        }
      }
      entry.parts = newParts as any
    }
  }
}
