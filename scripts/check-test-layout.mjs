/**
 * 测试文件结构自检。
 *
 * 背景：这个项目被 Node 24 的"宽容"骗过好几次。
 * `describe` 嵌套错位、文件末尾残留 `__PART2__` 之类的编辑痕迹，
 * 在 Node 24 上要么被忽略、要么照常通过；但 Node 18/20/22 会
 * 判定子测试「未在父测试结束前完成」而整片取消。
 *
 * 结果就是本地全绿、CI 全红，排查成本极高。
 *
 * 这里做三项静态检查，把这类问题挡在提交之前：
 *   1. 花括号必须收支平衡
 *   2. 顶层 describe 与顶层 }); 必须数量一致（不允许 suite 漏收尾）
 *   3. 不得残留分块写入的占位标记
 *
 * 运行：node scripts/check-test-layout.mjs
 */

import { readdirSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const testDir = resolve(ROOT, "test");
const errors = [];

/** 占位标记：分块写文件时容易忘记清理 */
const PLACEHOLDERS = [/__PART\d+__/, /TODO\s*:/, /\bFIXME\b/];

const files = readdirSync(testDir).filter((f) => f.endsWith(".test.js"));

for (const file of files) {
  const full = resolve(testDir, file);
  const source = readFileSync(full, "utf-8");
  const lines = source.split("\n");

  // 1. 花括号收支
  let depth = 0;
  for (let i = 0; i < lines.length; i++) {
    for (const ch of lines[i]) {
      if (ch === "{") depth++;
      else if (ch === "}") depth--;
    }
  }
  if (depth !== 0) {
    errors.push(`${file}: 花括号收支不平衡（差 ${depth}）`);
  }

  // 2. 顶层 describe 与顶层 }); 数量一致
  const describes = lines.filter((l) => /^describe\(/.test(l)).length;
  const closers = lines.filter((l) => l === "});").length;
  if (describes !== closers) {
    errors.push(
      `${file}: 顶层 describe ${describes} 个，但顶层 "});" ${closers} 个` +
      ` —— 疑似 suite 漏收尾，导致后续 describe 被嵌进上一个 suite`
    );
  }

  // 3. 不得有缩进过的 describe（说明嵌错了位置）
  const indented = lines
    .map((l, i) => ({ l, i }))
    .filter(({ l }) => /^\s+describe\(/.test(l));
  if (indented.length) {
    errors.push(
      `${file}: 第 ${indented[0].i + 1} 行 describe 有缩进，疑似嵌套错位`
    );
  }

  // 4. 占位标记
  for (const pattern of PLACEHOLDERS) {
    const hit = lines.findIndex((l) => pattern.test(l));
    if (hit >= 0) {
      errors.push(`${file}: 第 ${hit + 1} 行残留占位标记 ${l0(pattern)}`);
    }
  }
}

function l0(pattern) {
  return String(pattern).replace(/[\\^$*+?.()|[\]{}]/g, "").split("\\")[0];
}

if (files.length === 0) {
  errors.push("test/ 目录下没有找到任何 .test.js 文件");
}

for (const e of errors) console.error(`❌ ${e}`);

if (errors.length === 0) {
  console.log(`✅ 测试文件结构自检通过（${files.length} 个文件）`);
  process.exit(0);
}
console.error(`\n共 ${errors.length} 个问题。`);
process.exit(1);