/**
 * 文档事实核查。
 *
 * 背景：这个项目主打「把正确性摊开给你看」，那文档本身就不该说假话。
 * 实际发布前逐条核对时，查出过一批文档与代码脱节的问题：
 *   - README 动画用相对路径，docs/ 不在 npm 包内，npm 页面裂图
 *   - 路线图把早已发布的浏览器 demo 仍列为未完成
 *   - CHANGELOG 的相对链接写成 ../README.md，在仓库根目录下会跳出仓库
 *   - README 性能表里的数字其实是另一个尺寸的测量值
 *   - 宣称「不用类型化数组」，而全库都基于 Float64Array
 *   - CHANGELOG 缺 [0.2.0]/[0.2.1] 的链接引用定义
 *
 * 这些都不是代码 bug，跑测试发现不了，却恰恰是最伤公信力的那类问题。
 * 这里把它们变成可自动拦截的检查。
 *
 * 运行：node scripts/check-docs.mjs
 */

import { readFileSync, existsSync, readdirSync, statSync } from "node:fs";
import { dirname, resolve, join } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const errors = [];
const warnings = [];

const read = (p) => readFileSync(resolve(ROOT, p), "utf-8");
const pkg = JSON.parse(read("package.json"));

/* ---------------------------------------------------------------- 1. 零依赖 */

{
  const deps = Object.keys(pkg.dependencies || {});
  const dev = Object.keys(pkg.devDependencies || {});
  if (deps.length || dev.length) {
    errors.push(
      `零依赖被破坏：dependencies ${deps.length} 个、devDependencies ${dev.length} 个` +
        (deps.length ? `（${deps.join(", ")}）` : "")
    );
  }
}

/* ------------------------------------------------- 2. Markdown 相对链接可达 */

{
  const mdFiles = readdirSync(ROOT).filter((f) => f.endsWith(".md"));
  let checked = 0;
  for (const file of mdFiles) {
    const text = read(file);
    // ](target) —— 排除 http(s)、mailto、以及纯锚点
    const re = /\]\(([^)\s]+)\)/g;
    let m;
    while ((m = re.exec(text)) !== null) {
      const target = m[1];
      if (/^(https?:|mailto:|#)/.test(target)) continue;
      const path = target.split("#")[0];
      if (!path) continue;
      checked++;
      if (!existsSync(join(ROOT, path))) {
        errors.push(`${file}: 相对链接指向不存在的文件 —— ${target}`);
      }
    }
  }
  if (checked === 0) warnings.push("未找到任何相对链接，请确认正则是否失效");
}

/* ------------------------------- 3. CHANGELOG 版本标题必须有链接引用定义 */

{
  const text = read("CHANGELOG.md");
  const headings = [...text.matchAll(/^## \[(\d+\.\d+\.\d+)\]/gm)].map((m) => m[1]);
  const refs = new Set(
    [...text.matchAll(/^\[(\d+\.\d+\.\d+)\]:/gm)].map((m) => m[1])
  );
  for (const v of headings) {
    if (!refs.has(v)) {
      errors.push(`CHANGELOG.md: 版本 ${v} 缺少底部链接引用 [${v}]: ...，该链接在 GitHub 上失效`);
    }
  }
  // package.json 的当前版本必须已在 CHANGELOG 中有条目
  if (!headings.includes(pkg.version)) {
    errors.push(`CHANGELOG.md: 缺少 ${pkg.version} 的条目，但 package.json 已是该版本`);
  }
}

/* ------------------------------------------- 4. README 测试徽章 == 实际数量 */

{
  const readme = read("README.md");
  const badge = readme.match(/badge\/tests-(\d+)%20passed/);
  if (!badge) {
    warnings.push("README.md: 未找到测试数徽章，跳过该项检查");
  } else {
    const claimed = Number(badge[1]);
    let actual = null;
    // 默认 reporter 输出「ℹ tests N」，spec reporter 输出「# tests N」
    const parseCount = (s) => {
      const m = String(s).match(/^(?:ℹ|#) tests (\d+)$/m);
      return m ? Number(m[1]) : null;
    };
    try {
      actual = parseCount(
        execFileSync("node", ["--test"], {
          cwd: ROOT,
          encoding: "utf-8",
          stdio: ["ignore", "pipe", "pipe"],
          timeout: 120000,
        })
      );
    } catch (e) {
      // 测试失败时 node --test 仍会打印统计，从 stdout 里捞
      actual = parseCount(e.stdout || "");
    }
    if (actual === null) {
      warnings.push("无法解析实际测试数，跳过徽章比对");
    } else if (claimed !== actual) {
      errors.push(
        `README.md: 测试徽章写 ${claimed}，实际 ${actual} —— 徽章是硬编码的，` +
          `加/删用例会静默过期`
      );
    }
  }
}

/* --------------------------------- 5. 已作废的技术表述（曾导致可信度受损） */

{
  // 这些说法都曾出现在 README 与推广文案里，且都是错的
  const BANNED = [
    { re: /no typed arrays/i, why: "全库基于 Float64Array，正确说法是「用了类型化数组但未向量化」" },
    { re: /不做类型化|类型化数组(?!.*但)/, why: "同上" },
  ];
  for (const file of readdirSync(ROOT).filter((f) => f.endsWith(".md"))) {
    const text = read(file);
    text.split("\n").forEach((line, i) => {
      for (const { re, why } of BANNED) {
        // 排除「不要这样说」这类反向引用
        if (re.test(line) && !/不要说|注意：/.test(line)) {
          errors.push(`${file}:${i + 1} 出现已作废表述「${line.trim().slice(0, 40)}…」—— ${why}`);
        }
      }
    });
  }
}

/* --------------------------------------------- 6. README 引用的包名正确 */

{
  const readme = read("README.md");
  const m = readme.match(/npm install ([a-z0-9@._-]+)/);
  if (m && m[1] !== pkg.name) {
    errors.push(`README.md: 安装命令写的是 ${m[1]}，但 package.json 的 name 是 ${pkg.name}`);
  }
}

/* ------------------------------------------------------- 7. npm 包体积上限 */

{
  const MAX_FILES = 30;
  const MAX_KB = 200;
  try {
    const out = execFileSync(
      "npm",
      ["pack", "--dry-run", "--json", "--ignore-scripts"],
      { cwd: ROOT, encoding: "utf-8", stdio: ["ignore", "pipe", "pipe"], timeout: 120000 }
    );
    const info = JSON.parse(out)[0];
    const files = info.entryCount;
    const kb = info.unpackedSize / 1024;
    if (files > MAX_FILES) {
      warnings.push(`npm 包 ${files} 个文件，超过预期上限 ${MAX_FILES}，检查是否混入了不该发布的内容`);
    }
    if (kb > MAX_KB) {
      warnings.push(`npm 包解压后 ${kb.toFixed(0)} kB，超过上限 ${MAX_KB} kB`);
    }
    if (files === 0) errors.push("npm pack 未产出任何文件");
  } catch (e) {
    warnings.push(`无法执行 npm pack --dry-run：${String(e.message).slice(0, 80)}`);
  }
}

/* --------------------------------------------------------------- 输出汇总 */

for (const w of warnings) console.log(`⚠️  ${w}`);
for (const e of errors) console.error(`❌ ${e}`);

if (errors.length === 0) {
  console.log(
    `✅ 文档事实核查通过（${warnings.length} 条提示）` +
      (warnings.length ? "，详见上方 ⚠️" : "")
  );
  process.exit(0);
}
console.error(`\n共 ${errors.length} 个问题。`);
process.exit(1);