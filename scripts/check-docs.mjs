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
import { fileURLToPath, pathToFileURL } from "node:url";
import { execFileSync } from "node:child_process";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const errors = [];
const warnings = [];

const read = (p) => readFileSync(resolve(ROOT, p), "utf-8");
const pkg = JSON.parse(read("package.json"));
let actualTests = null;
let actualExports = null;

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
  // 递归收集全部 Markdown。docs/ 与 examples/ 里的文档同样在 GitHub 上渲染，
  // 链接断了就是 404；早期只扫仓库根目录，docs/ 里的链接坏了 CI 照样全绿。
  const mdFiles = [];
  (function walk(dir) {
    for (const entry of readdirSync(join(ROOT, dir), { withFileTypes: true })) {
      // 以 . 开头的目录一律跳过：.git、外部 Agent 工具落盘的 .workbuddy-ai/ 等
      // 都不是本仓库的内容，扫进去只会把别人的临时文件报成我们的死链接。
      if (entry.name.startsWith(".") || entry.name === "node_modules") continue;
      const rel = dir ? `${dir}/${entry.name}` : entry.name;
      if (entry.isDirectory()) walk(rel);
      else if (entry.name.endsWith(".md")) mdFiles.push(rel);
    }
  })("");

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
      // 相对链接是相对「当前文件所在目录」解析的，不是仓库根目录：
      // docs/finite-difference-gradient-checks.md 里的 ../src/gradient-check.js
      // 指的是仓库根的 src/gradient-check.js。
      const resolved = join(dirname(file), path);
      if (!existsSync(join(ROOT, resolved))) {
        errors.push(`${file}: 相对链接指向不存在的文件 —— ${target}`);
      }
    }
  }
  if (checked === 0) warnings.push("未找到任何相对链接，请确认正则是否失效");
  if (!mdFiles.some((f) => f.startsWith("docs/"))) {
    warnings.push("遍历结果里没有 docs/ 下的 Markdown，请确认递归是否失效");
  }
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

/* ---------------- 4. README 里的数字（徽章 / 正文测试数 / 导出数）== 实际 */

{
  const readme = read("README.md");

  // 默认 reporter 输出「ℹ tests N」，spec reporter 输出「# tests N」
  const parseCount = (s) => {
    const m = String(s).match(/^(?:ℹ|#) tests (\d+)$/m);
    return m ? Number(m[1]) : null;
  };

  try {
    actualTests = parseCount(
      execFileSync("node", ["--test"], {
        cwd: ROOT,
        encoding: "utf-8",
        stdio: ["ignore", "pipe", "pipe"],
        timeout: 120000,
      })
    );
  } catch (e) {
    // 测试失败时 node --test 仍会打印统计，从 stdout 里捞
    actualTests = parseCount(e.stdout || "");
  }

  if (actualTests === null) {
    warnings.push("无法解析实际测试数，跳过测试数比对");
  } else {
    const badge = readme.match(/badge\/tests-(\d+)%20passed/);
    if (!badge) {
      warnings.push("README.md: 未找到测试数徽章，跳过徽章比对");
    } else if (Number(badge[1]) !== actualTests) {
      errors.push(
        `README.md: 测试徽章写 ${badge[1]}，实际 ${actualTests} —— 徽章是硬编码的，` +
          `加/删用例会静默过期`
      );
    }

    // 正文里的测试数。徽章会记得改，正文常常忘——「60 个测试」曾挂在
    // 两个位置上，而实际已经是 80。
    const prosePatterns = [
      /共 \*\*\d+ 个导出\*\*，\*\*(\d+) 个测试\*\*/g,
      /npm test\s+#\s*(\d+) 个用例/g,
    ];
    let proseHits = 0;
    for (const re of prosePatterns) {
      for (const m of readme.matchAll(re)) {
        proseHits++;
        if (Number(m[1]) !== actualTests) {
          errors.push(
            `README.md: 正文写「${m[0].trim()}」，实际 ${actualTests} 个测试 —— ` +
              `徽章更新时正文容易漏改`
          );
        }
      }
    }
    if (proseHits === 0) {
      warnings.push("README.md: 正文未找到测试数声明，跳过该项检查");
    }
  }

  // 导出数：新增算子后最容易忘记同步的数字
  const claimedExports = [...readme.matchAll(/共 \*\*(\d+) 个导出\*\*/g)].map(
    (m) => Number(m[1])
  );
  if (claimedExports.length === 0) {
    warnings.push("README.md: 未找到导出数声明，跳过该项检查");
  } else {
    try {
      const entry = pathToFileURL(resolve(ROOT, "src/index.js")).href;
      const out = execFileSync(
        process.execPath,
        ["-e", `import(${JSON.stringify(entry)}).then(m=>console.log(Object.keys(m).length))`],
        { cwd: ROOT, encoding: "utf-8", stdio: ["ignore", "pipe", "pipe"], timeout: 60000 }
      );
      const n = Number(String(out).trim());
      if (Number.isInteger(n)) actualExports = n;
    } catch (e) {
      warnings.push(`无法统计实际导出数：${String(e.message).slice(0, 80)}`);
    }

    if (actualExports === null) {
      warnings.push("无法解析实际导出数，跳过导出数比对");
    } else {
      for (const claimed of claimedExports) {
        if (claimed !== actualExports) {
          errors.push(
            `README.md: 导出数写 ${claimed}，实际 ${actualExports} —— ` +
              `新增导出后记得同步这个数字`
          );
        }
      }
    }
  }
}

/* ----------- 4b. README「已实现」表格必须列全公开导出（双向核对） */

{
  // 数字对得上不代表清单对得上：曾出现「共 57 个导出」写对了，
  // 但下面的「已实现」表格只列了 53 个，漏掉的 4 个谁也发现不了。
  // 数字检查盯的是汇总，这里盯的是明细。
  const readme = read("README.md");
  const start = readme.indexOf("## 已实现");
  if (start === -1) {
    warnings.push("README.md: 未找到「已实现」章节，跳过表格核对");
  } else {
    const rest = readme.slice(start + "## 已实现".length);
    const nextHeading = rest.search(/^## /m);
    const section = nextHeading === -1 ? rest : rest.slice(0, nextHeading);
    // 排除 `Sequential.toJSON()` 这类「对象.方法」里的方法名
    const listed = new Set(
      [...section.matchAll(/(?<!\.)`([A-Za-z_][A-Za-z0-9_]*)`/g)].map((m) => m[1])
    );

    let names = null;
    try {
      const api = await import(pathToFileURL(resolve(ROOT, "src/index.js")).href);
      names = Object.keys(api);
    } catch (e) {
      warnings.push(`无法加载 src/index.js 以核对「已实现」表格：${String(e.message).slice(0, 80)}`);
    }

    if (names) {
      const missing = names.filter((n) => !listed.has(n));
      if (missing.length) {
        errors.push(
          `README.md「已实现」表格漏列 ${missing.length} 个公开导出 —— ${missing.join(", ")}` +
            `（汇总数字对得上，明细却少列了，读者会以为没有这些 API）`
        );
      }
      const extra = [...listed].filter((n) => !names.includes(n));
      if (extra.length) {
        warnings.push(
          `README.md「已实现」表格列了非公开导出 —— ${extra.join(", ")}，确认是否为笔误`
        );
      }
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

/* ------------------------------- 6. README 安装来源与 package.json 一致 */

{
  const readme = read("README.md");
  const m = readme.match(/npm install\s+([^\s`]+)/);
  if (m) {
    const spec = m[1];
    const repoMatch = String(pkg.repository?.url || "").match(
      /github\.com[/:]([^/]+)\/([^/.#]+)/
    );
    const githubSpec = repoMatch
      ? `github:${repoMatch[1]}/${repoMatch[2]}`
      : null;
    if (spec !== pkg.name && spec !== githubSpec) {
      errors.push(
        `README.md: 安装命令写的是 ${spec}，但 package.json 的 name 是 ${pkg.name}` +
          (githubSpec ? `，仓库地址对应 ${githubSpec}` : "")
      );
    }
  }
}

/* ------------------------------------------------ 7. 英文 API 文档覆盖导出 */

{
  let api;
  try {
    api = await import(pathToFileURL(resolve(ROOT, "src/index.js")).href);
  } catch (e) {
    errors.push(`无法加载 src/index.js 以检查 API 文档：${String(e.message).slice(0, 120)}`);
  }

  if (api) {
    const apiText = read("docs/api.md");
    const missing = Object.keys(api).filter(
      (name) => !apiText.includes(`\`${name}\``)
    );
    if (missing.length) {
      errors.push(`docs/api.md: 缺少公开导出的 API 文档 —— ${missing.join(", ")}`);
    }
  }
}

/* ------------------------------------------------------- 8. npm 包体积上限 */

{
  const MAX_FILES = 30;
  const MAX_KB = 200;

  // 试一次默认配置；失败则换独立缓存目录重试。
  // 本机 ~/.npm 若属主异常（如被别的 uid 写过），默认缓存会报 EEXIST，
  // 此时体积检查不该被静默跳过——退一步用临时缓存即可。
  const runPack = (extraArgs) =>
    execFileSync(
      "npm",
      ["pack", "--dry-run", "--json", "--ignore-scripts", ...extraArgs],
      { cwd: ROOT, encoding: "utf-8", stdio: ["ignore", "pipe", "pipe"], timeout: 120000 }
    );

  let info = null;
  let lastErr = null;
  for (const extra of [[], ["--cache", "/tmp/.axon-npm-cache"]]) {
    try {
      info = JSON.parse(runPack(extra))[0];
      break;
    } catch (e) {
      lastErr = e;
    }
  }

  if (!info) {
    warnings.push(`无法执行 npm pack --dry-run：${String(lastErr?.message).slice(0, 80)}`);
  } else {
    const files = info.entryCount;
    const kb = info.unpackedSize / 1024;
    if (files > MAX_FILES) {
      warnings.push(`npm 包 ${files} 个文件，超过预期上限 ${MAX_FILES}，检查是否混入了不该发布的内容`);
    }
    if (kb > MAX_KB) {
      warnings.push(`npm 包解压后 ${kb.toFixed(0)} kB，超过上限 ${MAX_KB} kB`);
    }
    if (files === 0) errors.push("npm pack 未产出任何文件");
  }
}

/* ------------------------- 9. 官网首页的产品数字与仓库事实一致 */

{
  for (const file of ["site/index.html", "site/en/index.html"]) {
    const text = read(file);

    const versions = [
      ...text.matchAll(/AXN(?:\s*\/\/\s*CORE|\s*-\s*CORE)\s+(\d+\.\d+\.\d+)/g),
    ].map((m) => m[1]);
    if (!versions.includes(pkg.version)) {
      errors.push(
        `${file}: 页面版本写 ${versions.join(", ") || "（未找到）"}，` +
          `package.json 是 ${pkg.version} —— 发布新版本时首页容易漏改`
      );
    }

    const selfTests = [
      ...text.matchAll(/SELF-TEST\s+(\d+)\/(\d+)\s+PASS/g),
    ];
    if (selfTests.length === 0) {
      warnings.push(`${file}: 未找到 SELF-TEST 声明，跳过测试数比对`);
    } else if (actualTests !== null) {
      for (const [, passed, total] of selfTests) {
        if (Number(passed) !== actualTests || Number(total) !== actualTests) {
          errors.push(
            `${file}: SELF-TEST 写 ${passed}/${total}，实际 ${actualTests} 个测试 —— ` +
              `首页状态栏需要随测试数同步`
          );
        }
      }
    }

    const exportsMatch = text.match(
      /<span>EXPORTS<\/span>[\s\S]*?<em>(\d+)<\/em>/
    );
    if (!exportsMatch) {
      warnings.push(`${file}: 未找到 EXPORTS 数量，跳过导出数比对`);
    } else if (actualExports !== null && Number(exportsMatch[1]) !== actualExports) {
      errors.push(
        `${file}: EXPORTS 写 ${exportsMatch[1]}，实际 ${actualExports} —— ` +
          `新增公开导出后首页容易漏改`
      );
    }
  }
}

/* ------------------ 9b. 社交预览卡模板的数字与仓库事实一致 */

{
  // site/og-card.html 是 site/og.png 的源模板（1200×630，人工截图导出）。og.png 会被
  // 发布并作为两个官网首页的 og:image，模板里的数字一旦漂移，重新出图就是错的——
  // 而第 9 段只核对两个官网首页，这张卡一直是盲区。
  const file = "site/og-card.html";
  const text = read(file);

  const versions = [
    ...text.matchAll(/AXN\s*\/\/\s*CORE\s+(\d+\.\d+\.\d+)/g),
  ].map((m) => m[1]);
  if (!versions.includes(pkg.version)) {
    errors.push(
      `${file}: 版本写 ${versions.join(", ") || "（未找到）"}，` +
        `package.json 是 ${pkg.version} —— 预览卡模板容易漏改`
    );
  }

  const selfTests = [...text.matchAll(/SELF-TEST\s+(\d+)\/(\d+)\s+PASS/g)];
  if (selfTests.length === 0) {
    warnings.push(`${file}: 未找到 SELF-TEST 声明，跳过测试数比对`);
  } else if (actualTests !== null) {
    for (const [, passed, total] of selfTests) {
      if (Number(passed) !== actualTests || Number(total) !== actualTests) {
        errors.push(
          `${file}: SELF-TEST 写 ${passed}/${total}，实际 ${actualTests} 个测试`
        );
      }
    }
  }

  // 这张卡的版式与官网首页不同：导出数是「57 EXPORTS」一段文字，
  // 测试数在右下角的读数区 <span>TESTS</span> ... <em>114</em>。
  const exportsMatch = text.match(/(\d+)\s+EXPORTS/);
  if (!exportsMatch) {
    warnings.push(`${file}: 未找到 EXPORTS 数量，跳过导出数比对`);
  } else if (actualExports !== null && Number(exportsMatch[1]) !== actualExports) {
    errors.push(
      `${file}: EXPORTS 写 ${exportsMatch[1]}，实际 ${actualExports} —— ` +
        `新增公开导出后预览卡容易漏改`
    );
  }

  const testsRow = text.match(/<span>TESTS<\/span>[\s\S]*?<em>(\d+)<\/em>/);
  if (!testsRow) {
    warnings.push(`${file}: 未找到 TESTS 读数，跳过比对`);
  } else if (actualTests !== null && Number(testsRow[1]) !== actualTests) {
    errors.push(
      `${file}: TESTS 读数写 ${testsRow[1]}，实际 ${actualTests} 个测试`
    );
  }
}

/* ------------ 10. 官网页面的相对链接在 Pages 部署布局下可达 */

{
  // .github/workflows/pages.yml 把 site/index.html 铺到 _site 根、site/en 铺到
  // _site/en/，再把 demo/、src/、examples/browser/ 等拷到根。于是 URL 路径
  // /axon/xxx 对应的仓库文件是：en/** → site/en/**，index.html → site/index.html，
  // og.png → site/og.png，其余 → 仓库根同名路径。
  //
  // 起因：英文站正文曾把示例链接写成相对当前页的 ./examples/...，发布在
  // /axon/en/ 下会解析成 /axon/en/examples/... 而 404，页脚却是 ../examples/，
  // 两处不一致。第 9 段只核对版本号 / SELF-TEST / EXPORTS，链接坏了照样全绿。
  const toRepoPath = (urlPath) => {
    if (urlPath.startsWith("/en/")) return `site/en/${urlPath.slice(4)}`;
    if (urlPath === "/index.html") return "site/index.html";
    if (urlPath === "/og.png") return "site/og.png";
    return urlPath.slice(1);
  };

  const pages = [
    ["site/index.html", "https://site.invalid/"],
    ["site/en/index.html", "https://site.invalid/en/"],
  ];

  let checkedLinks = 0;
  for (const [file, base] of pages) {
    const text = read(file);
    const re = /(?:href|src)="([^"]+)"/g;
    let m;
    while ((m = re.exec(text)) !== null) {
      const target = m[1];
      if (/^(https?:|mailto:|data:|javascript:|#)/.test(target)) continue;
      let urlPath = decodeURIComponent(new URL(target, base).pathname);
      if (urlPath.endsWith("/")) urlPath += "index.html";
      const repoPath = toRepoPath(urlPath);
      checkedLinks++;
      if (!existsSync(join(ROOT, repoPath))) {
        errors.push(
          `${file}: 相对链接 ${target} 在 Pages 布局下解析为 ${urlPath}，` +
            `对应仓库文件 ${repoPath} 不存在 —— 部署后是 404`
        );
      }
    }
  }
  if (checkedLinks === 0) {
    warnings.push("官网页面未找到任何相对链接，请确认正则是否失效");
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
