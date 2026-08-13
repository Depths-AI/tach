interface TypeRef {
  readonly tach: string;
  readonly kind: "void" | "bool" | "i32" | "u32" | "f32" | "vector" | "struct" | "atomic" | "fixedArray" | "runtimeArray";
  readonly name?: string;
  readonly elem?: TypeRef;
  readonly count?: number;
  readonly lanes?: number;
}
interface Field { readonly name: string; readonly type: TypeRef; readonly description?: string }
interface DocumentedType { readonly name: string; readonly summary?: string; readonly fields: readonly Field[] }
interface Parameter { readonly name: string; readonly type: TypeRef; readonly buffer: boolean; readonly access?: "read" | "write" | "readWrite" | "atomic"; readonly description?: string }
interface FunctionDoc {
  readonly name: string;
  readonly exported: boolean;
  readonly summary?: string;
  readonly coordinates: readonly { readonly name: string; readonly description?: string }[];
  readonly parameters: readonly Parameter[];
  readonly returns?: { readonly type: TypeRef; readonly description?: string };
}
export interface Documentation {
  readonly schema: 1;
  readonly source: string;
  readonly title?: string;
  readonly summary?: string;
  readonly types: readonly DocumentedType[];
  readonly functions: readonly FunctionDoc[];
}

export function renderDocumentation(module: Documentation): string {
  if (module.schema !== 1 || !Array.isArray(module.types) || !Array.isArray(module.functions)) throw new TypeError("invalid Tach documentation description");
  const title = module.title ?? module.source.replace(/\.[^.]*$/u, "");
  let markdown = `# ${title}\n\n${paragraph(module.summary)}`;
  if (module.types.length) {
    markdown += "## Types\n\n";
    for (const type of module.types) markdown += renderType(type);
  }
  const privateFunctions = module.functions.filter((fn) => !fn.exported);
  const exportedFunctions = module.functions.filter((fn) => fn.exported);
  if (privateFunctions.length) {
    markdown += "## Internal functions and stages\n\n";
    for (const fn of privateFunctions) markdown += renderFunction(fn);
  }
  if (exportedFunctions.length) {
    markdown += "## Exported programs\n\n";
    for (const fn of exportedFunctions) markdown += renderFunction(fn);
    markdown += "## TypeScript usage\n\nThis example is generated from the compiler-validated API.\n\n```ts\n";
    markdown += usage(module, exportedFunctions) + "```\n";
  }
  return markdown;
}

function renderType(type: DocumentedType): string {
  let out = `### \`${type.name}\`\n\n${paragraph(type.summary)}\`\`\`tach\ntype ${type.name} = {\n`;
  for (const field of type.fields) out += `  ${field.name}: ${field.type.tach},\n`;
  out += "};\n```\n\n| Field | Type | Description |\n|---|---|---|\n";
  for (const field of type.fields) out += `| \`${field.name}\` | \`${field.type.tach}\` | ${cell(field.description)} |\n`;
  return out + "\n";
}

function renderFunction(fn: FunctionDoc): string {
  const prefix = fn.exported ? "export " : "";
  const coordinates = fn.coordinates.length ? `[${fn.coordinates.map((item) => item.name).join(", ")}]` : "";
  const result = fn.returns ? `: ${fn.returns.type.tach}` : "";
  let out = `### \`${fn.name}\`\n\n${paragraph(fn.summary)}\`\`\`tach\n${prefix}function ${fn.name}${coordinates}(${fn.parameters.map((parameter) => `${parameter.name}: ${parameter.buffer ? `buffer<${parameter.type.tach}>` : parameter.type.tach}`).join(", ")})${result}\n\`\`\`\n\n`;
  if (fn.coordinates.length) {
    out += "| Coordinate | Type | Description |\n|---|---|---|\n";
    for (const coordinate of fn.coordinates) out += `| \`${coordinate.name}\` | \`uint32\` | ${cell(coordinate.description)} |\n`;
    out += "\n";
  }
  if (fn.parameters.length) {
    out += "| Parameter | Context | Type | Description |\n|---|---|---|---|\n";
    for (const parameter of fn.parameters) out += `| \`${parameter.name}\` | ${parameter.buffer ? access(parameter.access) : "value input"} | \`${parameter.buffer ? `buffer<${parameter.type.tach}>` : parameter.type.tach}\` | ${cell(parameter.description)} |\n`;
    out += "\n";
  }
  if (fn.returns) out += `**Returns:** \`${fn.returns.type.tach}\`${fn.returns.description ? ` — ${fn.returns.description}` : ""}\n\n`;
  return out;
}

function usage(module: Documentation, functions: readonly FunctionDoc[]): string {
  const base = module.source.replace(/\.[^.]*$/u, "");
  let out = `import { tach } from "@depths/tach";\nimport * as kernels from ${JSON.stringify(`./build/${base}.js`)};\n\n`;
  for (const fn of functions) {
    out += `export async function run_${fn.name}(\n`;
    for (const parameter of fn.parameters) out += `  ${parameter.name}: ${tsType(parameter.type)},\n`;
    if (fn.coordinates.length) out += `  $size: ${fn.coordinates.length === 1 ? "number" : `readonly [${fn.coordinates.map(() => "number").join(", ")}]`},\n`;
    out += ") {\n  return tach(async (gpu) => {\n";
    for (const parameter of fn.parameters) if (parameter.buffer) out += `    const $${parameter.name} = gpu.buffer(${parameter.name});\n`;
    const args = fn.parameters.map((parameter) => `${parameter.buffer ? "$" : ""}${parameter.name}`);
    if (fn.coordinates.length) args.push("{ size: $size }");
    out += `    await gpu.submit(kernels.${fn.name}(${args.join(", ")}));\n`;
    const output = fn.parameters.find((parameter) => parameter.buffer && parameter.access !== "read");
    if (output) out += `    return $${output.name}.read();\n`;
    out += "  });\n}\n\n";
  }
  return out;
}

function tsType(type: TypeRef): string {
  switch (type.kind) {
    case "bool": return "boolean";
    case "i32": case "u32": case "f32": case "atomic": return "number";
    case "struct": return `kernels.${type.name!}`;
    case "vector": return `readonly [${Array(type.lanes).fill("number").join(", ")}]`;
    case "fixedArray": return `readonly ${tsType(type.elem!)}[]`;
    case "runtimeArray": return typedArray(type.elem!) || `readonly ${tsType(type.elem!)}[]`;
    default: return "never";
  }
}

function typedArray(type: TypeRef): string | undefined {
  const element = type.kind === "atomic" ? type.elem! : type.kind === "vector" && type.lanes !== 3 ? type.elem! : type;
  if (type.kind === "vector" && type.lanes === 3) return undefined;
  const name = element.kind === "i32" ? "Int32Array" : element.kind === "u32" ? "Uint32Array" : element.kind === "f32" ? "Float32Array" : undefined;
  if (!name) return undefined;
  return type.kind === "vector" ? `${name} | ReadonlyArray<${tsType(type)}>` : `${name} | readonly ${tsType(type)}[]`;
}

function paragraph(value?: string): string { return value ? `${value}\n\n` : "" }
function cell(value?: string): string { return value ? value.replaceAll("|", "\\|").replaceAll("\n", "<br>") : "—" }
function access(value?: Parameter["access"]): string {
  return value === "atomic" ? "GPU buffer · atomic" : value === "readWrite" ? "GPU buffer · read/write" : value === "write" ? "GPU buffer · output" : "GPU buffer · read-only";
}
