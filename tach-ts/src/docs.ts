export interface TypeRef {
  readonly tach: string;
  readonly kind:
    | "void"
    | "bool"
    | "i32"
    | "u32"
    | "f32"
    | "vector"
    | "struct"
    | "atomic"
    | "fixedArray"
    | "runtimeArray";
  readonly name?: string;
  readonly elem?: TypeRef;
  readonly count?: number;
  readonly lanes?: number;
}
export interface Field {
  readonly name: string;
  readonly type: TypeRef;
  readonly description?: string;
}
export interface DocumentedType {
  readonly name: string;
  readonly summary?: string;
  readonly fields: readonly Field[];
}
export interface Parameter {
  readonly name: string;
  readonly type: TypeRef;
  readonly buffer: boolean;
  readonly access?: "read" | "write" | "readWrite" | "atomic";
  readonly description?: string;
}
export interface FunctionDoc {
  readonly name: string;
  readonly role: "helper" | "stage" | "kernel" | "program";
  readonly exported: boolean;
  readonly summary?: string;
  readonly coordinates: readonly {
    readonly name: string;
    readonly description?: string;
  }[];
  readonly parameters: readonly Parameter[];
  readonly returns?: { readonly type: TypeRef; readonly description?: string };
}
export interface KernelDoc {
  readonly name: string;
  readonly identity: string;
  readonly title?: string;
  readonly summary?: string;
  readonly types: readonly DocumentedType[];
  readonly functions: readonly FunctionDoc[];
}
export interface ModuleDoc {
  readonly name: string;
  readonly kernels: readonly KernelDoc[];
}
export interface Documentation {
  readonly schema: 2;
  readonly name: string;
  readonly version: string;
  readonly package: string;
  readonly title: string;
  readonly summary: string;
  readonly modules: readonly ModuleDoc[];
}

export interface DocumentationFiles {
  readonly readme: string;
  readonly modules: ReadonlyMap<string, string>;
}

export function renderDocumentation(
  project: Documentation,
): DocumentationFiles {
  validate(project);
  const modules = new Map<string, string>();
  for (const module of project.modules) {
    let markdown = `# ${module.name}\n\n`;
    for (const kernel of module.kernels) {
      markdown += `## ${
        kernel.title ?? kernel.name
      }\n\n**Source:** \`${kernel.identity}\`\n\n${paragraph(kernel.summary)}`;
      if (kernel.types.length) {
        markdown += "### Types\n\n";
        for (const type of kernel.types) markdown += renderType(type);
      }
      const internal = kernel.functions.filter((fn) => !fn.exported);
      const exported = kernel.functions.filter((fn) => fn.exported);
      if (internal.length) {
        markdown += "### Internal functions and stages\n\n";
        for (const fn of internal) markdown += renderFunction(fn);
      }
      if (exported.length) {
        markdown += "### TypeScript-callable programs\n\n";
        for (const fn of exported) markdown += renderFunction(fn);
      }
    }
    modules.set(module.name, markdown);
  }
  let readme =
    `# ${project.title}\n\n${project.summary}\n\n## Installation\n\n\`\`\`console\nnpm install @depths/tach ${project.package}\n\`\`\`\n\n## TypeScript usage\n\nThis example is generated from the compiler-validated public ABI.\n\n\`\`\`ts\n${
      usage(project)
    }\`\`\`\n\n## Modules\n\n`;
  for (const module of project.modules) {
    readme += `- [${module.name}](docs/${
      encodeURIComponent(module.name)
    }.md)\n`;
  }
  return { readme, modules };
}

function validate(project: Documentation): void {
  if (
    project.schema !== 2 || !Array.isArray(project.modules) || !project.name ||
    !project.version || !project.package || !project.title || !project.summary
  ) {
    throw new TypeError("invalid Tach project documentation description");
  }
}

function renderType(type: DocumentedType): string {
  let out = `#### \`${type.name}\`\n\n${
    paragraph(type.summary)
  }\`\`\`tach\ntype ${type.name} = {\n`;
  for (const field of type.fields) {
    out += `  ${field.name}: ${field.type.tach},\n`;
  }
  out += "};\n```\n\n| Field | Type | Description |\n|---|---|---|\n";
  for (const field of type.fields) {
    out += `| \`${field.name}\` | \`${field.type.tach}\` | ${
      cell(field.description)
    } |\n`;
  }
  return out + "\n";
}

function renderFunction(fn: FunctionDoc): string {
  const prefix = fn.exported ? "export " : "";
  const coordinates = fn.coordinates.length
    ? `[${fn.coordinates.map((item) => item.name).join(", ")}]`
    : "";
  const result = fn.returns ? `: ${fn.returns.type.tach}` : "";
  let out = `#### \`${fn.name}\`\n\n${
    paragraph(fn.summary)
  }**Role:** ${fn.role}${
    fn.exported ? " · exported to TypeScript" : " · Tach-internal"
  }\n\n\`\`\`tach\n${prefix}function ${fn.name}${coordinates}(${
    fn.parameters.map((parameter) =>
      `${parameter.name}: ${
        parameter.buffer
          ? `buffer<${parameter.type.tach}>`
          : parameter.type.tach
      }`
    ).join(", ")
  })${result}\n\`\`\`\n\n`;
  if (fn.coordinates.length) {
    out += "| Coordinate | Type | Description |\n|---|---|---|\n";
    for (const coordinate of fn.coordinates) {
      out += `| \`${coordinate.name}\` | \`uint32\` | ${
        cell(coordinate.description)
      } |\n`;
    }
    out += "\n";
  }
  if (fn.parameters.length) {
    out += "| Parameter | Context | Type | Description |\n|---|---|---|---|\n";
    for (const parameter of fn.parameters) {
      out += `| \`${parameter.name}\` | ${
        parameter.buffer ? access(parameter.access) : "value input"
      } | \`${
        parameter.buffer
          ? `buffer<${parameter.type.tach}>`
          : parameter.type.tach
      }\` | ${cell(parameter.description)} |\n`;
    }
    out += "\n";
  }
  if (fn.returns) {
    out += `**Returns:** \`${fn.returns.type.tach}\`${
      fn.returns.description ? ` — ${fn.returns.description}` : ""
    }\n\n`;
  }
  return out;
}

function usage(project: Documentation): string {
  const types = project.modules.flatMap((module) =>
    module.kernels.flatMap((kernel) => kernel.types.map((type) => type.name))
  );
  const functions = project.modules.flatMap((module) =>
    module.kernels.flatMap((kernel) =>
      kernel.functions.filter((fn) => fn.exported)
    )
  );
  const imports = [
    ...functions.map((fn) => fn.name),
    ...types.map((name) => `type ${name}`),
  ];
  let out = 'import { tach as $$tach } from "@depths/tach";\n';
  if (imports.length) {
    out += `import {\n${
      imports.map((name) => `  ${name},`).join("\n")
    }\n} from ${JSON.stringify(project.package)};\n`;
  }
  out += "\n";
  for (const fn of functions) {
    out += `export async function run_${fn.name}(\n`;
    for (const parameter of fn.parameters) {
      out += `  ${parameter.name}: ${typeScriptType(parameter.type)},\n`;
    }
    if (fn.coordinates.length) {
      out += `  $size: ${
        fn.coordinates.length === 1
          ? "number"
          : `readonly [${fn.coordinates.map(() => "number").join(", ")}]`
      },\n`;
    }
    out += ") {\n  return $$tach(async ($$gpu) => {\n";
    for (const parameter of fn.parameters) {
      if (parameter.buffer) {
        out +=
          `    const $${parameter.name} = $$gpu.buffer(${parameter.name});\n`;
      }
    }
    const args = fn.parameters.map((parameter) =>
      `${parameter.buffer ? "$" : ""}${parameter.name}`
    );
    if (fn.coordinates.length) args.push("{ size: $size }");
    out += `    await $$gpu.submit(${fn.name}(${args.join(", ")}));\n`;
    const output = fn.parameters.find((parameter) =>
      parameter.buffer && parameter.access !== "read"
    );
    if (output) out += `    return $${output.name}.read();\n`;
    out += "  });\n}\n\n";
  }
  return out;
}

export function typeScriptType(type: TypeRef): string {
  switch (type.kind) {
    case "bool":
      return "boolean";
    case "i32":
    case "u32":
    case "f32":
    case "atomic":
      return "number";
    case "struct":
      return type.name!;
    case "vector":
      return `readonly [${Array(type.lanes).fill("number").join(", ")}]`;
    case "fixedArray":
      return `readonly ${typeScriptType(type.elem!)}[]`;
    case "runtimeArray":
      return typedArray(type.elem!) ??
        `readonly ${typeScriptType(type.elem!)}[]`;
    default:
      return "never";
  }
}

function typedArray(type: TypeRef): string | undefined {
  const element = type.kind === "atomic"
    ? type.elem!
    : type.kind === "vector" && type.lanes !== 3
    ? type.elem!
    : type;
  if (type.kind === "vector" && type.lanes === 3) return undefined;
  const name = element.kind === "i32"
    ? "Int32Array"
    : element.kind === "u32"
    ? "Uint32Array"
    : element.kind === "f32"
    ? "Float32Array"
    : undefined;
  if (!name) return undefined;
  return type.kind === "vector"
    ? `${name} | ReadonlyArray<${typeScriptType(type)}>`
    : `${name} | readonly ${typeScriptType(type)}[]`;
}

function paragraph(value?: string): string {
  return value ? `${value}\n\n` : "";
}
function cell(value?: string): string {
  return value ? value.replaceAll("|", "\\|").replaceAll("\n", "<br>") : "—";
}
function access(value?: Parameter["access"]): string {
  return value === "atomic"
    ? "GPU buffer · atomic"
    : value === "readWrite"
    ? "GPU buffer · read/write"
    : value === "write"
    ? "GPU buffer · output"
    : "GPU buffer · read-only";
}
