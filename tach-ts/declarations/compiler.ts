import type { Documentation } from "../src/docs.ts";
export type {
  Documentation,
  DocumentedType,
  Field,
  FunctionDoc,
  KernelDoc,
  ModuleDoc,
  Parameter,
  TypeRef,
} from "../src/docs.ts";
export interface CompilerRunOptions {
  readonly cwd?: string;
  readonly env?: Readonly<Record<string, string>>;
}
export interface BuildOptions extends CompilerRunOptions {
  readonly verbose?: boolean;
}
export interface ProjectResult {
  readonly root: string;
  readonly description: Documentation;
}
export declare function packageVersion(): Promise<string>;
export declare function compilerPath(): Promise<string>;
export declare function build(options?: BuildOptions): Promise<ProjectResult>;
export declare function check(
  options?: CompilerRunOptions,
): Promise<ProjectResult>;
export declare function docs(
  options?: CompilerRunOptions,
): Promise<ProjectResult>;
export declare function format(options?: CompilerRunOptions): Promise<void>;
