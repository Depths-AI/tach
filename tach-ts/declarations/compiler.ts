import type { Documentation } from "../src/docs.ts";
import { TachError } from "@depths/tach";
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
export interface DiagnosticPosition {
  readonly offset: number;
  readonly line: number;
  readonly column: number;
}
export interface DiagnosticSpan {
  readonly file: string;
  readonly start: DiagnosticPosition;
  readonly end: DiagnosticPosition;
}
export interface RelatedDiagnostic {
  readonly span: DiagnosticSpan;
  readonly message: string;
  readonly source?: string;
}
export interface Diagnostic {
  readonly severity: "error" | "warning";
  readonly code: string;
  readonly span: DiagnosticSpan;
  readonly message: string;
  readonly help?: string;
  readonly source?: string;
  readonly related?: readonly RelatedDiagnostic[];
}
export declare class CompilerError extends TachError {
  readonly diagnostics: readonly Diagnostic[];
  constructor(diagnostics: readonly Diagnostic[], cause: unknown);
}
export interface ProjectResult {
  readonly root: string;
  readonly description: Documentation;
  readonly diagnostics: readonly Diagnostic[];
}
export declare function renderDiagnostics(
  diagnostics: readonly Diagnostic[],
): string;
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
