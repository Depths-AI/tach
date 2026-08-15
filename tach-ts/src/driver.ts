import type { TachAdapterInfo } from "./api.ts";

export type DriverBuffer = object | number;

export interface HostLayoutField {
  readonly name: string;
  readonly offset: number;
  readonly type: HostLayout;
}
export interface HostLayout {
  readonly kind:
    | "bool"
    | "i32"
    | "u32"
    | "f32"
    | "vector"
    | "array"
    | "runtime"
    | "struct";
  readonly size?: number;
  readonly stride?: number;
  readonly count?: number;
  readonly runtime?: boolean;
  readonly elem?: HostLayout;
  readonly fields?: readonly HostLayoutField[];
}

export interface ResourceDefinition {
  readonly name: string;
  readonly type: string;
  readonly byteSize?: number;
  readonly minimumByteSize: number;
  readonly runtime: boolean;
  readonly layout: HostLayout;
}

export interface PublicParameterDefinition {
  readonly name: string;
  readonly kind: "buffer" | "value";
  readonly type: string;
  readonly resource?: number;
}
export interface ParameterFieldDefinition {
  readonly type: string;
  readonly byteOffset: number;
  readonly layout: HostLayout;
}
export interface ParameterBlockDefinition {
  readonly group: 0;
  readonly binding: number;
  readonly byteSize: number;
  readonly fields: readonly ParameterFieldDefinition[];
}
export interface KernelDefinition {
  readonly entryPoint: string;
  readonly workgroupSize: readonly [number, number, number];
  readonly bindings: readonly {
    readonly group: 0;
    readonly binding: number;
    readonly access: "read" | "read_write";
    readonly type: string;
    readonly minimumByteSize: number;
  }[];
  readonly parameterBlock?: ParameterBlockDefinition;
}
export interface ShapeExpression {
  readonly op:
    | "constant"
    | "parameter"
    | "resourceLength"
    | "launchAxis"
    | "add"
    | "sub"
    | "mul"
    | "div"
    | "rem"
    | "min"
    | "max"
    | "ceilDiv";
  readonly value?: number;
  readonly parameter?: number;
  readonly resource?: number;
  readonly path?: readonly string[];
  readonly axis?: 0 | 1 | 2;
  readonly left?: ShapeExpression;
  readonly right?: ShapeExpression;
}
export interface ValueSource {
  readonly kind:
    | "parameter"
    | "bool"
    | "i32"
    | "u32"
    | "f32Bits"
    | "shape"
    | "repeat";
  readonly parameter?: number;
  readonly path?: readonly string[];
  readonly value?: number | boolean;
  readonly expression?: ShapeExpression;
}
export interface ResourceSource {
  readonly binding?: number;
  readonly kind: "external" | "transient";
  readonly resource: number;
}
export interface StepDefinition {
  readonly kind: "dispatch" | "barrier";
  readonly kernel?: number;
  readonly domain?: readonly ShapeExpression[];
  readonly resources: readonly ResourceSource[];
  readonly parameters?: readonly ValueSource[];
}
export interface TransientDefinition {
  readonly type: string;
  readonly stride: number;
  readonly alignment: number;
  readonly minimumByteSize: number;
  readonly length: ShapeExpression;
  readonly color: number;
  readonly firstStep: number;
  readonly lastStep: number;
}
export interface ProgramPlanDefinition {
  readonly program: number;
  readonly transients: readonly TransientDefinition[];
  readonly steps: readonly StepDefinition[];
  readonly repeatBarrier?: StepDefinition;
  readonly repeat: "program" | "invocation-loop";
}
export interface PublicProgramDefinition {
  readonly name: string;
  readonly parameters: readonly PublicParameterDefinition[];
  readonly resources: readonly ResourceDefinition[];
  readonly launch?: {
    readonly dimensions: 1 | 2 | 3;
    readonly inferFromResource?: number;
  };
}
export interface TargetDefinition {
  readonly vulkan?: string;
  readonly spirv?: string;
  readonly features?: readonly string[];
  readonly kernels: readonly KernelDefinition[];
  readonly programs: readonly ProgramPlanDefinition[];
}
export interface ModuleDefinition {
  readonly schema: 1;
  readonly types: readonly unknown[];
  readonly programs: readonly PublicProgramDefinition[];
  readonly shaders: { readonly web: URL; readonly spirv: URL };
  readonly targets: {
    readonly web: TargetDefinition;
    readonly spirv: TargetDefinition;
  };
}

export interface PreparedResource {
  readonly binding: number;
  readonly buffer?: DriverBuffer;
  readonly scratch?: number;
  readonly byteSize: number;
}
export interface PreparedBarrierResource {
  readonly buffer?: DriverBuffer;
  readonly scratch?: number;
}
export type PreparedStep = {
  readonly kind: "dispatch";
  readonly kernel: number;
  readonly groups: readonly [number, number, number];
  readonly resources: readonly PreparedResource[];
  readonly parameters?: Uint8Array;
} | {
  readonly kind: "barrier";
  readonly resources: readonly PreparedBarrierResource[];
};
export interface PreparedCommand {
  readonly module: ModuleDefinition;
  readonly shader: URL;
  readonly target: TargetDefinition;
  readonly steps: readonly PreparedStep[];
  readonly repeat: number;
  readonly repeatBarrier?: readonly PreparedBarrierResource[];
  readonly scratch: ReadonlyMap<number, number>;
}

export interface Driver {
  readonly adapter: TachAdapterInfo;
  createBuffer(label: string, bytes: Uint8Array): DriverBuffer;
  writeBuffer(buffer: DriverBuffer, bytes: Uint8Array): void;
  readBuffer(buffer: DriverBuffer, byteLength: number): Promise<Uint8Array>;
  destroyBuffer(buffer: DriverBuffer): void;
  prepare(commands: readonly PreparedCommand[]): Promise<void>;
  submit(commands: readonly PreparedCommand[]): Promise<void>;
  idle(): Promise<void>;
  close(): void;
}
