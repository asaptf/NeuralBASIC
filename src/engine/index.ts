/**
 * NeuralBASIC educational neural runtime.
 *
 * Pure TypeScript, fully client-side training & inference designed for
 * Immediate Mode visualization (weights, activations, decision boundaries,
 * attention maps). WebGPU/TF.js is not required; CPU path is deterministic
 * enough for toy curricula and unit tests.
 */

export * from "./types";
export * from "./math";
export * from "./datasets";
export * from "./split";
export * from "./model";
export * from "./train";
export * from "./dsl";
export * from "./export";
