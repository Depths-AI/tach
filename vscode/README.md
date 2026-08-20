# Tach

Syntax highlighting and editing conventions for
[Tach](https://github.com/Depths-AI/tach), a small typed language for GPU
compute from TypeScript. `.tach` files are the kernels. Ordinary
TypeScript still calls the generated functions.

The extension recognizes `.tach` files and highlights declarations, control
flow, types, structured documentation, built-in functions, literals, comments,
members, and operators. It also configures Tach's `//` comments, paired
delimiters, and two-space indentation. Control keywords such as `break` and
`continue`, and builtins such as inferred `vec` and `fma`, follow the same
grammar-owned highlighting as the rest of the language.

This deliberately small extension performs no compilation or background
analysis. Use the `tach` command from
[`@depths/tach`](https://www.npmjs.com/package/@depths/tach) to format,
check, build, and read the language guide.
