function fail(message) {
  throw new Error(message);
}
function show(value) {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function same(actual, expected) {
  if (Object.is(actual, expected)) return true;
  if (ArrayBuffer.isView(actual) || ArrayBuffer.isView(expected)) {
    return ArrayBuffer.isView(actual) && ArrayBuffer.isView(expected) &&
      actual.constructor === expected.constructor &&
      same(Array.from(actual), Array.from(expected));
  }
  if (
    !actual || !expected || typeof actual !== "object" ||
    typeof expected !== "object"
  ) return false;
  const left = Object.keys(actual), right = Object.keys(expected);
  return left.length === right.length &&
    left.every((key, index) =>
      key === right[index] && same(actual[key], expected[key])
    );
}

function validate(error, expected) {
  if (expected instanceof RegExp) {
    if (!expected.test(String(error?.message ?? error))) {
      fail(`expected ${show(error)} to match ${expected}`);
    }
  } else if (typeof expected === "function" && expected(error) !== true) {
    fail(`error validator rejected ${show(error)}`);
  }
}

export const assert = {
  deepEqual(actual, expected, message) {
    if (!same(actual, expected)) {
      fail(message ?? `${show(actual)} != ${show(expected)}`);
    }
  },
  doesNotMatch(actual, expected) {
    if (expected.test(actual)) {
      fail(`${show(actual)} unexpectedly matches ${expected}`);
    }
  },
  equal(actual, expected, message) {
    if (!Object.is(actual, expected)) {
      fail(message ?? `${show(actual)} != ${show(expected)}`);
    }
  },
  match(actual, expected) {
    if (!expected.test(actual)) {
      fail(`${show(actual)} does not match ${expected}`);
    }
  },
  notEqual(actual, expected, message) {
    if (Object.is(actual, expected)) {
      fail(message ?? `${show(actual)} == ${show(expected)}`);
    }
  },
  ok(value, message) {
    if (!value) {
      fail(message ?? `expected truthy value, received ${show(value)}`);
    }
  },
  async rejects(operation, expected) {
    try {
      await (typeof operation === "function" ? operation() : operation);
    } catch (error) {
      validate(error, expected);
      return error;
    }
    fail("expected rejection");
  },
  throws(operation, expected) {
    try {
      operation();
    } catch (error) {
      validate(error, expected);
      return error;
    }
    fail("expected exception");
  },
};
