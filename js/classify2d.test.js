/**
 * @jest-environment jsdom
 */

// import "ramda";?

const jquery = require("./jquery-1.8.3.min.js");
const classify2d = require("./classify2d");
const R = require("./ramda.min.js");
const math = require("./math.min.js");
// const { math } = require("@tensorflow/tfjs-core");

test("Test matrix inverse", () => {
  var matrices_input = [
    [
      [1, 0],
      [0, 1],
    ],
    [
      [2, 0],
      [0, 1],
    ],
    [
      [2, 1],
      [0, 1],
    ],
  ];
  var matrices_expected = [
    [
      [1, 0],
      [0, 1],
    ],
    [
      [0.5, 0],
      [0, 1],
    ],
    [
      [0.5, -0.5],
      [0, 1],
    ],
  ];

  var dets_expected = [1, 2, 2];
  [dets, inverses] = classify2d.invert_matrices(matrices_input);

  for (var m = 0; m < matrices_expected.length; m++) {
    expect(dets[m]).toBeCloseTo(dets_expected[m]);
    for (var i = 0; i < 2; i++) {
      for (var j = 0; j < 2; j++) {
        expect(inverses[m][i][j]).toBeCloseTo(matrices_expected[m][i][j]);
      }
    }
  }
});

test("Cholesky decomposision", () => {
  console.log(math.sqrt(0));
  var x = [
    [3, 0, 0.3],
    [0, 3, 0],
    [0.3, 0, 3],
  ];
  L = classify2d.choleskyDecomposition(x);

  L_mat = math.matrix(L);
  result = math.multiply(L_mat, L_mat);

  diff = math.abs(math.sum(math.subtract(result, x)));

  expect(diff).toBeCloseTo(0, (numDigits = 1));
});
