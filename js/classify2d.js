// const R = require("./ramda.min.js");
// const math = require("./math.min.js");

// Make empty data structures
var data_all = [];
var labels_all = [];
var num_points_all = 0;

// Plotting settings
var density = 10.0; // density of drawing in the heatmap, can be changed
var density_quadrature = 10.0; // density of drawing in the heatmap, can be changed
var num_grid = 400; // Hardcoded in index.html, should be the same
var num_discr = math.ceil(num_grid / density_quadrature);
var dotSize = 8.0;
var extent_plot = 4;
var extent_plot_posterior = 1;
var ss_plot = num_grid / (extent_plot * 2); // scale for drawing
var ss_plot_posterior = num_grid / (extent_plot_posterior * 2); // scale for drawing

var weights_gibbs_nonpriv = fillArray(0.0, 3);
var weights_gibbs_priv = fillArray(0.0, 3);
gibbs_index = 0;

var num_locs = Math.floor(num_grid / density);
var num_locs_quadrature = Math.floor(num_grid / density_quadrature);
num_updates = 0;
var grid_xlocs = fillArray(0.0, num_locs*num_locs);
var grid_ylocs = fillArray(0.0, num_locs*num_locs);
var grid_label = fillArray(fillArray(0.0, num_locs*num_locs), 4);

var quad_xlocs = fillArray(0.0, num_locs_quadrature*num_locs_quadrature);
var quad_ylocs = fillArray(0.0, num_locs_quadrature*num_locs_quadrature);
var quad_value = fillArray(0.0, num_locs_quadrature*num_locs_quadrature);
var quad_samp = fillArray(0.0, num_locs_quadrature*num_locs_quadrature);
var quad_value_priv = fillArray(0.0, num_locs_quadrature*num_locs_quadrature);
var quad_samp_priv = fillArray(0.0, num_locs_quadrature*num_locs_quadrature);
z_ranges = fillArray(0.0, 4);

var convex_background = 0.02;
var convex_accept= 0.01;

var num_samples_taken = 0;
var num_recent_samples = 5;
var posterior_samples = fillArray(fillArray([0.0, 0.0], num_recent_samples), 4);
acceptance_rates = fillArray(1.0, 4);

var sigma_likelihood = 0.707;  // math.sqrt(0.5) to simplify math;
var sigma_prior = 0.8;
targetEpsilon = 1.0;

function myinit() {}

function setEpsilon() {
  // There's a factor two difference between
  targetEpsilon = parseFloat(document.getElementById('epsilonInput').value);
  console.log('targetEpsilon set to:', targetEpsilon);

  // Then recalculate quadrature
  calcQuadratureOnce();
}

function sigmoid_activation(x) {
  return tf.div(1.0, tf.add(1.0, tf.exp(tf.neg(x))));}

// function sigmoid_activation_prime(x) {
//   return tf.mul(sigmoid_activation(x), tf.sub(1.0, sigmoid_activation(x)));}

function calculate_log_posterior__vectorized(data_tf, label_tf, weights, private = true) {
  // all arguments should be tensorflow tensors already

  // data_tf in shape [num_points_all, 3]
  // label_tf in shape [num_points_all, 1]
  // weights in shape [num_locs_quadrature*num_locs_quadrature, 3]

  // diff_vec in shape [num_points_all, num_locs_quadrature*num_locs_quadrature]
  var diff_vec = tf.sub(label_tf, tf.matMul(data_tf, tf.transpose(weights)));
  // pointwise operations
  var error_vec = tf.add(tf.mul(-1.0 / (2.0 * sigma_likelihood ** 2), tf.square(diff_vec)), 2.0);

  if (private) {
    var loglik = tf.sum(tf.mul(targetEpsilon/2.0, sigmoid_activation(error_vec)), axis=0);
  } else {
    var loglik = tf.sum(error_vec, axis=0);
  }

  var log_prior = tf.mul(-1.0 / (2.0 * sigma_prior ** 2), tf.sum(tf.square(weights), axis=1));
  return tf.add(loglik, log_prior);}

function random_data() {
  var data = [];
  var labels = [];
  for (var k = 0; k < 40; k++) {
    data.push([convnetjs.randf(-3, 3), convnetjs.randf(-3, 3)]);
    labels.push(convnetjs.randf(0, 1) > 0.5 ? 1 : -1);
  }

  // Put data and labels in federated objects
  data_all = R.clone(data);
  labels_all = R.clone(labels);
  num_points_all = labels.length;

  calcQuadratureOnce();
}

function choleskyDecomposition(matrix) {
  // Argument "matrix" can be either math.matrix or standard 2D array
  const A = math.matrix(matrix);
  // Matrix A must be symmetric
  console.assert(math.deepEqual(A, math.transpose(A)));

  const n = A.size()[0];
  // Prepare 2D array with 0
  const L = new Array(n).fill(0).map((_) => new Array(n).fill(0));

  math.range(0, n).forEach((i) => {
    math.range(0, i + 1).forEach((k) => {
      var s = 0;
      for (var j = 0; j < k; j++) {
        s += L[i][j] * L[k][j];
      }
      L[i][k] =
        i === k
          ? math.sqrt(A.get([k, k]) - s)
          : (1 / L[k][k]) * (A.get([i, k]) - s);
    });
  });
  return L;
}

function simple_data() {
  var data = [];
  var labels = [];
  data.push([-1.4326, 1.1909]);
  labels.push(1);
  // data.push([1.1253, -0.376]);
  // labels.push(1);
  // data.push([1.2877, 0.03273]);
  // labels.push(1);
  data.push([-1.1465, 0.1746]);
  labels.push(1);
  data.push([0.9, 2.8]);
  labels.push(1);
  data.push([0.3, 3.2]);
  labels.push(1);
  // data.push([2.0, -0.2]);
  // labels.push(1);
  data.push([3.0, 3.0]);
  labels.push(-1);
  data.push([1.3133, 1.0139]);
  labels.push(-1);
  data.push([1.7258, 1.668]);
  labels.push(-1);
  data.push([2.1117, 0.8893]);
  labels.push(-1);
  data.push([2.632, 1.3544]);
  labels.push(-1);
  data.push([2.2636, -1.8677]);
  labels.push(-1);

  // Put data and labels in federated objects
  data_all = R.clone(data);
  labels_all = R.clone(labels);
  num_points_all = labels.length;

  // Update quadrature
  // calcQuadratureOnce();
}

function separable_data() {
  var data = [];
  var labels = [];
  data.push([-1.4326, 1.1909]);
  labels.push(1);

  data.push([-2.1465, 0.1746]);
  labels.push(1);

  data.push([-1.9, 2.8]);
  labels.push(1);

  data.push([-2.05, 2.3]);
  labels.push(1);

  data.push([-2.17, 2.6]);
  labels.push(1);

  data.push([-2.3, 3.2]);
  labels.push(1);

  data.push([3.0, 3.0]);
  labels.push(-1);

  data.push([1.3133, -1.0139]);
  labels.push(-1);

  data.push([1.7258, 1.668]);
  labels.push(-1);

  data.push([2.1117, 0.8893]);
  labels.push(-1);

  data.push([2.632, 1.3544]);
  labels.push(-1);

  data.push([2.2636, -1.8677]);
  labels.push(-1);

  data.push([3.2636, -3.3677]);
  labels.push(-1);

  data.push([3.3636, -2.8]);
  labels.push(-1);

  // Put data and labels in federated objects
  data_all = R.clone(data);
  labels_all = R.clone(labels);
  num_points_all = labels.length;

  // Update quadrature
  calcQuadratureOnce();
}

function big_data() {
  var data = [];
  var labels = [];
  data.push([-1.4326, 1.1909]);
  labels.push(1);
  data.push([-1.4326, -1.1909]);
  labels.push(1);

  data.push([-2.1465, -0.1746]);
  labels.push(1);
  data.push([-2.1465, -0.3746]);
  labels.push(1);

  data.push([-1.9, 2.8]);
  labels.push(1);
  data.push([-1.9, -2.7]);
  labels.push(1);

  data.push([-2.05, 2.3]);
  labels.push(1);
  data.push([-2.05, -2.2]);
  labels.push(1);

  data.push([-2.17, 2.6]);
  labels.push(1);
  data.push([-2.17, -2.5]);
  labels.push(1);

  data.push([-2.3, 3.2]);
  labels.push(1);
  data.push([-2.3, -3.2]);
  labels.push(1);

  data.push([-3.0, 3.0]);
  labels.push(1);
  data.push([-2.5, 2.5]);
  labels.push(1);

  data.push([-0.8, -3.1]);
  labels.push(1);
  data.push([-.8, 2.6]);
  labels.push(1);

  data.push([-3.5, -0.8]);
  labels.push(1);
  data.push([-3.4, 0.7]);
  labels.push(1);

  data.push([-3.7, -3.3]);
  labels.push(1);
  data.push([-3.4, 3.7]);
  labels.push(1);

  data.push([1.1, -2.0139]);
  labels.push(-1);
  data.push([1.0, 2.9139]);
  labels.push(-1);

  data.push([2.1, -2.0139]);
  labels.push(-1);
  data.push([2.0, -2.9139]);
  labels.push(-1);

  data.push([1.3133, -1.0139]);
  labels.push(-1);
  data.push([2.3133, -0.9139]);
  labels.push(-1);

  data.push([1.7258, 1.668]);
  labels.push(-1);
  data.push([1.9258, 1.468]);
  labels.push(-1);

  data.push([2.1117, 0.8893]);
  labels.push(-1);
  data.push([2.3117, 0.9893]);
  labels.push(-1);

  data.push([2.632, 1.3544]);
  labels.push(-1);
  data.push([2.732, -1.4544]);
  labels.push(-1);

  data.push([2.2636, -1.8677]);
  labels.push(-1);
  data.push([2.1636, -1.5677]);
  labels.push(-1);

  data.push([3.2636, -3.3677]);
  labels.push(-1);
  data.push([3.3636, -2.3677]);
  labels.push(-1);

  data.push([3.3636, -2.8]);
  labels.push(-1);
  data.push([2.3636, -1.8]);
  labels.push(-1);

  data.push([3.1636, 3.6]);
  labels.push(-1);
  data.push([3.3636, 3.1]);
  labels.push(-1);
  data.push([2.0, 3.1]);
  labels.push(-1);

  // Put data and labels in federated objects
  data_all = R.clone(data);
  labels_all = R.clone(labels);
  num_points_all = labels.length;

  // Update quadrature
  calcQuadratureOnce();
}

function fillArray(value, len) {
  var arr = [];
  for (var i = 0; i < len; i++) {
    arr.push(R.clone(value));
  }
  return arr;
}

function RegressionInit() {}

function invert_matrix(matrix_input) {
  var det =
    matrix_input[0][0] * matrix_input[1][1] -
    matrix_input[0][1] * matrix_input[1][0];
  determinant = Math.abs(det);

  var inverse = [
    [matrix_input[1][1] / det, -matrix_input[0][1] / det],
    [-matrix_input[1][0] / det, matrix_input[0][0] / det],
  ];
  return [inverse, determinant];
}

function invert_matrices(cov_matrices) {
  var inverses = [];
  var determinants = [];

  cov_matrices.forEach((cov_matrix) => {
    [inverse, det] = invert_matrix(cov_matrix);
    determinants.push(det);
    inverses.push(inverse);
  });
  return [determinants, inverses];
}

function drawAxisValues(context, value, printw = false) {
  var margin = 15;
  context.fillStyle = "rgb(0,0,0)";
  context.font = "15px sans-serif";
  context.fillText(-value, 0, HEIGHT / 2 + margin);
  context.fillText(value, WIDTH - margin, HEIGHT / 2 + margin);
  // context.fillText(value, WIDTH / 2, 0 + margin);
  context.fillText(-value, WIDTH / 2, HEIGHT);
  context.fillText(0, WIDTH / 2 - margin, HEIGHT / 2 + margin);

  // draw axis lines
  context.beginPath();
  context.strokeStyle = "rgb(50,50,50)";
  context.lineWidth = 1;
  context.moveTo(0, HEIGHT / 2);
  context.lineTo(WIDTH, HEIGHT / 2);
  context.moveTo(WIDTH / 2, 0);
  context.lineTo(WIDTH / 2, HEIGHT);
  context.stroke();

  if (printw) {
    // Prints w0 and w1 on the axes
    context.fillText("w0", WIDTH - (margin*2.0), HEIGHT / 2 - 0.3*margin);
    context.fillText("w1", WIDTH / 2, margin);
  } else {
    // Prints x and y on the axes
    context.fillText("x", WIDTH - (margin*2.), HEIGHT / 2 - 0.3*margin);
    context.fillText("y", WIDTH / 2, margin);
  }
}

function draw() {
  // Clear and plot axis values for scatters
  contextsScatters.forEach((context) => {
    // Draw axis values
    context.clearRect(0, 0, WIDTH, HEIGHT);
    context.fillStyle = "rgb(255, 255, 224)"; // Light yellow background
    context.fillRect(0, 0, context.canvas.width, context.canvas.height);
    drawAxisValues(context, extent_plot, false);
  });

  contextsDecisions.forEach((context, pane_index) => {
    context.clearRect(0, 0, WIDTH, HEIGHT);
    // Draw the background color to indicate mean decision
    for (var numg = 0; numg < num_grid*num_grid; numg++) {
      var x = grid_xlocs[numg];
      var y = grid_ylocs[numg];
      var value = grid_label[pane_index][numg];

      var red_value = Math.floor(250 - 100 * value);
      var green_value = Math.floor(150 + 100 * value);

      context.fillStyle = "rgb(" + red_value + ", " + green_value + ", 150)";

      context.fillRect(
        x, // - density / 2 - 1,
        y, // - density / 2 - 1,
        density, // + 2,
        density, // + 2
      );
    }
    drawAxisValues(context, extent_plot, false);
  });


  // draw scatter datapoints.
  contextsScatters.forEach((context) => {
    context.strokeStyle = "rgb(0,0,0)";
    context.lineWidth = 1;
    for (var i = 0; i < num_points_all; i++) {
      if (labels_all[i] == 1) fillstyle = "rgb(100,200,100)";
      else fillstyle = "rgb(200,100,100)";
      fillstyles = ["rgb(100,200,100)", "rgb(200,100,100)", "rgb(100,100,200)"];

      // Draw points for each of the clients
      drawCircle(
        context,
        WIDTH / 2 + data_all[i][0] * ss_plot,
        HEIGHT / 2 - data_all[i][1] * ss_plot,
        dotSize,
        fillstyle
      );
    }
  });
}

function drawSingleSample(context, pane_index) {
  var w0, w1, w2;
  var int_accept = 0;

  gibbs_index = (gibbs_index + 1) % 3;

  if (num_updates % 1000 == 0) {
    weights_gibbs_priv = [-0.4, 0.05, 0.0];
    weights_gibbs_nonpriv = [-0.4, 0.05, 0.0];
  }

  w2 = 0.0;
  if (pane_index == 0) {
    // Draw non-private, quadrature sample

    var tnum = Math.random();
    for (var idx = 0; idx < num_locs_quadrature**2; idx++) {
      if (quad_samp[idx] > tnum) {
        w0 = (quad_xlocs[idx] - WIDTH / 2) / ss_plot_posterior;
        w1 = (HEIGHT / 2 - quad_ylocs[idx]) / ss_plot_posterior;
        int_accept = 1;
        break
      }
    }

    num_updates += 1;
  } else if (pane_index == 1) {
    // Gibbs sampling
    var label_tf = tf.tensor2d(labels_all, (shape = [num_points_all, 1]));
    var data_tf = tf.concat(
      [tf.tensor2d(data_all), tf.ones([num_points_all, 1])],
      (axis = 1));

    var list_weights = [
      tf.mul(tf.ones([num_discr]), weights_gibbs_nonpriv[0]),
      tf.mul(tf.ones([num_discr]), weights_gibbs_nonpriv[1]),
      tf.mul(tf.ones([num_discr]), weights_gibbs_nonpriv[2]),
    ]
    var weight_range = tf.linspace(-extent_plot_posterior, extent_plot_posterior, num_discr);
    list_weights[gibbs_index] = weight_range;
    var weights_gibbs = tf.stack(list_weights, axis=1);

    log_prob = calculate_log_posterior__vectorized(data_tf, label_tf, weights_gibbs, false);
    idx = tf.multinomial(log_prob, 1).dataSync()[0];
    weights_gibbs_nonpriv = tf.transpose(tf.slice(weights_gibbs, [idx, 0], [1, 3])).dataSync();
    w0 = weights_gibbs_nonpriv[0];
    w1 = weights_gibbs_nonpriv[1];
    w2 = weights_gibbs_nonpriv[2];
    int_accept = 1;

  } else if (pane_index == 2) {
    // Gibbs sampling
    var label_tf = tf.tensor2d(labels_all, (shape = [num_points_all, 1]));
    var data_tf = tf.concat(
      [tf.tensor2d(data_all), tf.ones([num_points_all, 1])],
      (axis = 1));

    var list_weights = [
      tf.mul(tf.ones([num_discr]), weights_gibbs_priv[0]),
      tf.mul(tf.ones([num_discr]), weights_gibbs_priv[1]),
      tf.mul(tf.ones([num_discr]), weights_gibbs_priv[2]),
    ]
    var weight_range = tf.linspace(-extent_plot_posterior, extent_plot_posterior, num_discr);
    list_weights[gibbs_index] = weight_range;
    var weights_gibbs = tf.stack(list_weights, axis=1);

    log_prob = calculate_log_posterior__vectorized(data_tf, label_tf, weights_gibbs, true);
    idx = tf.multinomial(log_prob, 1).dataSync()[0];
    weights_gibbs_priv = tf.transpose(tf.slice(weights_gibbs, [idx, 0], [1, 3])).dataSync();
    w0 = weights_gibbs_priv[0];
    w1 = weights_gibbs_priv[1];
    w2 = weights_gibbs_priv[2];
    int_accept = 1;

  } else if (pane_index == 3) {
    // Sample from quadrature
    var tnum = Math.random();
    for (var idx = 0; idx < num_locs_quadrature**2; idx++) {
      if (quad_samp_priv[idx] > tnum) {
        w0 = (quad_xlocs[idx] - WIDTH / 2) / ss_plot_posterior;
        w1 = (HEIGHT / 2 - quad_ylocs[idx]) / ss_plot_posterior;
        int_accept = 1.
        break
      }
    }
  }
  acceptance_rates[pane_index] = acceptance_rates[pane_index] * (1. - convex_accept) + convex_accept * int_accept;

  if (num_updates % 2 == 0) {
    posterior_samples[pane_index][num_samples_taken % num_recent_samples][0] = w0;
    posterior_samples[pane_index][num_samples_taken % num_recent_samples][1] = w1;

    if (pane_index == 3) {
      num_samples_taken += 1;
    }
  }

  var x_start = 0;
  var y_start =
    HEIGHT / 2 +
    ((-extent_plot * w0 + w2) / w1) * ss_plot;

  var x_end = WIDTH;
  var y_end =
    HEIGHT / 2 +
    ((extent_plot * w0 + w2) / w1) * ss_plot;

  var xclose = -w0 * w2 / (w0**2 + w1**2);
  xclose = WIDTH / 2 + xclose * ss_plot
  var yclose = -w1 * w2 / (w0**2 + w1**2);
  yclose = HEIGHT / 2 - yclose * ss_plot

  context.beginPath();
  context.strokeStyle = "rgb(50,50,250)";
  context.lineWidth = 5;
  context.moveTo(x_start, y_start);
  context.lineTo(x_end, y_end);
  context.stroke();
  context.beginPath();
  context.lineWidth = 1;
  context.moveTo(xclose, yclose);
  context.lineTo(xclose + w0*100.0, yclose-w1*100.0);
  context.stroke();

  drawCircle(
    context,
    xclose,
    yclose,
    2,
    "rgb(50,50,250)"
  );

  // Update running mean of the decisions
  for (var xint = 0; xint < num_locs; xint += 1) {
    for (var yint = 0; yint < num_locs; yint += 1) {
      var x_pixel = xint*density;
      var y_pixel = yint*density;

      var xt = (x_pixel - WIDTH / 2) / ss_plot;
      var yt = (HEIGHT / 2 - y_pixel) / ss_plot;

      var prediction_positive = w0 * xt + w1 * yt + w2 > 0;

      var index = xint + yint * num_locs;
      var current_mean = grid_label[pane_index][index];

      grid_xlocs[index] = x_pixel;
      grid_ylocs[index] = y_pixel;
      grid_label[pane_index][index] = current_mean * (1 - convex_background) + prediction_positive * convex_background;
    }
  }
}

function drawRegressionLines() {
  contextsScatters.forEach((ctx, pane_index) => {
    drawSingleSample(ctx, pane_index);
  });
}

function drawPosteriorSamples() {
  contextsPosteriors.forEach((context, pane_index) => {
    // clear canvas
    context.clearRect(0, 0, WIDTH, HEIGHT);

    // Background shade with quadrature
    for (var idx = 0; idx < num_locs_quadrature**2; idx++) {
      var x = quad_xlocs[idx];
      var y = quad_ylocs[idx];
      if (pane_index < 2) {
        var value = quad_value[idx];
      } else {
        var value = quad_value_priv[idx];
      }

      // Shade colors accordingly
      var red_value = Math.floor(251 - 100 * value);
      var green_value = Math.floor(251 - 100 * value);
      context.fillStyle = "rgb(" + red_value + ", " + green_value + ", 250)";
      context.fillRect(
        x, // x_start
        y, // y_start
        density_quadrature,  // width in x direction
        density_quadrature  // height in y direction
      );
    }

    // Draw axis values
    drawAxisValues(context, extent_plot_posterior, true);

    if (pane_index <= 3) {

      num_samples_end = Math.min(num_samples_taken, num_recent_samples);
      for (var num_sample = 0; num_sample < num_samples_end; num_sample++) {
        var w0 = posterior_samples[pane_index][num_sample][0];
        var w1 = posterior_samples[pane_index][num_sample][1];

        // Plot coordinate as scatter Plot
        var x = WIDTH / 2 + w0 * ss_plot_posterior;
        var y = HEIGHT / 2 - w1 * ss_plot_posterior;

        context.fillStyle = "rgb(0,0,0)";
        context.beginPath();
        context.arc(x, y, 3, 0, 2 * Math.PI, true);
        context.closePath();
        context.fill();
      }
    };
  });
}

function calcQuadratureOnce() {
  resetRunningMean();

  var param_matrix = [];
  var index = 0;
  for (var xint = 0; xint < num_locs_quadrature; xint += 1) {
    for (var yint = 0; yint < num_locs_quadrature; yint += 1) {
      var x_pixel = xint * density_quadrature;
      var y_pixel = yint * density_quadrature;

      var xt = (x_pixel - WIDTH / 2) / ss_plot_posterior;
      var yt = (HEIGHT / 2 - y_pixel) / ss_plot_posterior;

      index += 1;

      quad_xlocs[index] = x_pixel;
      quad_ylocs[index] = y_pixel;

      param_matrix.push([xt, yt, 0.0]);
    }
  }

  param_tf = tf.tensor2d(param_matrix, (shape = [num_locs_quadrature*num_locs_quadrature, 3]));
  var label_tf = tf.tensor2d(labels_all, (shape = [num_points_all, 1]));
  var data_tf = tf.concat(
    [tf.tensor2d(data_all), tf.ones([num_points_all, 1])],
    (axis = 1));

  var log_density_np = calculate_log_posterior__vectorized(data_tf, label_tf, param_tf, false);
  var log_density_priv = calculate_log_posterior__vectorized(data_tf, label_tf, param_tf, true);

  quad_samp = tf.cumsum(tf.softmax(log_density_np)).dataSync();
  quad_samp_priv = tf.cumsum(tf.softmax(log_density_priv)).dataSync();

  z_ranges[0] = tf.min(log_density_np).dataSync()[0];
  z_ranges[1] = tf.max(log_density_np).dataSync()[0];
  z_ranges[2] = tf.min(log_density_priv).dataSync()[0];
  z_ranges[3] = tf.max(log_density_priv).dataSync()[0];

  var quad_value_pre = tf.softmax(log_density_np);
  quad_value = (tf.div(quad_value_pre, tf.max(quad_value_pre))).dataSync();
  var quad_value_priv_pre = tf.softmax(log_density_priv);
  quad_value_priv = (tf.div(quad_value_priv_pre, tf.max(quad_value_priv_pre))).dataSync();
}

function resetRunningMean() {
  num_updates = 0;

  quad_value = fillArray(0.0, num_locs_quadrature*num_locs_quadrature);
  quad_value_priv = fillArray(0.0, num_locs_quadrature*num_locs_quadrature);

  weights_gibbs_priv = [-0.4, 0.05, 0.0];
}

function mouseClick(x, y, shiftPressed, ctrlPressed) {
  // x and y transformed to data space coordinates
  var xt = (x - WIDTH / 2) / ss_plot;
  var yt = (HEIGHT / 2 - y) / ss_plot;

  if (ctrlPressed) {
    // remove closest data point
    var index_min = -1;
    var dist_min = 99999;
    for (var k = 0, n = data_all.length; k < n; k++) {
      var dx = data_all[k][0] - xt;
      var dy = data_all[k][1] - yt;
      var d = dx * dx + dy * dy;
      if (d < dist_min || k == 0) {
        dist_min = d;
        index_min = k;
      }
    }
    if (index_min >= 0) {
      console.log("splicing " + index_min);
      data_all.splice(index_min, 1);
      labels_all.splice(index_min, 1);
      num_points_all -= 1;
    }
  } else {
    // add datapoint at location of click
    data_all.push([xt, yt]);
    labels_all.push(shiftPressed ? 1 : -1);
    num_points_all += 1;
  }

  // Reset running mean
  calcQuadratureOnce();
}

function keyDown(key) {}

function keyUp(key) {}

$(function () {
  // note, globals
  RegressionInit();

  separable_data();

  NPGinit(8);
});

module.exports = {
  invert_matrices: invert_matrices,
  choleskyDecomposition: choleskyDecomposition,
};
