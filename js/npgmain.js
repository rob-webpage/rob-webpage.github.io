//Simple game engine
//Author: Andrej Karpathy
//License: BSD
//This function does all the boring canvas stuff. To use it, just create functions:
//update()          gets called every frame
//draw()            gets called every frame
//myinit()          gets called once in beginning
//mouseClick(x, y)  gets called on mouse click
//keyUp(keycode)    gets called when key is released
//keyDown(keycode)  gets called when key is pushed

// const { update } = require("rambda");

var canvas;
var ctx;
var WIDTH;
var HEIGHT;
var FPS;

function drawBubble(ctx, x, y, w, h, radius) {
  var r = x + w;
  var b = y + h;
  ctx.beginPath();
  ctx.strokeStyle = "black";
  ctx.lineWidth = "2";
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + radius / 2, y - 10);
  ctx.lineTo(x + radius * 2, y);
  ctx.lineTo(r - radius, y);
  ctx.quadraticCurveTo(r, y, r, y + radius);
  ctx.lineTo(r, y + h - radius);
  ctx.quadraticCurveTo(r, b, r - radius, b);
  ctx.lineTo(x + radius, b);
  ctx.quadraticCurveTo(x, b, x, b - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.stroke();
}

function drawRect(ctx, x, y, w, h) {
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

function drawCircle(ctx, x, y, r, fillstyle) {
  if (typeof fillstyle === "undefined") {
    fillstyle = "rgb(0,0,0)";
  }

  ctx.fillStyle = fillstyle;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2, true);
  ctx.closePath();
  ctx.stroke();
  ctx.fill();
}

//uniform distribution integer
function randi(s, e) {
  return Math.floor(Math.random() * (e - s) + s);
}

//uniform distribution
function randf(s, e) {
  return Math.random() * (e - s) + s;
}

//normal distribution random number
function randn(mean, variance) {
  var V1, V2, S;
  do {
    var U1 = Math.random();
    var U2 = Math.random();
    V1 = 2 * U1 - 1;
    V2 = 2 * U2 - 1;
    S = V1 * V1 + V2 * V2;
  } while (S > 1);
  X = Math.sqrt((-2 * Math.log(S)) / S) * V1;
  X = mean + Math.sqrt(variance) * X;
  return X;
}

function eventClick(e, cvs, canvas_id) {
  //get position of cursor relative to top left of canvas
  var x;
  var y;
  if (e.pageX || e.pageY) {
    x = e.pageX;
    y = e.pageY;
  } else {
    x =
      e.clientX +
      document.body.scrollLeft +
      document.documentElement.scrollLeft;
    y =
      e.clientY + document.body.scrollTop + document.documentElement.scrollTop;
  }
  x -= cvs.offsetLeft;
  y -= cvs.offsetTop;

  //call user-defined callback
  mouseClick(x, y, e.shiftKey, e.ctrlKey || e.metaKey);
}

//event codes can be found here:
//http://www.aspdotnetfaq.com/Faq/What-is-the-list-of-KeyCodes-for-JavaScript-KeyDown-KeyPress-and-KeyUp-events.aspx
function eventKeyUp(e) {
  var keycode = "which" in e ? e.which : e.keyCode;
  keyUp(keycode);
}

function eventKeyDown(e) {
  var keycode = "which" in e ? e.which : e.keyCode;
  keyDown(keycode);
}

function NPGinit(FPS) {
  //takes frames per secont to run at
  if (typeof FPS === "undefined") var FPS = 10;

  canvas0 = document.getElementById("NPGcanvas0");
  ctx0 = canvas0.getContext("2d");
  WIDTH = canvas0.width;
  HEIGHT = canvas0.height;
  canvas0.addEventListener(
    "click",
    (e) => {
      eventClick(e, canvas0, 0);
    },
    false
  );

  // Play with second canvas
  canvas1 = document.getElementById("NPGcanvas1");
  ctx1 = canvas1.getContext("2d");
  canvas1.addEventListener(
    "click",
    (e) => {
      eventClick(e, canvas1, 1);
    },
    false
  );

  // Play with second canvas
  canvas2 = document.getElementById("NPGcanvas2");
  ctx2 = canvas2.getContext("2d");
  canvas2.addEventListener(
    "click",
    (e) => {
      eventClick(e, canvas2, 2);
    },
    false
  );

  // Play with third canvas
  canvas3 = document.getElementById("NPGcanvas3");
  ctx3 = canvas3.getContext("2d");
  canvas3.addEventListener(
    "click",
    (e) => {
      eventClick(e, canvas3, 2);
    },
    false
  );

  // Canvases for posterior
  canvasPosterior0 = document.getElementById("NPGCanvasClientPosterior0");
  ctxPosterior0 = canvasPosterior0.getContext("2d");
  canvasPosterior1 = document.getElementById("NPGCanvasClientPosterior1");
  ctxPosterior1 = canvasPosterior1.getContext("2d");
  canvasPosterior2 = document.getElementById("NPGCanvasClientPosterior2");
  ctxPosterior2 = canvasPosterior2.getContext("2d");
  canvasPosterior3 = document.getElementById("NPGCanvasClientPosterior3");
  ctxPosterior3 = canvasPosterior3.getContext("2d");

  // Get a reference to the step counter
  stepCounter = document.getElementById("stepCounter");
  epsilonDisplay = document.getElementById("epsilonDisplay");
  acceptDisp0 = document.getElementById("accept0");
  acceptDisp1 = document.getElementById("accept1");
  acceptDisp2 = document.getElementById("accept2");
  acceptDisp3 = document.getElementById("accept3");

  contextsClients = [ctx0, ctx1, ctx2, ctx3];
  contextsPosteriors = [ctxPosterior0, ctxPosterior1, ctxPosterior2, ctxPosterior3];

  //canvas element cannot get focus by default. Requires to either set
  //tabindex to 1 so that it's focusable, or we need to attach listeners
  //to the document. Here we do the latter
  document.addEventListener("keyup", eventKeyUp, true);
  document.addEventListener("keydown", eventKeyDown, true);

  setInterval(NPGtick, 1000 / FPS);

  calcQuadratureOnce();

  myinit();
}

function updateStepCounter() {
  stepCounter.innerHTML = "Step: " + num_updates + "\n Learning rate: " + (10000*learning_rate).toFixed(2) + "*10^-5";

  acceptDisp0.innerHTML = "Acceptance: " + (100*acceptance_rates[0]).toFixed(1) + " %";
  acceptDisp1.innerHTML = "Acceptance: " + (100*acceptance_rates[1]).toFixed(1) + " %";
  acceptDisp2.innerHTML = "Acceptance: " + (100*acceptance_rates[2]).toFixed(1) + " %";
  acceptDisp3.innerHTML = "Acceptance: " + (100*acceptance_rates[3]).toFixed(1) + " %";

  document.getElementById("zrange0").innerHTML = "log-prob: min: "+z_ranges[0].toFixed(1)+", max: "+z_ranges[1].toFixed(1) + ",   range: "+(z_ranges[1]-z_ranges[0]).toFixed(1) + " bits";
  document.getElementById("zrange1").innerHTML = "log-prob: min: "+z_ranges[0].toFixed(1)+", max: "+z_ranges[1].toFixed(1) + ",   range: "+(z_ranges[1]-z_ranges[0]).toFixed(1) + " bits";
  document.getElementById("zrange2").innerHTML = "log-prob: min: "+z_ranges[2].toFixed(1)+", max: "+z_ranges[3].toFixed(1) + ",   range: "+(z_ranges[3]-z_ranges[2]).toFixed(1) + " bits";
  document.getElementById("zrange3").innerHTML = "log-prob: min: "+z_ranges[2].toFixed(1)+", max: "+z_ranges[3].toFixed(1) + ",   range: "+(z_ranges[3]-z_ranges[2]).toFixed(1) + " bits";

  epsilonDisplay.innerHTML = "Epsilon: " + targetEpsilon.toFixed(2);
}

function NPGtick() {

  updateStepCounter();

  draw();
  drawRegressionLines();
  drawPosteriorSamples();
}
