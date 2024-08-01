$(document).ready(function () {
  $(".project-carousel").slick({
    dots: true,
    infinite: true,
    speed: 1000,
    slidesToShow: 1,
    centerMode: true,
    adaptiveHeight: true,
    autoplay: true,
    autoplaySpeed: 5000,
    arrows: false,
    draggable: false,
  });
});

// Pause autoplay when a modal is shown
$(".modal").on("shown.bs.modal", function () {
  $(".project-carousel").slick("slickPause");
});

// Resume autoplay when a modal is hidden
$(".modal").on("hidden.bs.modal", function () {
  $(".project-carousel").slick("slickPlay");
});

let tiles = [];
let cols;
let rows;
let size = 55;
let colors;

function setup() {
  let canvasWidth = windowWidth * 0.85;
  let canvasHeight = windowHeight * 0.65; // Adjust as needed
  let canvas = createCanvas(canvasWidth, canvasHeight);
  canvas.parent("p5-canvas");
  angleMode(DEGREES);
  colors = [
    color(14, 62, 36),
    color(115, 155, 208),
    color(219, 83, 30),
    color(255, 204, 0),
  ];
  cols = width / size;
  rows = height / size;
  for (let i = 0; i < cols; i++) {
    tiles[i] = [];
    for (let j = 0; j < rows; j++) {
      tiles[i][j] = new Tile(
        i * size,
        j * size,
        floor(random(2)),
        colors[floor(random(4))]
      );
    }
  }
}
function draw() {
  background(243, 245, 240);
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      tiles[i][j].display();
    }
  }
}
