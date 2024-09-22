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

let activeSpan = null; // To track the currently active span

function handleClick(spanElement, imageUrl) {
  const container = document.getElementById("background-container"); // or .top-section depending on your structure

  // Change the background image
  container.style.backgroundImage = `url(${imageUrl})`;

  // Remove 'active' class from the previously clicked span, if any
  if (activeSpan) {
    activeSpan.classList.remove("active");
  }

  // Add 'active' class to the clicked span
  spanElement.classList.add("active");

  // Update the active span tracker
  activeSpan = spanElement;

  // Re-trigger the zoom animation
  const topSection = document.querySelector(".top-section");

  topSection.classList.remove("zoom-animation"); // Remove the animation class to reset
  void topSection.offsetWidth; // Trigger reflow/repaint to restart the animation
  topSection.classList.add("zoom-animation"); // Reapply the animation class
}

// Set 'computers' as the default active span on page load
window.onload = function () {
  const defaultSpan = document.querySelector(".trigger.active");
  handleClick(defaultSpan, "assets/chip2.jpg");
};
/*let tiles = [];
let cols;
let rows;
let size = 55;
let colors;

function setup() {
  let canvasWidth = windowWidth * 0.85;
  let canvasHeight = windowHeight * 0.65; // Adjust as needed
  let canvas = createCanvas(canvasWidth, canvasHeight);
  canvas.remove();
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
  */
