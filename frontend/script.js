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

$(".modal").on("shown.bs.modal", function () {
  $(".project-carousel").slick("slickPause");
});

$(".modal").on("hidden.bs.modal", function () {
  $(".project-carousel").slick("slickPlay");
});

let activeSpan = null;

function handleClick(spanElement, imageUrl) {
  const container = document.getElementById("background-container");

  container.style.backgroundImage = `url(${imageUrl})`;

  if (activeSpan) {
    activeSpan.classList.remove("active");
  }

  spanElement.classList.add("active");

  activeSpan = spanElement;

  const topSection = document.querySelector(".top-section");

  topSection.classList.remove("zoom-animation");
  void topSection.offsetWidth;
  topSection.classList.add("zoom-animation");
}

window.onload = function () {
  const defaultSpan = document.querySelector(".trigger.active");
  handleClick(defaultSpan, "assets/chip2.jpg");
};
