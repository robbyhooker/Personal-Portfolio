const menu = document.querySelector('.menu')
const sideBar = document.querySelector('.side-bar')
const backdrop = document.querySelector('.backdrop')
const closeBtn = document.querySelector('.close-btn')
const links = document.querySelectorAll('.side-menu li')

menu.onclick = function (){
    openSidebar()
}

closeBtn.onclick = function (){
    closeSidebar()
}

backdrop.onclick = function (){
    closeSidebar()
}

links.forEach(function (link) {
    link.onclick = function () {
        closeSidebar()
        setTimeout(() => {
        window.location.replace(link.getAttribute('data-link'))
      }, 500)
    }
})

function openSidebar(){
    backdrop.style.display = 'block'
    setTimeout(() => {
        sideBar.classList.add('open')
    }, 0);
}

function closeSidebar(){
    sideBar.classList.remove('open')
    setTimeout(() => {
        backdrop.style.display = 'none'
    }, 100);
}
document.addEventListener('DOMContentLoaded', function () {
    var textToType = document.getElementById('text-to-type').innerText;
    var words = textToType.split(/\s+/);

    var typingContainer = document.getElementById('typing-container');
    typingContainer.innerHTML = ''; // Clear the initial text

    typeText(words, 0, typingContainer);

    function typeText(words, index, container) {
      if (index < words.length-1) {
        container.innerText += words[index] + ' ';

        // Add a line break after each sentence, excluding the last period
        if (words[index].includes('.') && index < words.length - 1) {
          container.innerHTML += '<br>';
        }

        setTimeout(function () {
          typeText(words, index+1, container);
        }, 75); // Adjust the delay for typing speed
      }
    }
  });