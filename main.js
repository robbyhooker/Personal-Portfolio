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
    }, 500);
}
