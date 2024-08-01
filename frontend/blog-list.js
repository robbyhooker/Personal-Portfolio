// blog-list.js
document.addEventListener("DOMContentLoaded", () => {
  const blogList = document.getElementById("blog-list");
  const postsArray = Object.keys(posts).map((key) => posts[key]);

  // Sort posts by date (latest first)
  postsArray.sort((a, b) => new Date(b.date) - new Date(a.date));

  postsArray.forEach((post) => {
    const postElement = document.createElement("div");
    postElement.classList.add("post");

    postElement.innerHTML = `
            <div class="blogs-wrapper" onclick="location.href='blog-template.html?id=post${post.id}';">
            <h3>${post.title}</h3>
            <p>${post.synopsis}</p>
            <div class="author">${post.author} | ${post.date}</div>
            </div>
        `;

    blogList.appendChild(postElement);
  });
});
