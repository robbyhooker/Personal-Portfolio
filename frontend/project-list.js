document.addEventListener("DOMContentLoaded", () => {
  const blogList = document.getElementById("blog-list");
  const postsArray = Object.keys(projects).map((key) => projects[key]);

  // Sort posts by date (latest first)
  //postsArray.sort((a, b) => new Date(b.date) - new Date(a.date));

  postsArray.forEach((post) => {
    const postElement = document.createElement("div");
    postElement.classList.add("post");

    let projectUrl = `project_display.html?id=post${post.id}`;
    if (post.id === 4) {
      // Special case for project 4
      projectUrl = "dashboard.html";
    }

    if (post.id === 7) {
      // Special case for project 7
      projectUrl = "e-commerce.html";
    }

    postElement.innerHTML = `
            <div class="blogs-wrapper" onclick="location.href='${projectUrl}';">
            <h3>${post.title}</h3>
            <p>${post.synopsis}</p>
            </div>
        `;

    blogList.appendChild(postElement);
  });
});
