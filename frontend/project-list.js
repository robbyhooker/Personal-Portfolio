document.addEventListener("DOMContentLoaded", () => {
  const blogList = document.getElementById("blog-list");
  const postsArray = Object.keys(projects).map((key) => projects[key]);

  postsArray.forEach((post) => {
    const postElement = document.createElement("div");
    postElement.classList.add("post");

    let projectUrl = `project_display.html?id=post${post.id}`;
    if (post.id === 4) {
      projectUrl = "dashboard.html";
    }

    if (post.id === 7) {
      // Special case for project 7
      projectUrl =
        "window.open('https://public.tableau.com/app/profile/robert.hooker/viz/MockE-CommerceReport/KPIDash', '_blank')";
      postElement.innerHTML = `
        <div class="blogs-wrapper" onclick="${projectUrl};">
        <h3>${post.title}</h3>
        <p>${post.synopsis}</p>
        </div>
    `;
    } else {
      postElement.innerHTML = `
              <div class="blogs-wrapper" onclick="location.href='${projectUrl}';">
              <h3>${post.title}</h3>
              <p>${post.synopsis}</p>
              </div>
          `;
    }

    blogList.appendChild(postElement);
  });
});
