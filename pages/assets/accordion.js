document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll(".accordion-header").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var body = btn.nextElementSibling;
      var isOpen = btn.classList.contains("active");

      if (isOpen) {
        btn.classList.remove("active");
        body.style.maxHeight = null;
        body.classList.remove("open-padding");
      } else {
        btn.classList.add("active");
        body.classList.add("open-padding");
        body.style.maxHeight = body.scrollHeight + "px";
      }
    });
  });
});
